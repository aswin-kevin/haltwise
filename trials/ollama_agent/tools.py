"""The agent's tools: read-only lookups and a calculator that can only do arithmetic.

Nothing the model writes is ever executed. The calculator parses expressions into a
syntax tree and evaluates only numbers and + - * / with parentheses.
"""

from __future__ import annotations

import ast
import json
import operator
from collections.abc import Callable, Mapping
from typing import Any

from trials.ollama_agent.world import World

TOOL_DOCS = """\
search_employees(query): find employees whose name contains the text (at most 5 shown)
get_employee(id): full record of one employee (id like "E123")
list_department(name, page): ids and names in a department, 8 per page (page 1, 2, ...)
get_department(name): a department's budget and the id of its head
calculate(expression): arithmetic only, e.g. "1200 + 300 * 2"
submit(answer): give your final answer (ends the task)"""

_OPERATORS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_MAX_MAGNITUDE = 1e15
_PAGE_SIZE = 8


def safe_calculate(expression: str) -> float:
    """Evaluate plain arithmetic; raise ValueError for anything else."""

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                raise ValueError("only numbers and + - * / are allowed")
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            value = _OPERATORS[type(node.op)](evaluate(node.left), evaluate(node.right))
            if abs(value) > _MAX_MAGNITUDE:
                raise ValueError("number too large")
            return value
        raise ValueError("only numbers and + - * / are allowed")

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"not an arithmetic expression: {error.msg}") from None
    try:
        return evaluate(tree.body)
    except ZeroDivisionError:
        raise ValueError("division by zero") from None


class Toolbox:
    """Answers tool calls against one company. Returns ``(observation, ok)``."""

    def __init__(self, world: World) -> None:
        self._world = world

    def call(self, name: str, args: Mapping[str, Any]) -> tuple[str, bool]:
        handlers: dict[str, Callable[[Mapping[str, Any]], str]] = {
            "search_employees": self._search,
            "get_employee": self._employee,
            "list_department": self._list_department,
            "get_department": self._department,
            "calculate": lambda a: f"{safe_calculate(str(a.get('expression', ''))):g}",
        }
        handler = handlers.get(name)
        if handler is None:
            return f"Error: unknown tool {name!r}", False
        try:
            return handler(args), True
        except (KeyError, ValueError) as error:
            return f"Error: {error.args[0] if error.args else error}", False

    def _search(self, args: Mapping[str, Any]) -> str:
        query = str(args.get("query", "")).strip().casefold()
        if not query:
            raise ValueError("query must not be empty")
        matches = [e for e in self._world.employees.values() if query in e.name.casefold()]
        if not matches:
            return "No employees match."
        shown = sorted(matches, key=lambda e: e.id)[:5]
        lines = [f"{e.id}: {e.name} ({e.department})" for e in shown]
        if len(matches) > 5:
            lines.append(f"... and {len(matches) - 5} more; refine the query")
        return "\n".join(lines)

    def _employee(self, args: Mapping[str, Any]) -> str:
        employee_id = str(args.get("id", "")).strip().upper()
        employee = self._world.employees.get(employee_id)
        if employee is None:
            raise ValueError(f"No employee with id {employee_id!r}")
        return json.dumps(employee.__dict__)

    def _department_name(self, args: Mapping[str, Any]) -> str:
        name = str(args.get("name", "")).strip().title()
        if name not in self._world.departments:
            raise ValueError(f"Unknown department {name!r}")
        return name

    def _list_department(self, args: Mapping[str, Any]) -> str:
        name = self._department_name(args)
        members = sorted(
            (e for e in self._world.employees.values() if e.department == name), key=lambda e: e.id
        )
        pages = max(1, -(-len(members) // _PAGE_SIZE))
        try:
            page = int(args.get("page", 1))
        except (TypeError, ValueError):
            raise ValueError("page must be a whole number") from None
        if not 1 <= page <= pages:
            raise ValueError(f"page must be between 1 and {pages}")
        shown = members[(page - 1) * _PAGE_SIZE : page * _PAGE_SIZE]
        lines = [f"{e.id}: {e.name}" for e in shown]
        return f"{name}, page {page} of {pages}:\n" + "\n".join(lines)

    def _department(self, args: Mapping[str, Any]) -> str:
        department = self._world.departments[self._department_name(args)]
        return json.dumps({"budget": department.budget, "head_id": department.head_id})
