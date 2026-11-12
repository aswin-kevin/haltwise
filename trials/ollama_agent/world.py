"""A small synthetic company and multi-step questions with exactly known answers.

Every task builds its own company from its seed, so tasks are independent, and repeated
attempts at a task face the same company. Look-alike names ("Ana Silva", "Ana Silvo")
and search results capped at five make agents mis-step, loop and fail realistically.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

DEPARTMENTS = ("Sales", "Research", "Support", "Finance", "Legal")
FIRST_NAMES = (
    "Ana", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jules",
    "Kofi", "Lena", "Mateo", "Nia", "Omar", "Priya", "Quinn", "Rosa", "Sven", "Tara",
)  # fmt: skip
SURNAMES = ("Silva", "Okafor", "Novak", "Haddad", "Lindqvist", "Moreau", "Tanaka", "Reyes")


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    department: str
    salary: int
    start_year: int
    manager_id: str | None


@dataclass(frozen=True)
class Department:
    name: str
    budget: int
    head_id: str


@dataclass(frozen=True)
class World:
    employees: dict[str, Employee]
    departments: dict[str, Department]

    def by_name(self, name: str) -> Employee:
        return next(e for e in self.employees.values() if e.name == name)


def _look_alike(surname: str, rng: random.Random) -> str:
    position = rng.randrange(1, len(surname))
    letter = rng.choice("aeiou" if surname[position] not in "aeiou" else "rstln")
    return surname[:position] + letter + surname[position + 1 :]


def make_world(seed: int, size: int = 60) -> World:
    """Build a company of ``size`` employees across the five departments."""
    rng = random.Random(seed)
    names: set[str] = set()
    while len(names) < size:
        surname = rng.choice(SURNAMES)
        if rng.random() < 0.3:
            surname = _look_alike(surname, rng)
        names.add(f"{rng.choice(FIRST_NAMES)} {surname}")
    ids = rng.sample(range(100, 1000), size)

    staff = []
    for index, name in enumerate(sorted(names)):
        department = DEPARTMENTS[index % len(DEPARTMENTS)]
        staff.append((f"E{ids[index]}", name, department))
    rng.shuffle(staff)

    heads = {dept: next(e for e in staff if e[2] == dept) for dept in DEPARTMENTS}
    ceo_department = rng.choice(DEPARTMENTS)
    employees: dict[str, Employee] = {}
    for employee_id, name, department in staff:
        head_id = heads[department][0]
        if employee_id == head_id:
            manager = None if department == ceo_department else heads[ceo_department][0]
        else:
            manager = head_id
        employees[employee_id] = Employee(
            employee_id,
            name,
            department,
            salary=rng.randrange(40, 200) * 1000,
            start_year=rng.randrange(2010, 2025),
            manager_id=manager,
        )
    departments = {
        dept: Department(dept, budget=rng.randrange(5, 50) * 100_000, head_id=heads[dept][0])
        for dept in DEPARTMENTS
    }
    return World(employees, departments)


@dataclass(frozen=True)
class Task:
    id: int
    question: str
    answer: str | int | float
    kind: str


@dataclass(frozen=True)
class _Context:
    """Everything a question builder may ask about."""

    world: World
    person: Employee
    other: Employee
    department: str
    members: list[Employee]
    head: Employee
    year: int


Question = tuple[str, str | int | float]


def _salary(c: _Context) -> Question:
    return f"What is the annual salary of {c.person.name}?", c.person.salary


def _head_of(c: _Context) -> Question:
    head = c.world.employees[c.world.departments[c.person.department].head_id]
    return f"What is the name of the head of the department {c.person.name} works in?", head.name


def _boss_of_head(c: _Context) -> Question:
    boss = c.world.employees[c.head.manager_id] if c.head.manager_id else c.head
    return (
        f"What is the salary of the manager of the head of the {c.department} department? "
        "(If that head has no manager, give the head's own salary.)",
        boss.salary,
    )


def _difference(c: _Context) -> Question:
    return (
        f"By how much does {c.person.name}'s annual salary differ from {c.other.name}'s? "
        "Give the absolute difference.",
        abs(c.person.salary - c.other.salary),
    )


def _total_salary(c: _Context) -> Question:
    return (
        f"What is the total annual salary of everyone in the {c.department} department?",
        sum(e.salary for e in c.members),
    )


def _started_before(c: _Context) -> Question:
    return (
        f"How many people in the {c.department} department started before {c.year}?",
        sum(e.start_year < c.year for e in c.members),
    )


def _total_before(c: _Context) -> Question:
    return (
        f"What is the total annual salary of the people in the {c.department} department "
        f"who started before {c.year}?",
        sum(e.salary for e in c.members if e.start_year < c.year),
    )


def _longest_serving(c: _Context) -> Question:
    veteran = min(c.members, key=lambda e: (e.start_year, -e.salary))
    return (
        f"Who in the {c.department} department has worked at the company longest (earliest "
        "start year)? If several started the same year, give the one with the highest salary.",
        veteran.name,
    )


def _team_average(c: _Context) -> Question:
    team = [e for e in c.world.employees.values() if e.manager_id == c.head.id]
    return (
        "What is the average annual salary, rounded to the nearest whole number, of the "
        f"people whose manager is the head of the {c.department} department?",
        round(sum(e.salary for e in team) / len(team)),
    )


_EASY = {"salary": _salary, "head_of": _head_of, "boss_of_head": _boss_of_head,
         "difference": _difference}  # fmt: skip
_HARD = {"total_salary": _total_salary, "started_before": _started_before,
         "total_before": _total_before, "longest_serving": _longest_serving,
         "team_average": _team_average}  # fmt: skip


def make_task(task_id: int) -> Task:
    """Return task ``task_id``: a question about its own company, with the right answer.

    About 30% of tasks are easy lookups; the rest need many lookups and arithmetic.
    """
    rng = random.Random(f"task-{task_id}")
    world = make_world(task_id)
    people = sorted(world.employees.values(), key=lambda e: e.id)
    person, other = rng.sample(people, 2)
    department = rng.choice(DEPARTMENTS)
    context = _Context(
        world=world,
        person=person,
        other=other,
        department=department,
        members=[e for e in people if e.department == department],
        head=world.employees[world.departments[department].head_id],
        year=rng.randrange(2014, 2022),
    )
    pool = _EASY if rng.random() < 0.3 else _HARD
    kind = rng.choice(sorted(pool))
    question, answer = pool[kind](context)
    return Task(task_id, question, answer, kind)


_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_EMPLOYEE_ID = re.compile(r"\bE\d+\b")
_YEAR_PHRASE = re.compile(r"\b(?:before|in|since|started|from)\s+(?:19|20)\d\d\b", re.IGNORECASE)


def _candidate_numbers(submitted: str) -> list[float]:
    """A bare number is the answer. Otherwise the first and last numbers, once employee ids
    and years are removed, are candidates: explanations put the result first or last."""
    text = submitted.replace(",", "").replace("$", "").strip()
    if _NUMBER.fullmatch(text):
        return [float(text)]
    text = _YEAR_PHRASE.sub(" ", _EMPLOYEE_ID.sub(" ", text))
    numbers = _NUMBER.findall(text)
    return [float(numbers[0]), float(numbers[-1])] if numbers else []


def _first_name_mentioned(submitted: str, names: list[str]) -> str | None:
    lowered = submitted.casefold()
    positions = [(lowered.find(name.casefold()), name) for name in names]
    found = [(position, name) for position, name in positions if position >= 0]
    return min(found)[1] if found else None


def check_answer(task: Task, submitted: str) -> bool:
    """Judge a submitted answer, tolerating explanations around it.

    Names: the first employee name mentioned must be the right one. Numbers: a bare
    number, or else the first or last number once employee ids ("E840") and years
    ("before 2018") are removed, must match within 0.5.
    """
    if isinstance(task.answer, str):
        names = [e.name for e in make_world(task.id).employees.values()]
        return _first_name_mentioned(submitted, names) == task.answer
    return any(abs(value - float(task.answer)) < 0.5 for value in _candidate_numbers(submitted))
