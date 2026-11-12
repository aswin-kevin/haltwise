import pytest

from trials.ollama_agent.tools import Toolbox, safe_calculate
from trials.ollama_agent.world import Task, check_answer, make_task, make_world


class TestWorld:
    def test_same_seed_gives_the_same_world(self) -> None:
        assert make_world(7) == make_world(7)

    def test_every_department_head_works_in_that_department(self) -> None:
        world = make_world(3)
        for department in world.departments.values():
            assert world.employees[department.head_id].department == department.name

    def test_contains_look_alike_names(self) -> None:
        names = [e.name for e in make_world(1).employees.values()]
        surnames = [n.split()[1] for n in names]

        assert len(set(surnames)) < len(surnames)

    def test_tasks_are_deterministic_and_answerable(self) -> None:
        for task_id in range(40):
            task = make_task(task_id)
            assert task == make_task(task_id)
            assert check_answer(task, str(task.answer))


class TestCheckAnswer:
    def test_numbers_may_carry_formatting(self) -> None:
        task = make_task(0)
        if isinstance(task.answer, str):
            pytest.skip("task 0 has a name answer")

        assert check_answer(task, f"${task.answer:,.2f}")
        assert not check_answer(task, str(float(task.answer) + 7))

    def test_names_ignore_case_and_spacing(self) -> None:
        task = next(make_task(i) for i in range(200) if isinstance(make_task(i).answer, str))

        assert check_answer(task, f"  {str(task.answer).upper()} ")
        assert not check_answer(task, "Nobody Atall")


class TestSafeCalculate:
    @pytest.mark.parametrize(("expression", "value"), [("2 + 3 * 4", 14), ("(10 - 4) / 3", 2)])
    def test_does_arithmetic(self, expression: str, value: float) -> None:
        assert safe_calculate(expression) == value

    @pytest.mark.parametrize(
        "expression",
        ["__import__('os').system('ls')", "open('x')", "[1, 2]", "a + 1", "2 ** 999999"],
    )
    def test_refuses_anything_but_plain_arithmetic(self, expression: str) -> None:
        with pytest.raises(ValueError, match=r"allowed|large|arithmetic"):
            safe_calculate(expression)


class TestToolbox:
    def test_unknown_employee_is_an_error(self) -> None:
        observation, ok = Toolbox(make_world(0)).call("get_employee", {"id": "E999"})

        assert not ok
        assert "No employee" in observation

    def test_unknown_tool_is_an_error(self) -> None:
        _, ok = Toolbox(make_world(0)).call("delete_everything", {})

        assert not ok

    def test_search_finds_an_existing_employee(self) -> None:
        world = make_world(0)
        someone = next(iter(world.employees.values()))

        observation, ok = Toolbox(world).call("search_employees", {"query": someone.name})

        assert ok
        assert someone.id in observation


class TestHarderWorld:
    def test_department_listing_is_paginated(self) -> None:
        toolbox = Toolbox(make_world(0))

        first, _ = toolbox.call("list_department", {"name": "Sales"})
        second, _ = toolbox.call("list_department", {"name": "Sales", "page": 2})

        assert "page 1 of" in first
        assert first != second

    def test_all_question_kinds_occur(self) -> None:
        kinds = {make_task(i).kind for i in range(300)}

        assert {"total_before", "longest_serving", "team_average"} <= kinds


class TestRobustGrading:
    def _task_with(self, kind_is_name: bool) -> Task:
        return next(
            make_task(i) for i in range(500) if isinstance(make_task(i).answer, str) == kind_is_name
        )

    def test_a_correct_name_with_explanation_is_correct(self) -> None:
        task = self._task_with(kind_is_name=True)

        assert check_answer(task, f"{task.answer} (E840) - she started earliest, in 2013.")

    def test_the_first_name_mentioned_is_the_answer(self) -> None:
        task = self._task_with(kind_is_name=True)
        world = make_world(task.id)
        someone_else = next(e.name for e in world.employees.values() if e.name != task.answer)

        assert not check_answer(task, f"{someone_else}, not {task.answer}")

    def test_employee_ids_are_not_read_as_numbers(self) -> None:
        task = self._task_with(kind_is_name=False)

        assert check_answer(task, f"E840 has {task.answer}")

    def test_a_count_before_the_total_does_not_fool_the_grader(self) -> None:
        task = self._task_with(kind_is_name=False)

        assert check_answer(task, f"The total for all 12 employees is ${float(task.answer):,.0f}.")
        assert not check_answer(task, f"The total is {float(task.answer) + 1000:.0f}, not 3.")

    def test_an_answer_stated_before_its_explanation_is_correct(self) -> None:
        task = self._task_with(kind_is_name=False)

        reply = f"${float(task.answer):,.0f} - the employees are Priya ($62,000, 2017) and Dara."
        assert check_answer(task, reply)
