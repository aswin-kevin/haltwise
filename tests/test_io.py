from pathlib import Path

from haltwise import Run, Step, load_runs, save_runs


def test_save_then_load_returns_the_same_runs(tmp_path: Path) -> None:
    runs = [
        Run([Step.tool_call("ls", {"p": "."}), Step.error("boom")], succeeded=False, id="a"),
        Run([Step.llm("done")], succeeded=True, id="b", task="t"),
    ]
    path = tmp_path / "runs.jsonl"

    save_runs(runs, path)

    assert load_runs(path) == runs


def test_load_accepts_prefixguard_style_field_names(tmp_path: Path) -> None:
    path = tmp_path / "pg.jsonl"
    path.write_text(
        '{"trajectory_id": "x", "task_id": "t1", "final_success": true,'
        ' "steps": [{"kind": "llm"}]}\n'
    )

    (run,) = load_runs(path)

    assert (run.id, run.task, run.succeeded) == ("x", "t1", True)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "runs.jsonl"
    path.write_text('{"steps": []}\n\n{"steps": []}\n')

    assert len(load_runs(path)) == 2
