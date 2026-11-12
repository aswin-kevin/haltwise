from haltwise.synthetic import make_runs


def test_runs_are_labelled_and_mixed() -> None:
    runs = make_runs(200, seed=0)

    outcomes = {run.succeeded for run in runs}
    assert outcomes == {True, False}


def test_same_seed_gives_identical_runs() -> None:
    assert make_runs(20, seed=7) == make_runs(20, seed=7)


def test_failed_runs_are_longer_on_average() -> None:
    runs = make_runs(400, seed=1)
    failed = [len(run) for run in runs if run.failed]
    succeeded = [len(run) for run in runs if run.succeeded]

    assert sum(failed) / len(failed) > sum(succeeded) / len(succeeded)


def test_runs_carry_task_ids_for_grouped_splits() -> None:
    runs = make_runs(50, seed=0, tasks=10)

    assert len({run.task for run in runs}) == 10
