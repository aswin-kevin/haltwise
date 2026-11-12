"""haltwise in 30 seconds, on synthetic agent runs (no downloads, no API keys)."""

from haltwise import Monitor, split_runs
from haltwise.synthetic import make_runs

# 1. Past runs of your agent, each labelled succeeded=True/False.
past_runs, new_runs = split_runs(make_runs(2000, seed=0), test_share=0.3)

# 2. Learn what doomed runs look like; allow at most 5% of good runs to be stopped.
monitor = Monitor(false_kill_rate=0.05).fit(past_runs)

# 3. Check the promise on runs the monitor has never seen.
print(monitor.evaluate(new_runs))

# 4. Guard a live run, step by step.
doomed = next(run for run in new_runs if run.failed)
watch = monitor.watch()
for step in doomed.steps:
    decision = watch.observe(step)
    if decision.should_stop:
        print(f"\nStopped at step {decision.step + 1} of {len(doomed)}: {decision.reason}")
        break

monitor.save("monitor.json")  # reload later with Monitor.load("monitor.json")
