# haltwise

**Stop doomed LLM-agent runs early, while capping how many good runs you kill.**

Agents loop, retry the same failing command, and burn tokens on runs that are already
lost. `max_steps` and timeouts are blunt: set them low and you kill good runs; set them
high and you pay for the bad ones.

haltwise learns from your past agent runs what a failing run looks like, watches live
runs step by step, and tells you when to stop. You choose a **false-kill rate**, e.g.
"stop at most 5% of runs that would have succeeded", and haltwise calibrates its stopping
threshold so that rate holds on runs from new tasks.

- **Works with any model and any framework.** It reads only what the agent *did* (tool
  calls, results, errors), never logits or hidden states. Message text is optional.
- **Light.** Pure Python plus numpy. No GPU, no PyTorch, no service to run.
- **Honest.** The guarantee, its assumptions and its limits are spelled out below, and
  `evaluate()` measures it on your own held-out runs.

> Status: alpha (`0.1.0.dev0`). The API may still change.

## Install

```bash
pip install haltwise                # core
pip install "haltwise[datasets]"    # + loaders for public trajectory datasets
```

## 30-second tour

```python
from haltwise import Monitor, split_runs
from haltwise.synthetic import make_runs  # stand-in for your own logged runs

past_runs, new_runs = split_runs(make_runs(2000), test_share=0.3)

monitor = Monitor(false_kill_rate=0.05).fit(past_runs)  # learn + calibrate
print(monitor.evaluate(new_runs))  # check the promise held

watch = monitor.watch()  # guard a live run
for step in agent_steps():
    decision = watch.observe(step)
    if decision.should_stop:
        print(decision.reason)  # "risk 0.97: 2 of the last 3 steps were above ..."
        break
```

Run it: `python examples/quickstart.py`.

## Describing what your agent does

haltwise needs a `Step` for each thing your agent does, and for past runs a label saying
whether the run succeeded:

```python
from haltwise import Run, Step

steps = [
    Step.llm("I'll run the tests first"),
    Step.tool_call("run_tests", {"path": "tests/"}),  # args are fingerprinted, not stored
    Step.error("2 failed"),
    Step.tool_call("edit_file", {"path": "app.py"}),
    Step.tool_result("edit_file", "ok"),
]
run = Run(steps, succeeded=False, task="issue-1234")
```

`task` matters: runs on the same task behave alike, so haltwise splits and calibrates by
task (see below). Save and load runs as JSON Lines with `save_runs` / `load_runs`.

## The simplest useful thing: a calibrated `max_steps`

```python
from haltwise import calibrate_step_cap

max_steps = calibrate_step_cap(past_runs, false_kill_rate=0.05)
```

This replaces a gut-feeling step limit with one that cuts off at most 5% of runs that
would have succeeded (on new tasks, same agent setup). On our benchmarks it captures most
of the savings available, so it is the baseline any smarter scorer must beat.
`StepCountScorer` is the same idea as a scorer, for use with `Monitor` or
`AdaptiveMonitor`.

## No labelled runs yet? Start with rules

```python
from haltwise import FirstCrossing, Monitor, RuleScorer

monitor = Monitor(RuleScorer(), policy=FirstCrossing(), threshold=0.5)
```

`RuleScorer` flags looping (the same call with the same arguments repeated), error
streaks, and stalling. `threshold=0.5` means "stop once any sign goes past its limit".
There is no guarantee without calibration, so once you have about 100 labelled tasks,
call `monitor.fit(runs)`.

## How it works

```
step → Scorer (risk of failure so far, 0–1) → Policy (is that enough to stop?) → Decision
                                                        ↑
                               threshold calibrated on past successful runs
```

| Piece | Built-in options |
|---|---|
| **Scorer**: how risky does this run look? | `FeatureScorer` (default; learned behavioural + step-sequence features), `RuleScorer` (no training), `SelfReportScorer` (asks an LLM) |
| **Policy**: when is the risk high enough to stop? | `Density(hits=2, window=3)` (default), `FirstCrossing()` |
| **Calibration**: which threshold meets your false-kill rate? | split conformal over whole runs, one run per task |

`FeatureScorer` accepts any scikit-learn-style classifier:
`FeatureScorer(model=HistGradientBoostingClassifier())`.

Save a fitted monitor with `monitor.save("monitor.json")` and load it with
`Monitor.load(...)`. Use `monitor.check(steps)` when you have a whole trace rather than
a live loop.

`SelfReportScorer` asks a language model every few steps how likely the run is to fail.
It needs no training data, and it works with a local Ollama server or Ollama Cloud (it
reads `OLLAMA_API_KEY` from the environment). In our tests it added no signal over
`FeatureScorer` for *early* stopping, and a little near the end of a run, at about 7 s
per check. It is best suited to a final "submit or retry?" decision:

```python
from haltwise.llm import ollama
from haltwise.scorers import SelfReportScorer

scorer = SelfReportScorer(ollama("gpt-oss:20b"), every=5)
```

## When your agent changes: `AdaptiveMonitor`

A calibrated threshold is only valid for the setup it was calibrated on. `AdaptiveMonitor`
keeps correcting the threshold from the outcomes of finished runs, so the false-kill rate
stays on target while you swap models, edit prompts or add tools:

```python
from haltwise import AdaptiveMonitor

adaptive = AdaptiveMonitor(Monitor().fit(past_runs))

watch = adaptive.watch()
for step in agent_steps():
    if watch.observe(step).should_stop:
        break
adaptive.record(watch, succeeded=run_passed)  # None if unknown
adaptive.save("adaptive.json")  # keep the learned threshold
```

How it works:

- A good run the monitor spared nudges the threshold down a little.
- A good run it would have stopped nudges it up a lot.
- The two forces balance when exactly α of good runs get stopped. This is online
  quantile tracking (adaptive conformal inference), so it needs no assumption that
  tomorrow looks like yesterday.

A stopped run never reveals whether it would have succeeded. So a random 30% of flagged
runs keep going (`Action.SHADOW`), and their outcomes are reweighted to stay unbiased.
That is the price of insurance. Replay your logs with `adaptive.dry_run(runs)` to see
what it would have done.

## OpenTelemetry

If your agent emits OpenTelemetry GenAI spans (`execute_tool`, `chat`, ...), haltwise can
read them directly. There is no OpenTelemetry dependency, and message content is not needed:

```python
from haltwise.adapters.otel import HaltwiseSpanProcessor, spans_to_runs

runs = spans_to_runs(exported_spans, outcomes={trace_id: succeeded})  # offline
monitor = Monitor().fit(runs)
provider.add_span_processor(HaltwiseSpanProcessor(monitor, on_stop=cancel))  # live
```

Enable capture of `gen_ai.tool.call.arguments` so haltwise can tell a repeated call from a
new one. See `examples/otel_live.py`.

## Stop, then what? `simulate_budget`

Stopping a doomed run only pays off if the saved budget is used well. Replay your own
logs to compare strategies before wiring one into production:

```python
from haltwise import simulate_budget

print(simulate_budget(monitor, held_out_runs, budgets=[50, 100, 200, None]))
```

It compares five strategies:

- **single attempt:** one attempt, never stopped;
- **give up:** one attempt, stopped early if flagged;
- **continue:** always finish, retry after natural failures;
- **restart:** stop doomed attempts and retry;
- **random restart:** stops just as often and as early as the monitor, but at random.

On 546 held-out SWE-agent tasks (about 22 logged attempts each, 20 random orders):

| Question | Answer on this agent |
|---|---|
| Does stop-and-restart solve *more* tasks under a fixed budget? | **No.** Restart ≈ continue at 50/100/200 steps (e.g. 18.0% vs 17.7% at 200), matching arXiv 2610.35971 |
| Does the monitor stop the *right* attempts? | **Slightly.** It beats random restarts by about 1 point of solves |
| Does stopping save money? | **Yes.** With no budget, restart reaches about the same solves (23.8% vs 24.2%) with **27% fewer steps**; for single-attempt agents, *give up* costs 0.3 points of solves and saves **24% of steps** |

In short, use haltwise to **spend less for the same results**, not to solve more. A
calibrated step cap did as well as the learned scorer here.

## Live trial on a real agent

A tool-using agent (`glm-5.3-flash` on Ollama Cloud) answered multi-step questions over a
synthetic company database, with real token counts. Under a 16-turn limit, on 30 new
tasks:

| Arm | Solved | Tokens/task |
|---|---|---|
| continue (haltwise watching in shadow mode) | 56.7% | 22,324 |
| haltwise stops + cold restart | 50.0% | 24,224 |
| haltwise stops + warm restart (retry told what went wrong) | 66.7% | 22,606 |

The warm-restart gain looked promising, so we pre-registered a confirmation on 100 new
tasks. **It did not replicate:** +2.0 points [−4.0, +8.0], and −7% tokens per task
(interval includes zero).

What the live trials show:

- **The guarantee held live:** 0 of 241 successful attempts would have been stopped,
  against a 10% target.
- **Savings were modest:** 7–13% of the tokens spent on failed attempts.
- **Restarting, cold or warm, did not solve more tasks.**
- **When nothing goes wrong, there is nothing to save.** With a generous turn limit, the
  agent solved 97% and haltwise had nothing to do.

The trial harness is in [`trials/ollama_agent/`](trials/ollama_agent/). It is fully
safe: read-only tools, and an arithmetic-only calculator, so no model output is ever
executed. Reproduce with `python -m trials.ollama_agent.collect` and
`python -m trials.ollama_agent.trial` (needs `OLLAMA_API_KEY`).

## The guarantee, precisely

For each held-out successful run, haltwise computes the largest threshold at which the
policy would have stopped it. The stopping threshold is the ⌈(n+1)(1−α)⌉-th smallest of
those values. That is split conformal prediction applied to whole trajectories. Taking
the whole trajectory into account makes the guarantee hold even though haltwise checks
after every step.

**Promise:** a successful run on a *new* task, from the same agent setup, is stopped with
probability at most α.

**Assumptions and limits:**

- **One run per task.** Runs on one task are near-copies, so haltwise calibrates on one
  random successful run per task, and `n` counts *tasks*. A 5% target needs at least
  19 tasks with a success. Below about 100 you get a warning, because the realised
  rate can wobble a lot.
- **Same setup.** The guarantee holds only if new runs resemble the calibration runs.
  **Change the model, prompt or tools, and you must recalibrate**, or use
  `AdaptiveMonitor`. The benchmark below shows the rate rising to 8–17% when a monitor
  is reused on a different model.
- **On average, unless you ask for confidence.** By default the bound holds on average
  over calibration sets. An unlucky calibration set can overshoot; one of our nebius
  splits reached 7.2%. For a promise about *your* threshold, use
  `Monitor(false_kill_rate=0.05, confidence=0.95)` (also `calibrate_step_cap(...,
  confidence=0.95)`): with 95% probability, at most 5% of good runs are stopped.
  It needs at least 59 calibration tasks and costs savings, less with more tasks:

  | | Default: good runs stopped, mean (worst) / steps saved | 95% confidence: good runs stopped, mean (worst) / steps saved |
  |---|---|---|
  | nebius, about 79 calibration tasks | 3.1% (7.2%) / 21.5% | 0.6% (1.9%) / 10.3% |
  | SWE-smith, about 500 calibration tasks | 4.1% (4.9%) / 12.2% | 2.6% (3.3%) / 8.7% |
- **Censoring.** Once you stop runs in production, you never learn whether they would
  have succeeded. `AdaptiveMonitor` handles this with shadow runs; with a plain
  `Monitor`, let a small random share of flagged runs finish if you want to recalibrate.

## Results on real agent runs

**The honest scorecard** (`benchmarks/audit.py`): every monitor is calibrated to a 5%
false-kill target on the *same* calibration tasks and evaluated on held-out tasks; means
over 5 task-grouped splits.

| | nebius SWE-agent (llama, 40k runs): failed-run steps saved | SWE-smith (Claude/GPT-4o, 6k runs): failed-run steps saved | false-kill rate per task |
|---|---|---|---|
| step cap (baseline) | 15.2% | 10.4% | 2.1% / 4.5% |
| `RuleScorer` | 19.2% | 4.1% | 2.5% / 5.1% |
| `FeatureScorer` | **21.5%** | **12.2%** | 3.1% / 4.1% |

What this means:

- **The guarantee holds** on both datasets, for every scorer.
- **The learned scorer beats a calibrated step cap, but not by a lot:** about +6 points
  of savings on the llama agent and about +2 on Claude-class agents. Much of what
  failure predictors detect is simply that failing runs run long ("Continue or
  Restart?", arXiv 2610.35971).
- **With little labelled data, the step cap catches up.** It needs no training, so it
  can calibrate on all your runs. Given every run, it saves 21% on nebius, about the
  same as `FeatureScorer`.
- **Early warnings are weak.** Comparing attempts on the *same* task, `FeatureScorer`'s
  AUROC is 0.51 at step 10 and 0.64 at step 40 on nebius. Genuine signal arrives late.
- **Savings depend on the agent:** 21–25% of failed-run steps for the llama agent,
  about 12% for Claude-class agents, which fail more quietly.

So start with `calibrate_step_cap`, and switch to `FeatureScorer` once you have a few
hundred labelled tasks and the audit shows it earns its keep on your agent. Earlier versions of this README quoted AUROC "by
progress"; that metric leaked run length and has been replaced.

**When the agent changes**, a fixed threshold breaks its promise and `AdaptiveMonitor`
restores it. Calibrated on llama-70b, then 20 random orders of streamed runs; false-kill
rate is mean (worst):

| Stream | Fixed threshold | `AdaptiveMonitor` |
|---|---|---|
| Same model (llama-70b) | 4.6% (4.6%), saves 31.8% | 3.5% (4.7%), saves 18.8% |
| New model (llama-8b) | **16.7%** (16.7%), saves 44.9% | **4.3%** (7.9%), saves 11.8% |
| Switch mid-stream | **10.2%** (11.2%), saves 39.2% | **4.0%** (7.0%), saves 14.6% |

The adaptive monitor is insurance: it keeps the promise under change, and it costs part
of the savings when nothing changes. It adapts in both directions. A monitor calibrated
on the *other* dataset stopped nothing on SWE-smith, and `AdaptiveMonitor` loosened it
to a useful threshold. Every number here can be reproduced with the scripts in
[`benchmarks/`](benchmarks/) (`audit.py`, `online.py`, `swe_agent.py`, `swe_smith.py`,
`budget.py`); public datasets download on first use.

## Prior work

haltwise packages and builds on recent research. It does not claim these ideas:

- Labelling every prefix with the final outcome and learning a risk score:
  **PrefixGuard** (arXiv 2605.06455), **StopLoss** (arXiv 2610.25525),
  **Fail-Fast, Restart-Smart** (arXiv 2608.03222).
- Calibrating an abort threshold on successful runs over the whole path: **StopLoss**;
  the max-over-checkpoints conformal stopping rule: **Statistical Early Stopping for
  Reasoning Models** (arXiv 2602.13935).
- Density-style stopping instead of stopping at the first warning: **OnTrack**
  (arXiv 2610.12375).
- Calibration that does not transfer across settings: **Reliability Audit of Early
  Outcome Prediction** (arXiv 2609.25647).
- One-sample-per-group conformal calibration for hierarchical data: Dunn, Wasserman &
  Ramdas (2022).
- Online quantile tracking for drifting data: adaptive conformal inference (Gibbs &
  Candès, 2021) and conformal PID control (Angelopoulos, Candès & Tibshirani, 2023).
- State-machine views of traces: **Automata from Agent Traces** (arXiv 2608.23670).
  Self-assessed failure: **BAGEN** (arXiv 2606.00198), **Last Step Matters**
  (arXiv 2608.29685).

What haltwise adds:
- a framework-independent, pip-installable runtime;
- task-aware calibration and reporting;
- censoring-aware online recalibration of abort thresholds, using shadow runs with
  inverse-propensity weighting (we found no prior implementation of this);
- one API and benchmark harness for comparing scorers.

## Development

```bash
git clone <this repository> && cd haltwise
uv sync --extra datasets            # creates ./.venv with dev tools
uv run pytest                       # tests
uv run ruff check && uv run mypy    # lint + strict type check
uv run python benchmarks/audit.py   # the honest scorecard (downloads public datasets)
```

## License

[Apache-2.0](LICENSE)
