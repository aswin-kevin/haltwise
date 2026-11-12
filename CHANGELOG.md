# Changelog

## 0.1.0 (2026-11-12)

First public release.

- **Calibrated stopping:** `Monitor` with split-conformal calibration of the false-kill
  rate, one run per task, and an optional high-confidence mode (`confidence=`).
- **Calibrated `max_steps`:** `calibrate_step_cap` and `StepCountScorer`.
- **Scorers:**
  - `RuleScorer` (looping, error streaks, stalling);
  - `FeatureScorer` (behavioural plus step-sequence features, numpy logistic regression
    or any scikit-learn model);
  - `SelfReportScorer` (asks an LLM via the built-in Ollama client).
- **Policies:** `FirstCrossing` and `Density`.
- **Under drift:** `AdaptiveMonitor`, online recalibration with censoring-aware shadow
  runs, and `dry_run` to replay your logs.
- **Evaluation:** per-task false-kill rate, and AUROC at fixed steps and within tasks.
- **Budget replay:** `simulate_budget` compares continue / stop / restart under budgets.
- **Adapters:** OpenTelemetry GenAI (`spans_to_runs`, `HaltwiseSpanProcessor`), JSON
  Lines, and loaders for SWE-agent and SWE-smith trajectories.
