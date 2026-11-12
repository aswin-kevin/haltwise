"""haltwise: stop doomed LLM-agent runs early, with a guaranteed cap on false kills.

Quick start::

    from haltwise import Monitor, Step

    monitor = Monitor(false_kill_rate=0.05).fit(past_runs)
    watch = monitor.watch()
    for step in agent_steps:
        if watch.observe(step).should_stop:
            break
"""

from haltwise.budget import BudgetReport, Strategy, simulate_budget
from haltwise.errors import CalibrationError, HaltwiseError, NotCalibratedError
from haltwise.evaluation import EvaluationReport
from haltwise.io import load_runs, save_runs
from haltwise.monitor import Action, Decision, Monitor, Watch
from haltwise.online import AdaptiveMonitor, StreamReport
from haltwise.policies import Density, FirstCrossing, Policy
from haltwise.scorers import (
    FeatureScorer,
    RuleScorer,
    Scorer,
    StepCountScorer,
    calibrate_step_cap,
)
from haltwise.splits import split_runs
from haltwise.trace import Run, Step, StepKind, StepStatus, digest_args

__all__ = [
    "Action",
    "AdaptiveMonitor",
    "BudgetReport",
    "CalibrationError",
    "Decision",
    "Density",
    "EvaluationReport",
    "FeatureScorer",
    "FirstCrossing",
    "HaltwiseError",
    "Monitor",
    "NotCalibratedError",
    "Policy",
    "RuleScorer",
    "Run",
    "Scorer",
    "Step",
    "StepCountScorer",
    "StepKind",
    "StepStatus",
    "Strategy",
    "StreamReport",
    "Watch",
    "calibrate_step_cap",
    "digest_args",
    "load_runs",
    "save_runs",
    "simulate_budget",
    "split_runs",
]
__version__ = "0.1.0.dev0"
