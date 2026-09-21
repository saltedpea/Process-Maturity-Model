"""Deterministic ProcessIQ scoring engine.

The chatbot and UI may collect evidence, but they must not invent scores.
Classification always goes through assess_process().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd


PASS_THRESHOLD = 80.0


@dataclass(frozen=True)
class Criterion:
    code: str
    name: str
    evidence: str
    weight: int
    interview_prompt: str


LEVELS: Tuple[Tuple[int, str], ...] = (
    (1, "Initial"),
    (2, "Managed"),
    (3, "Defined"),
    (4, "Measured"),
    (5, "Optimized"),
)

CRITERIA: Mapping[int, Tuple[Criterion, ...]] = {
    2: (
        Criterion(
            "C2.1",
            "Process modelled per BPMN standard",
            "BPMN-compliant model exists (system-generated checks)",
            25,
            "Is there a BPMN-compliant model, and has it passed system or quality checks?",
        ),
        Criterion(
            "C2.2",
            "Naming convention followed",
            "Process name matches naming convention rules (PI team and Quality Manager check)",
            25,
            "Does the process name follow the naming convention, and has PI / Quality confirmed it?",
        ),
        Criterion(
            "C2.3",
            "Ownership - Domain and E2E process level",
            "Responsible is populated with an identified owner for Category and E2E scenario",
            17,
            "Is a responsible owner named for the domain / category and the end-to-end scenario?",
        ),
        Criterion(
            "C2.4",
            "Ownership - main process level",
            "Process Owner field is populated",
            17,
            "Is the Process Owner field populated with a named owner?",
        ),
        Criterion(
            "C2.5",
            "Ownership - subprocess level",
            "Subprocess Owner field is populated",
            16,
            "Is a Subprocess Owner populated for the relevant subprocesses?",
        ),
    ),
    3: (
        Criterion("C3.1", "RACI defined", "RACI facet populated", 16, "Is a RACI matrix populated for this process?"),
        Criterion("C3.2", "Inputs / Outputs defined", "Input/Output facet populated", 16, "Are inputs and outputs defined in the process record?"),
        Criterion("C3.3", "Systems mapped", "Systems facet populated", 16, "Are supporting systems mapped on the process record?"),
        Criterion("C3.4", "Organizational units mapped", "Org Unit facet populated", 16, "Are organizational units mapped to the process?"),
        Criterion("C3.5", "KPIs defined", "KPI facet populated (definition only)", 16, "Are KPIs defined on the process (definition is enough at this level)?"),
        Criterion("C3.6", "Standards linked", "Standards / policy facet populated", 10, "Are standards or policies linked to the process?"),
        Criterion("C3.7", "Risks & controls defined", "Risk & Control facet populated", 10, "Are risks and controls recorded on the process?"),
    ),
    4: (
        Criterion(
            "C4.1",
            "Process Cockpit active and KPIs connected to Celonis",
            "Live KPI feed is active and KPIs refresh periodically",
            50,
            "Is a Process Cockpit live, with KPIs connected to Celonis and refreshing periodically?",
        ),
        Criterion(
            "C4.2",
            "PAM active (where applicable)",
            "PAM is linked; conformance, PIG and relevant KPIs are tracked",
            30,
            "If PAM applies, is it linked with conformance, PIG, and relevant KPIs tracked?",
        ),
        Criterion(
            "C4.3",
            "Process adherence tracked and deviations identified",
            "Process Cockpit shows deviations and friction areas for improvement actions",
            20,
            "Does the cockpit show adherence, deviations, and friction areas?",
        ),
    ),
    5: (
        Criterion(
            "C5.1",
            "Improvement opportunities identified",
            "At least one open CI initiative is linked in WAVE",
            35,
            "Is at least one open continuous-improvement initiative linked in WAVE?",
        ),
        Criterion(
            "C5.2",
            "Automation opportunities tracked",
            "At least one automation opportunity is logged using Celonis Action Flow, Orchestration Flow or AI",
            35,
            "Is at least one automation opportunity logged in Celonis Action Flow, Orchestration Flow, or AI?",
        ),
        Criterion(
            "C5.3",
            "Measurable business outcome realized",
            "KPI value improved and is visible on Process Cockpit / PAM dashboards",
            30,
            "Has a KPI improved, and is that improvement visible on the cockpit or PAM dashboard?",
        ),
    ),
}

CRITERION_INDEX: Dict[str, Tuple[int, Criterion]] = {
    criterion.code: (level, criterion)
    for level, criteria in CRITERIA.items()
    for criterion in criteria
}

ALL_CODES: Tuple[str, ...] = tuple(CRITERION_INDEX)


def level_label(level: int) -> str:
    return dict(LEVELS)[level]


def score_level(criteria: Iterable[Criterion], results: Mapping[str, bool]) -> float:
    """Return the weighted score as a percentage in the range 0-100."""
    return round(sum(c.weight for c in criteria if results.get(c.code, False)), 2)


def assess_process(results: Mapping[str, bool]) -> Dict[str, object]:
    """Classify a process using sequential, prerequisite-aware level gates."""
    scores = {
        level: score_level(criteria, results)
        for level, criteria in CRITERIA.items()
    }
    attained = 1
    blocked_by: List[str] = []
    for level in range(2, 6):
        prior_level = level - 1
        if attained < prior_level:
            blocked_by.append(f"Level {level} requires Level {prior_level} first")
        elif scores[level] >= PASS_THRESHOLD:
            attained = level
        else:
            blocked_by.append(
                f"Level {level} score is {scores[level]:.0f}% (requires {PASS_THRESHOLD:.0f}%)"
            )
    return {
        "level": attained,
        "label": level_label(attained),
        "scores": scores,
        "blocked_by": blocked_by,
    }


def unmet_codes(results: Mapping[str, bool], start_after_level: int | None = None) -> List[str]:
    codes: List[str] = []
    for level, criteria in CRITERIA.items():
        if start_after_level is not None and level <= start_after_level:
            continue
        codes.extend(c.code for c in criteria if not results.get(c.code, False))
    return codes


def next_actions(results: Mapping[str, bool], attained_level: int) -> List[str]:
    actions: List[str] = []
    for level, criteria in CRITERIA.items():
        if level <= attained_level:
            continue
        for criterion in criteria:
            if not results.get(criterion.code, False):
                actions.append(
                    f"{criterion.code}: {criterion.name} — evidence needed: {criterion.evidence}"
                )
        break
    return actions


def catalog_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for level, criteria in CRITERIA.items():
        for criterion in criteria:
            rows.append({
                "Level": level,
                "Label": level_label(level),
                "Criterion": criterion.code,
                "Definition": criterion.name,
                "Evidence": criterion.evidence,
                "Weight": f"{criterion.weight}%",
            })
    return rows


def catalog_for_prompt() -> str:
    lines = [
        "ProcessIQ official catalog. Criteria are binary. Unstated evidence = not met.",
        f"Pass threshold: {PASS_THRESHOLD:.0f}% weighted score per level. Levels are sequential.",
    ]
    for level, criteria in CRITERIA.items():
        lines.append(f"\nLevel {level} - {level_label(level)}")
        for criterion in criteria:
            lines.append(
                f"- {criterion.code} ({criterion.weight}%): {criterion.name}. "
                f"Required evidence: {criterion.evidence}"
            )
    return "\n".join(lines)


def csv_template() -> str:
    columns = ["process_name"] + list(ALL_CODES)
    return pd.DataFrame([{column: "" for column in columns}]).to_csv(index=False)


def evaluate_batch(uploaded_file) -> pd.DataFrame:
    frame = pd.read_csv(uploaded_file)
    required = {"process_name"} | set(ALL_CODES)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    rows = []
    for _, row in frame.iterrows():
        results = {
            code: str(row[code]).strip().lower() in {"1", "true", "yes", "y", "pass"}
            for code in ALL_CODES
        }
        assessment = assess_process(results)
        rows.append({
            "process_name": row["process_name"],
            "maturity_level": assessment["level"],
            "maturity_label": assessment["label"],
            **{f"{level}_score": score for level, score in assessment["scores"].items()},
        })
    return pd.DataFrame(rows)
