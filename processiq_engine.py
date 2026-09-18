"""
ProcessIQ scoring engine.

This module is a direct port of the governance-scoring, maturity-calculation,
readiness-gating and rule-triggering logic from the original ProcessIQ notebook,
restructured so it can be reused by both the Flask API and the ML training step.
"""
import ast
import numpy as np
import pandas as pd

DG_WEIGHTS = {
    "data_completeness": 20,
    "data_accuracy": 20,
    "data_consistency": 15,
    "data_timeliness": 10,
    "data_lineage": 15,
    "data_ownership": 10,
    "metadata_completeness": 5,
    "data_access_compliance": 5,
}

DG_LABELS = {
    "data_completeness": "Data completeness",
    "data_accuracy": "Data accuracy",
    "data_consistency": "Data consistency",
    "data_timeliness": "Data timeliness",
    "data_lineage": "Data lineage",
    "data_ownership": "Data ownership",
    "metadata_completeness": "Metadata completeness",
    "data_access_compliance": "Data access/compliance",
}

MATURITY_LABELS = {
    1: "Initial",
    2: "Managed",
    3: "Defined",
    4: "Measured",
    5: "Optimized",
}

RULE_DESCRIPTIONS = {
    "DG01": {"name": "Data Completeness Rule", "condition": "Data completeness < 80%",
             "description": "Checks whether all required process data is available.",
             "action": "Improve data completeness"},
    "DG02": {"name": "Data Accuracy Rule", "condition": "Data accuracy < 80%",
             "description": "Checks whether process data accurately represents the underlying business information.",
             "action": "Improve data accuracy"},
    "DG03": {"name": "Data Consistency Rule", "condition": "Data consistency < 80%",
             "description": "Checks whether process data remains consistent across records and systems.",
             "action": "Standardize inconsistent data"},
    "DG04": {"name": "Data Timeliness Rule", "condition": "Data timeliness < 75%",
             "description": "Checks whether process data is available within the required timeframe.",
             "action": "Improve data timeliness"},
    "DG05": {"name": "Data Lineage Rule", "condition": "Data lineage < 70%",
             "description": "Checks whether data can be traced from its source through transformations and usage.",
             "action": "Establish end-to-end data lineage"},
    "DG06": {"name": "Data Ownership Rule", "condition": "Data owner not assigned",
             "description": "Checks whether accountability for process data has been established.",
             "action": "Assign a data owner/steward"},
    "DG07": {"name": "Metadata Completeness Rule", "condition": "Metadata completeness < 80%",
             "description": "Checks whether sufficient metadata exists to understand and govern the data.",
             "action": "Complete required data metadata"},
    "DG08": {"name": "Data Access & Compliance Rule", "condition": "Data access/compliance < 80%",
             "description": "Checks whether appropriate data access and compliance controls are in place.",
             "action": "Review data access/compliance"},
    "P01": {"name": "Process Ownership Rule", "condition": "Process owner not assigned",
            "description": "Checks whether accountability for the process has been established.",
            "action": "Assign a process owner"},
    "P02": {"name": "Process Roles Rule", "condition": "Roles defined < 75%",
            "description": "Checks whether process roles and responsibilities are sufficiently defined.",
            "action": "Clarify process roles"},
    "P03": {"name": "BPMN Compliance Rule", "condition": "BPMN compliance < 75%",
            "description": "Checks whether the process model meets the required modelling standards.",
            "action": "Improve BPMN/process model quality"},
    "P04": {"name": "KPI Linkage Rule", "condition": "KPI linked = No",
            "description": "Checks whether measurable KPIs are linked to the process.",
            "action": "Define and link process KPIs"},
    "P05": {"name": "Execution Data Rule", "condition": "Event data unavailable",
            "description": "Checks whether sufficient execution data exists for process analysis.",
            "action": "Establish reliable process execution data"},
    "P06": {"name": "Deviation Monitoring Rule", "condition": "Deviation monitoring = No",
            "description": "Checks whether deviations from expected process behaviour are actively monitored.",
            "action": "Enable deviation monitoring"},
    "PI01": {"name": "High Exception Rule", "condition": "Exception rate > 25%",
             "description": "Detects excessive process exceptions that may affect process stability.",
             "action": "Investigate high-frequency exceptions"},
    "PI02": {"name": "High Rework Rule", "condition": "Rework rate > 20%",
             "description": "Detects excessive repeated work within process execution.",
             "action": "Investigate rework drivers"},
    "PI03": {"name": "Manual Handling Rule", "condition": "Manual handling > 65%",
             "description": "Identifies processes with significant manual effort that may have automation potential.",
             "action": "Evaluate automation of manual work"},
    "GATE01": {"name": "Data Governance Gate", "condition": "Data Governance Score < 60",
               "description": "Prevents advanced AI or automation recommendations when the underlying data foundation is not sufficiently governed.",
               "action": "Prioritize data-governance remediation"},
    "GATE02": {"name": "Optimized Readiness Gate", "condition": "One or more optimized readiness prerequisites failed",
               "description": "Prevents a process from being classified as Optimized until mandatory prerequisites are satisfied.",
               "action": "Satisfy optimized-readiness prerequisites"},
}

FEATURES = [
    "data_completeness", "data_accuracy", "data_consistency",
    "data_timeliness", "data_lineage", "data_ownership",
    "metadata_completeness", "data_access_compliance",
    "owner_assigned", "roles_defined", "hierarchy_alignment",
    "bpmn_compliance", "standardized_definition",
    "kpi_linked", "system_linked", "event_data_available",
    "deviation_monitoring", "execution_data_quality",
    "process_adherence", "execution_visibility",
    "process_standardization", "automation_opportunity",
    "decision_structure", "ai_data_readiness",
    "improvement_actions_defined", "monitoring_capability",
    "feedback_availability", "remediation_tracking",
    "cycle_time_hours", "waiting_time_hours", "exception_rate",
    "rework_rate", "manual_handling_rate", "handoff_count",
]


def dg_status(score):
    if score < 40:
        return "Critical"
    elif score < 60:
        return "Needs Improvement"
    elif score < 80:
        return "Managed"
    return "Strong"


def maturity_level_from_score(score):
    if score <= 20:
        return 1
    elif score <= 40:
        return 2
    elif score <= 60:
        return 3
    elif score <= 80:
        return 4
    return 5


def _rule_hits(r):
    """Evaluate every rule condition for a row and return (triggered_rules, gaps)."""
    hits = []
    gaps = []

    def check(rule_id, condition, gap_text):
        if condition:
            hits.append(rule_id)
            gaps.append(gap_text)

    check("DG01", r["data_completeness"] < 0.80, f"Data completeness at {r['data_completeness']*100:.0f}% (target 80%)")
    check("DG02", r["data_accuracy"] < 0.80, f"Data accuracy at {r['data_accuracy']*100:.0f}% (target 80%)")
    check("DG03", r["data_consistency"] < 0.80, f"Data consistency at {r['data_consistency']*100:.0f}% (target 80%)")
    check("DG04", r["data_timeliness"] < 0.75, f"Data timeliness at {r['data_timeliness']*100:.0f}% (target 75%)")
    check("DG05", r["data_lineage"] < 0.70, f"Data lineage at {r['data_lineage']*100:.0f}% (target 70%)")
    check("DG06", r["data_ownership"] < 0.80, f"Data ownership coverage at {r['data_ownership']*100:.0f}% (target 80%)")
    check("DG07", r["metadata_completeness"] < 0.80, f"Metadata completeness at {r['metadata_completeness']*100:.0f}% (target 80%)")
    check("DG08", r["data_access_compliance"] < 0.80, f"Data access/compliance at {r['data_access_compliance']*100:.0f}% (target 80%)")
    check("P01", r["owner_assigned"] == 0, "No process owner assigned")
    check("P02", r["roles_defined"] < 0.75, f"Process roles only {r['roles_defined']*100:.0f}% defined (target 75%)")
    check("P03", r["bpmn_compliance"] < 0.75, f"BPMN compliance at {r['bpmn_compliance']*100:.0f}% (target 75%)")
    check("P04", r["kpi_linked"] == 0, "No KPIs linked to this process")
    check("P05", r["event_data_available"] == 0, "No execution/event data available")
    check("P06", r["deviation_monitoring"] == 0, "Deviation monitoring not enabled")
    check("PI01", r["exception_rate"] > 0.25, f"Exception rate at {r['exception_rate']*100:.0f}% (threshold 25%)")
    check("PI02", r["rework_rate"] > 0.20, f"Rework rate at {r['rework_rate']*100:.0f}% (threshold 20%)")
    check("PI03", r["manual_handling_rate"] > 0.65, f"Manual handling at {r['manual_handling_rate']*100:.0f}% (threshold 65%)")
    check("GATE01", r["data_governance_score"] < 60, f"Data governance score {r['data_governance_score']:.1f} is below the 60-point gate")
    check("GATE02", not r["ai_readiness_gate"], "AI readiness prerequisites not fully satisfied")

    return hits, gaps


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full ProcessIQ scoring pipeline over a raw dataframe and
    return an enriched dataframe with governance, maturity, gates, gaps,
    triggered rules and recommendations."""
    df = df.copy()

    # 1. Data governance score
    df["data_governance_score"] = df.apply(
        lambda r: sum(r[c] * w for c, w in DG_WEIGHTS.items()), axis=1
    )
    df["data_governance_status"] = df["data_governance_score"].apply(dg_status)

    # Readiness gates
    df["pam_readiness_gate"] = (
        (df["data_governance_score"] >= 60)
        & (df["event_data_available"] == 1)
        & (df["execution_data_quality"] >= 0.75)
        & (df["process_adherence"] >= 0.75)
        & (df["deviation_monitoring"] == 1)
    )
    df["ai_readiness_gate"] = (
        (df["data_governance_score"] >= 80)
        & (df["data_completeness"] >= 0.80)
        & (df["data_lineage"] >= 0.70)
        & (df["metadata_completeness"] >= 0.80)
        & (df["ai_data_readiness"] >= 0.75)
        & (df["process_adherence"] >= 0.75)
        & (df["deviation_monitoring"] == 1)
    )

    # 2. Maturity calculation
    df["data_governance_points"] = df["data_governance_score"] / 100 * 20
    df["maturity_score"] = (
        df["data_governance_points"]
        + df["process_governance_score"]
        + df["process_measurement_score"]
        + df["pam_readiness_score"]
        + df["ai_readiness_score"]
        + df["continuous_improvement_score"]
    ).clip(0, 100).round(2)

    # Preserve an explicitly edited maturity level if the dataset already contains one.
    # This lets the dashboard and file reflect the updated values rather than overwriting them.
    if "maturity_level" not in df.columns:
        df["maturity_level"] = df["maturity_score"].apply(maturity_level_from_score)

    # A process cannot claim a level its gate prerequisites don't support unless
    # the dataset explicitly contains a manual override.
    if "maturity_level" in df.columns:
        df.loc[(df["maturity_level"] == 5) & (~df["ai_readiness_gate"]), "maturity_level"] = 4
        df.loc[(df["maturity_level"] == 4) & (~df["pam_readiness_gate"]), "maturity_level"] = 3

    df["maturity_label"] = df["maturity_level"].map(MATURITY_LABELS)

    # 3. Rules / gaps / recommendations (use dataset columns if already present,
    # otherwise derive them live from the rule engine)
    if not {"all_gaps", "triggered_rules", "recommendations"}.issubset(df.columns):
        rules_col, gaps_col, recs_col = [], [], []
        for _, r in df.iterrows():
            hits, gaps = _rule_hits(r)
            recs = list(dict.fromkeys(RULE_DESCRIPTIONS[h]["action"] for h in hits))
            rules_col.append("|".join(hits) if hits else "None")
            gaps_col.append(str(gaps) if gaps else "[]")
            recs_col.append("|".join(recs) if recs else "No immediate action required")
        df["triggered_rules"] = rules_col
        df["all_gaps"] = gaps_col
        df["recommendations"] = recs_col

    if "target_next_state" not in df.columns:
        target_map = {1: "Level 2 — Managed", 2: "Level 3 — Defined",
                      3: "Level 4 — Measured", 4: "Level 5 — Optimized"}
        df["target_next_state"] = df["maturity_level"].map(
            lambda lv: target_map.get(lv, "Maintain Optimized")
        )

    if "process_id" not in df.columns:
        df["process_id"] = [f"PRC-{i+1:03d}" for i in range(len(df))]
    if "department" not in df.columns:
        df["department"] = "Unassigned"

    return df


def parse_gaps(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return [g.strip() for g in value.split("|") if g.strip()]
    return []


def row_to_dict(r):
    triggered = [t for t in str(r["triggered_rules"]).split("|") if t and t != "None"]
    rules_detail = [
        {"code": t, **RULE_DESCRIPTIONS[t]} for t in triggered if t in RULE_DESCRIPTIONS
    ]
    recs = [x for x in str(r["recommendations"]).split("|") if x]

    return {
        "process_id": r["process_id"],
        "process_name": r["process_name"],
        "department": r.get("department", "Unassigned"),
        "governance": {
            "score": round(float(r["data_governance_score"]), 1),
            "status": r["data_governance_status"],
            "dimensions": [
                {"key": k, "label": DG_LABELS[k], "value": round(float(r[k]) * 100, 1)}
                for k in DG_WEIGHTS
            ],
        },
        "maturity": {
            "score": round(float(r["maturity_score"]), 1),
            "level": int(r["maturity_level"]),
            "label": r["maturity_label"],
            "target_next_state": r["target_next_state"],
        },
        "gates": {
            "pam_ready": bool(r["pam_readiness_gate"]),
            "ai_ready": bool(r["ai_readiness_gate"]),
        },
        "gaps": parse_gaps(r["all_gaps"]),
        "triggered_rules": rules_detail,
        "recommendations": recs,
        "indicators": {
            "cycle_time_hours": round(float(r["cycle_time_hours"]), 1),
            "waiting_time_hours": round(float(r["waiting_time_hours"]), 1),
            "exception_rate": round(float(r["exception_rate"]) * 100, 1),
            "rework_rate": round(float(r["rework_rate"]) * 100, 1),
            "manual_handling_rate": round(float(r["manual_handling_rate"]) * 100, 1),
            "handoff_count": int(r["handoff_count"]),
        },
    }
