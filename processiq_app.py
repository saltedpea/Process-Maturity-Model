"""ProcessIQ process-maturity assessment app.

Run with:
    streamlit run processiq_app.py
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import streamlit as st


PASS_THRESHOLD = 80.0


@dataclass(frozen=True)
class Criterion:
    code: str
    name: str
    evidence: str
    weight: int


LEVELS: Tuple[Tuple[int, str], ...] = (
    (1, "Initial"),
    (2, "Managed"),
    (3, "Defined"),
    (4, "Measured"),
    (5, "Optimized"),
)

CRITERIA: Mapping[int, Tuple[Criterion, ...]] = {
    2: (
        Criterion("C2.1", "Process modelled per BPMN standard",
                  "BPMN-compliant model exists (system-generated checks)", 25),
        Criterion("C2.2", "Naming convention followed",
                  "Process name matches naming convention rules (PI team and Quality Manager check)", 25),
        Criterion("C2.3", "Ownership - Domain and E2E process level",
                  "Responsible is populated with an identified owner for Category and E2E scenario", 17),
        Criterion("C2.4", "Ownership - main process level",
                  "Process Owner field is populated", 17),
        Criterion("C2.5", "Ownership - subprocess level",
                  "Subprocess Owner field is populated", 16),
    ),
    3: (
        Criterion("C3.1", "RACI defined", "RACI facet populated", 16),
        Criterion("C3.2", "Inputs / Outputs defined", "Input/Output facet populated", 16),
        Criterion("C3.3", "Systems mapped", "Systems facet populated", 16),
        Criterion("C3.4", "Organizational units mapped", "Org Unit facet populated", 16),
        Criterion("C3.5", "KPIs defined", "KPI facet populated (definition only)", 16),
        Criterion("C3.6", "Standards linked", "Standards / policy facet populated", 10),
        Criterion("C3.7", "Risks & controls defined", "Risk & Control facet populated", 10),
    ),
    4: (
        Criterion("C4.1", "Process Cockpit active and KPIs connected to Celonis",
                  "Live KPI feed is active and KPIs refresh periodically", 50),
        Criterion("C4.2", "PAM active (where applicable)",
                  "PAM is linked; conformance, PIG and relevant KPIs are tracked", 30),
        Criterion("C4.3", "Process adherence tracked and deviations identified",
                  "Process Cockpit shows deviations and friction areas for improvement actions", 20),
    ),
    5: (
        Criterion("C5.1", "Improvement opportunities identified",
                  "At least one open CI initiative is linked in WAVE", 35),
        Criterion("C5.2", "Automation opportunities tracked",
                  "At least one automation opportunity is logged using Celonis Action Flow, Orchestration Flow or AI", 35),
        Criterion("C5.3", "Measurable business outcome realized",
                  "KPI value improved and is visible on Process Cockpit / PAM dashboards", 30),
    ),
}


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
    label = dict(LEVELS)[attained]
    return {"level": attained, "label": label, "scores": scores, "blocked_by": blocked_by}


def render_criterion_checklist(
    level: int, defaults: Mapping[str, bool] | None = None
) -> Dict[str, bool]:
    values: Dict[str, bool] = {}
    for criterion in CRITERIA[level]:
        default = bool(defaults.get(criterion.code, False)) if defaults else False
        values[criterion.code] = st.checkbox(
            f"{criterion.code} — {criterion.name} ({criterion.weight}%)",
            value=default,
            help=criterion.evidence,
            key=f"criterion_{criterion.code}",
        )
    return values


def csv_template() -> str:
    columns = ["process_name"] + [c.code for criteria in CRITERIA.values() for c in criteria]
    return pd.DataFrame([{column: "" for column in columns}]).to_csv(index=False)


def evaluate_batch(uploaded_file) -> pd.DataFrame:
    frame = pd.read_csv(uploaded_file)
    required = {"process_name"} | {c.code for criteria in CRITERIA.values() for c in criteria}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    rows = []
    for _, row in frame.iterrows():
        results = {
            code: str(row[code]).strip().lower() in {"1", "true", "yes", "y", "pass"}
            for code in required
            if code != "process_name"
        }
        assessment = assess_process(results)
        rows.append({
            "process_name": row["process_name"],
            "maturity_level": assessment["level"],
            "maturity_label": assessment["label"],
            **{f"{level}_score": score for level, score in assessment["scores"].items()},
        })
    return pd.DataFrame(rows)


def main() -> None:
    st.set_page_config(page_title="ProcessIQ", page_icon="📈", layout="wide")
    st.title("ProcessIQ")
    st.caption("Process maturity classification using evidence-based weighted gates")

    with st.sidebar:
        st.header("Assessment mode")
        mode = st.radio("Choose an assessment mode", ("Single process", "Batch CSV"))
        st.divider()
        st.markdown("**Classification rule**")
        st.write(
            "Every process starts at Level 1 - Initial. A level is attained only "
            "when its weighted score is at least 80% and all prior levels have passed."
        )
        st.download_button(
            "Download CSV template",
            data=csv_template(),
            file_name="processiq_assessment_template.csv",
            mime="text/csv",
        )

    if mode == "Batch CSV":
        st.subheader("Batch assessment")
        st.write("Use `true`, `yes`, `1`, `y`, `pass` for a met criterion; any other value is treated as not met.")
        uploaded = st.file_uploader("Upload assessment CSV", type="csv")
        if uploaded:
            try:
                output = evaluate_batch(uploaded)
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(str(exc))
            else:
                st.dataframe(output, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download scored results",
                    output.to_csv(index=False),
                    "processiq_scored_results.csv",
                    "text/csv",
                )
        return

    st.subheader("Single-process assessment")
    process_name = st.text_input("Process name", placeholder="e.g., Customer Onboarding")
    st.info("Level 1 - Initial is the default classification. Mark only criteria supported by current evidence.")

    results: Dict[str, bool] = {}
    for level in range(2, 6):
        with st.expander(f"Level {level} - {dict(LEVELS)[level]}", expanded=level == 2):
            results.update(render_criterion_checklist(level))
            current_score = score_level(CRITERIA[level], results)
            st.progress(int(current_score), text=f"Weighted score: {current_score:.0f}% / {PASS_THRESHOLD:.0f}% required")

    if st.button("Classify process", type="primary", disabled=not process_name.strip()):
        assessment = assess_process(results)
        level = int(assessment["level"])
        st.success(f"{process_name}: Level {level} - {assessment['label']}")

        score_columns = st.columns(4)
        for column, scored_level in zip(score_columns, range(2, 6)):
            score = assessment["scores"][scored_level]
            column.metric(f"Level {scored_level}", f"{score:.0f}%", "PASS" if score >= PASS_THRESHOLD else "Below 80%")

        unmet = [
            c.code for scored_level, criteria in CRITERIA.items()
            if scored_level > level for c in criteria if not results.get(c.code, False)
        ]
        if unmet:
            st.warning("Next actions: complete unmet criteria " + ", ".join(unmet))

    with st.expander("Methodology and criterion definitions"):
        for level, label in LEVELS[1:]:
            st.markdown(f"**Level {level} - {label}**")
            st.table(pd.DataFrame([
                {"Criterion": c.code, "Definition": c.name, "Evidence": c.evidence, "Weight": f"{c.weight}%"}
                for c in CRITERIA[level]
            ]))


if __name__ == "__main__":
    main()
