import pandas as pd

from processiq_engine import compute


def test_compute_preserves_existing_maturity_level():
    df = pd.DataFrame([
        {
            "data_completeness": 0.95,
            "data_accuracy": 0.92,
            "data_consistency": 0.94,
            "data_timeliness": 0.88,
            "data_lineage": 0.9,
            "data_ownership": 0.85,
            "metadata_completeness": 0.9,
            "data_access_compliance": 0.88,
            "owner_assigned": 1,
            "roles_defined": 0.9,
            "hierarchy_alignment": 0.85,
            "bpmn_compliance": 0.8,
            "standardized_definition": 0.9,
            "kpi_linked": 1,
            "system_linked": 1,
            "event_data_available": 1,
            "deviation_monitoring": 1,
            "execution_data_quality": 0.9,
            "process_adherence": 0.88,
            "execution_visibility": 0.85,
            "process_standardization": 0.9,
            "automation_opportunity": 0.75,
            "decision_structure": 0.8,
            "ai_data_readiness": 0.9,
            "improvement_actions_defined": 1,
            "monitoring_capability": 0.9,
            "feedback_availability": 0.85,
            "remediation_tracking": 0.8,
            "cycle_time_hours": 10,
            "waiting_time_hours": 5,
            "exception_rate": 0.05,
            "rework_rate": 0.08,
            "manual_handling_rate": 0.2,
            "handoff_count": 3,
            "process_governance_score": 88,
            "process_measurement_score": 82,
            "pam_readiness_score": 90,
            "ai_readiness_score": 85,
            "continuous_improvement_score": 80,
            "maturity_level": 3,
        }
    ])

    result = compute(df)

    assert result.loc[0, "maturity_level"] == 3
    assert result.loc[0, "maturity_label"] == "Defined"
