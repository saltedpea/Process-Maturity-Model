"""
Generates a synthetic-but-realistic ProcessIQ dataset so the dashboard
runs out of the box. Replace data/processiq_v2_corrected_dataset.csv
with a real export at any time -- the app will pick it up automatically
as long as the same column names are present.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

PROCESS_LIBRARY = [
    "Purchase-to-Pay", "Order-to-Cash", "Customer Onboarding", "Invoice Processing",
    "Employee Onboarding", "Incident Management", "Change Management", "Vendor Management",
    "Procurement Approval", "Accounts Payable", "Accounts Receivable", "Claims Processing",
    "Loan Origination", "KYC Verification", "Returns & Refunds", "Inventory Replenishment",
    "Production Scheduling", "Quality Inspection", "Shipment Tracking", "Contract Review",
    "Expense Reimbursement", "Payroll Processing", "IT Service Request", "Asset Provisioning",
    "Supplier Onboarding", "Demand Forecasting", "Warehouse Picking", "Customer Support Ticketing",
    "Marketing Campaign Approval", "Budget Planning", "Audit Preparation", "Regulatory Reporting",
    "Data Access Request", "Facility Maintenance", "Recruitment Pipeline", "Performance Review Cycle",
    "Field Service Dispatch", "Credit Risk Assessment", "Fraud Investigation", "Master Data Governance",
    "Change Order Management", "Sales Quote Approval", "Product Launch Readiness", "Capacity Planning",
    "Vendor Invoice Matching", "Customer Churn Response", "Returns Authorization", "Batch Release QA",
    "Safety Incident Reporting", "Digital Access Provisioning",
]

DEPARTMENTS = ["Finance", "Operations", "HR", "IT", "Procurement", "Supply Chain",
               "Customer Service", "Compliance", "Sales", "Manufacturing"]


def clip01(x):
    return float(np.clip(x, 0.0, 1.0))


rows = []
for i, name in enumerate(PROCESS_LIBRARY):
    # A hidden "quality tier" biases every metric for this process so the dataset
    # produces a believable spread across maturity levels rather than pure noise.
    tier = rng.choice([0, 1, 2, 3, 4], p=[0.16, 0.24, 0.28, 0.22, 0.10])
    base = 0.35 + tier * 0.14 + rng.normal(0, 0.05)

    def m(spread=0.12, floor=0.0, ceil=1.0):
        return float(np.clip(rng.normal(base, spread), floor, ceil))

    row = {
        "process_id": f"PRC-{i+1:03d}",
        "process_name": name,
        "department": DEPARTMENTS[i % len(DEPARTMENTS)],

        # Data governance dimensions (fractions 0-1)
        "data_completeness": m(),
        "data_accuracy": m(),
        "data_consistency": m(),
        "data_timeliness": m(0.15),
        "data_lineage": m(0.15),
        "data_ownership": m(0.15),
        "metadata_completeness": m(0.15),
        "data_access_compliance": m(0.1, ceil=1.0),

        # Process governance / measurement / readiness sub-scores (0-20 scale each,
        # combined with data-governance points out of 20 to build the 0-100 maturity score)
        "process_governance_score": float(np.clip(rng.normal(base * 16, 3), 0, 18)),
        "process_measurement_score": float(np.clip(rng.normal(base * 16, 3), 0, 18)),
        "pam_readiness_score": float(np.clip(rng.normal(base * 16, 3), 0, 18)),
        "ai_readiness_score": float(np.clip(rng.normal(base * 14, 3), 0, 16)),
        "continuous_improvement_score": float(np.clip(rng.normal(base * 16, 3), 0, 18)),

        # Process fundamentals
        "owner_assigned": int(rng.random() < clip01(0.3 + base * 0.7)),
        "roles_defined": m(0.15),
        "hierarchy_alignment": m(0.15),
        "bpmn_compliance": m(0.15),
        "standardized_definition": int(rng.random() < clip01(0.3 + base * 0.7)),
        "kpi_linked": int(rng.random() < clip01(0.25 + base * 0.75)),
        "system_linked": int(rng.random() < clip01(0.3 + base * 0.7)),

        # Execution / monitoring
        "event_data_available": int(rng.random() < clip01(0.25 + base * 0.75)),
        "deviation_monitoring": int(rng.random() < clip01(0.2 + base * 0.8)),
        "execution_data_quality": m(0.15),
        "process_adherence": m(0.15),
        "execution_visibility": m(0.15),
        "process_standardization": m(0.15),
        "automation_opportunity": clip01(1 - base + rng.normal(0, 0.15)),
        "decision_structure": m(0.15),
        "ai_data_readiness": m(0.15),

        # Continuous improvement
        "improvement_actions_defined": m(0.15),
        "monitoring_capability": m(0.15),
        "feedback_availability": m(0.15),
        "remediation_tracking": m(0.15),

        # Performance indicators
        "cycle_time_hours": float(np.clip(rng.normal(48 - base * 30, 10), 2, 240)),
        "waiting_time_hours": float(np.clip(rng.normal(20 - base * 14, 6), 0, 120)),
        "exception_rate": clip01(np.clip(rng.normal(0.35 - base * 0.3, 0.08), 0, 1)),
        "rework_rate": clip01(np.clip(rng.normal(0.28 - base * 0.24, 0.07), 0, 1)),
        "manual_handling_rate": clip01(np.clip(rng.normal(0.75 - base * 0.55, 0.12), 0, 1)),
        "handoff_count": int(np.clip(rng.normal(9 - base * 5, 2), 1, 20)),
    }
    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("data/processiq_v2_corrected_dataset.csv", index=False)
print(f"Wrote {len(df)} rows to data/processiq_v2_corrected_dataset.csv")
