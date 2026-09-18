import os
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from processiq_engine import compute, row_to_dict, FEATURES, RULE_DESCRIPTIONS

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processiq_v2_corrected_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "processiq_v2_maturity_model.pkl")

app = Flask(__name__)

STATE = {"df": None, "model": None, "ml_report": None, "feature_importance": None}


def load_dataset(path=DATA_PATH):
    raw = pd.read_csv(path)
    df = compute(raw)
    STATE["df"] = df
    return df


def train_model(df):
    X = df[FEATURES]
    y = df["maturity_level"]

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=3,
        random_state=42, class_weight="balanced",
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    report = classification_report(y_test, pred, zero_division=0, output_dict=True)
    accuracy = accuracy_score(y_test, pred)

    importance = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True
    )[:10]

    STATE["model"] = model
    STATE["ml_report"] = {"accuracy": round(float(accuracy), 4), "report": report,
                           "test_size": len(y_test), "train_size": len(y_train)}
    STATE["feature_importance"] = [{"feature": f, "importance": round(float(v), 4)} for f, v in importance]

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


def bootstrap():
    df = load_dataset()
    train_model(df)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/overview")
def api_overview():
    df = STATE["df"]
    level_counts = df["maturity_level"].value_counts().sort_index()
    status_counts = df["data_governance_status"].value_counts()
    dept_counts = df.groupby("department")["maturity_score"].mean().round(1)

    dg_dim_avg = {
        k: round(float(df[k].mean()) * 100, 1)
        for k in ["data_completeness", "data_accuracy", "data_consistency", "data_timeliness",
                   "data_lineage", "data_ownership", "metadata_completeness", "data_access_compliance"]
    }

    return jsonify({
        "total_processes": int(len(df)),
        "avg_maturity_score": round(float(df["maturity_score"].mean()), 1),
        "avg_governance_score": round(float(df["data_governance_score"].mean()), 1),
        "pam_ready_pct": round(float(df["pam_readiness_gate"].mean()) * 100, 1),
        "ai_ready_pct": round(float(df["ai_readiness_gate"].mean()) * 100, 1),
        "maturity_level_counts": {int(k): int(v) for k, v in level_counts.items()},
        "governance_status_counts": {k: int(v) for k, v in status_counts.items()},
        "avg_maturity_by_department": dept_counts.to_dict(),
        "governance_dimension_avg": dg_dim_avg,
        "top_gaps": _top_gaps(df),
    })


def _top_gaps(df, limit=8):
    from collections import Counter
    counter = Counter()
    for rules in df["triggered_rules"]:
        for code in str(rules).split("|"):
            if code and code != "None" and code in RULE_DESCRIPTIONS:
                counter[code] += 1
    ranked = counter.most_common(limit)
    return [
        {"code": code, "name": RULE_DESCRIPTIONS[code]["name"], "count": count,
         "action": RULE_DESCRIPTIONS[code]["action"]}
        for code, count in ranked
    ]


@app.route("/api/processes")
def api_processes():
    df = STATE["df"]
    search = request.args.get("q", "").lower().strip()
    level = request.args.get("level")
    status = request.args.get("status")

    view = df
    if search:
        view = view[view["process_name"].str.lower().str.contains(search)
                     | view["department"].str.lower().str.contains(search)]
    if level:
        view = view[view["maturity_level"] == int(level)]
    if status:
        view = view[view["data_governance_status"] == status]

    rows = []
    for _, r in view.iterrows():
        rows.append({
            "process_id": r["process_id"],
            "process_name": r["process_name"],
            "department": r.get("department", "Unassigned"),
            "maturity_score": round(float(r["maturity_score"]), 1),
            "maturity_level": int(r["maturity_level"]),
            "maturity_label": r["maturity_label"],
            "governance_score": round(float(r["data_governance_score"]), 1),
            "governance_status": r["data_governance_status"],
            "pam_ready": bool(r["pam_readiness_gate"]),
            "ai_ready": bool(r["ai_readiness_gate"]),
            "exception_rate": round(float(r["exception_rate"]) * 100, 1),
        })
    rows.sort(key=lambda x: x["maturity_score"], reverse=True)
    return jsonify({"count": len(rows), "processes": rows})


@app.route("/api/processes/<process_id>")
def api_process_detail(process_id):
    df = STATE["df"]
    match = df[df["process_id"] == process_id]
    if match.empty:
        return jsonify({"error": "process not found"}), 404
    return jsonify(row_to_dict(match.iloc[0]))


@app.route("/api/ml")
def api_ml():
    return jsonify({
        "metrics": STATE["ml_report"],
        "feature_importance": STATE["feature_importance"],
        "maturity_labels": {
            "1": "Initial", "2": "Managed", "3": "Defined",
            "4": "Measured", "5": "Optimized",
        },
    })


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "no file uploaded"}), 400
    file = request.files["file"]
    try:
        raw = pd.read_csv(io.StringIO(file.stream.read().decode("utf-8")))
        df = compute(raw)
        STATE["df"] = df
        train_model(df)
        return jsonify({"ok": True, "rows": len(df)})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 400


@app.route("/api/reset", methods=["POST"])
def api_reset():
    bootstrap()
    return jsonify({"ok": True, "rows": len(STATE["df"])})


if __name__ == "__main__":
    bootstrap()
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        debug=False,
    )
