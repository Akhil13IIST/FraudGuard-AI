import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Elliptic Fraud Detection",
    page_icon="🔍",
    layout="wide",
)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.title("Elliptic Fraud Detection System")
st.caption(
    "Illicit transaction classification using XGBoost "
    "and the Elliptic Bitcoin transaction dataset"
)

# --------------------------------------------------
# API connection
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )
    health_response.raise_for_status()
    health = health_response.json()

    st.sidebar.success("Prediction API connected")
    st.sidebar.write("Model: Elliptic XGBoost")
    st.sidebar.write(f"Features: {health['feature_count']}")
    st.sidebar.write(
        f"Decision threshold: {health['threshold']:.3f}"
    )

except requests.RequestException:
    st.sidebar.error("Prediction API unavailable")
    st.error(
        "Start the FastAPI server before using the dashboard."
    )
    st.stop()


# --------------------------------------------------
# Model performance
# --------------------------------------------------

st.header("Model Performance")

try:
    metrics_response = requests.get(
        f"{API_URL}/model/metrics",
        timeout=5,
    )
    metrics_response.raise_for_status()
    metrics = metrics_response.json()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Average Precision",
        f"{metrics['average_precision']:.4f}",
    )

    col2.metric(
        "ROC-AUC",
        f"{metrics['roc_auc']:.4f}",
    )

    col3.metric(
        "Precision",
        f"{metrics['precision']:.2%}",
    )

    col4.metric(
        "Recall",
        f"{metrics['recall']:.2%}",
    )

    with st.expander("Additional evaluation details"):
        st.write(f"F1-score: {metrics['f1']:.4f}")
        st.write(
            f"Decision threshold: {metrics['threshold']:.4f}"
        )

        st.write("Confusion matrix")

        cm = metrics["confusion_matrix"]

        st.dataframe(
            pd.DataFrame(
                cm,
                index=["Actual licit", "Actual illicit"],
                columns=["Predicted licit", "Predicted illicit"],
            ),
            use_container_width=True,
        )

    st.caption(
        "Metrics are from the held-out test evaluation. "
        "They are not live performance estimates."
    )

except requests.RequestException as error:
    st.error(f"Could not load model metrics: {error}")

# --------------------------------------------------
# Feature Group Ablation Study
# --------------------------------------------------

st.divider()
st.header("Feature Group Ablation Study")

st.caption(
    "Comparison of XGBoost models trained with different "
    "feature configurations on the same temporal test split. "
    "These experiments are separate from the production model."
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_COMPARISON_FILE = (
    PROJECT_ROOT
    / "reports"
    / "elliptic_feature_group_comparison.json"
)

if FEATURE_COMPARISON_FILE.exists():
    try:
        with open(FEATURE_COMPARISON_FILE, "r", encoding="utf-8") as file:
            feature_report = json.load(file)

        experiments = feature_report["experiments"]
        feature_df = pd.DataFrame(experiments)

        st.info(
            "All four experiments use the same chronological data splits. "
            "Decision thresholds were selected using validation F1, "
            "then evaluated on the held-out test set."
        )

        best_model = feature_report["best_by_average_precision"]

        best_result = next(
            item for item in experiments
            if item["experiment"] == best_model
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Test Average Precision",
            f"{best_result['average_precision']:.4f}",
        )

        col2.metric(
            "Best Performing Configuration",
            best_model.replace("_", " ").title(),
        )

        col3.metric(
            "Features Used",
            best_result["feature_count"],
        )

        st.subheader("Experimental Results")

        display_df = feature_df[
            [
                "experiment",
                "feature_group",
                "feature_count",
                "threshold",
                "validation_f1",
                "average_precision",
                "roc_auc",
                "precision",
                "recall",
                "f1",
            ]
        ].copy()

        display_df.columns = [
            "Experiment",
            "Feature Configuration",
            "Feature Count",
            "Threshold",
            "Validation F1",
            "Test AP",
            "Test ROC-AUC",
            "Test Precision",
            "Test Recall",
            "Test F1",
        ]

        st.dataframe(
            display_df.style.format({
                "Threshold": "{:.4f}",
                "Validation F1": "{:.4f}",
                "Test AP": "{:.4f}",
                "Test ROC-AUC": "{:.4f}",
                "Test Precision": "{:.4f}",
                "Test Recall": "{:.4f}",
                "Test F1": "{:.4f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Performance Comparison")

        chart_df = feature_df.set_index("experiment")[
            ["average_precision", "roc_auc", "f1"]
        ]

        chart_df.columns = [
            "Average Precision",
            "ROC-AUC",
            "F1-score",
        ]

        col1, col2 = st.columns(2)

        with col1:
            st.caption("Test Average Precision")
            st.bar_chart(chart_df[["Average Precision"]])

        with col2:
            st.caption("Test ROC-AUC")
            st.bar_chart(chart_df[["ROC-AUC"]])

        st.caption("Test F1-score")
        st.bar_chart(chart_df[["F1-score"]])

        st.subheader("Interpretation")

        local_result = next(
            item for item in experiments
            if item["experiment"] == "local_only"
        )

        baseline_result = next(
            item for item in experiments
            if item["experiment"] == "baseline"
        )

        st.write(
            f"- Local-only model uses {local_result['feature_count']} "
            f"transaction features and achieves test AP of "
            f"{local_result['average_precision']:.4f}."
        )

        st.write(
            f"- Full baseline uses {baseline_result['feature_count']} "
            f"features and achieves test AP of "
            f"{baseline_result['average_precision']:.4f}."
        )

        st.write(
            f"- Best configuration is **{best_model}**, "
            f"with test AP of {best_result['average_precision']:.4f}."
        )

        st.caption(
            "These are offline experimental results, not live "
            "production performance estimates. The comparison does "
            "not establish statistical significance or real-time "
            "feature availability."
        )

    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        st.error(
            f"Could not load feature-group comparison: {error}"
        )

else:
    st.warning(
        "Feature-group comparison report not found. "
        "Run scripts/compare_elliptic_feature_groups.py first."
    )



st.divider()


# --------------------------------------------------

# --------------------------------------------------
# Experimental model evaluation
# --------------------------------------------------

st.header("Experimental Model Evaluation")
st.caption(
    "Drift-ablation XGBoost vs GraphSAGE. "
    "These are research experiments, separate from the production API model."
)

try:
    comparison_response = requests.get(
        f"{API_URL}/model/comparison",
        timeout=10,
    )
    comparison_response.raise_for_status()

    comparison = comparison_response.json()

    selected = comparison["selected_blend"]
    individual = comparison["individual_test_results"]
    interpretation = comparison["interpretation"]

    st.info(
        "This experiment uses the 159-feature drift-ablation XGBoost "
        "artifact, not the 165-feature production XGBoost model."
    )

    st.subheader("Validation-based Model Selection")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Drift-ablation XGBoost Validation AP",
        f"{selected['validation_ap']:.4f}",
    )

    col2.metric(
        "GraphSAGE Test AP",
        f"{individual['graphsage']['average_precision']:.4f}",
    )

    col3.metric(
        "Selected XGBoost Weight",
        f"{selected['xgb_weight']:.0%}",
    )

    comparison_df = pd.DataFrame({
        "Model": [
            "Drift-ablation XGBoost (159 features)",
            "GraphSAGE",
        ],
        "Test Average Precision": [
            individual["xgboost"]["average_precision"],
            individual["graphsage"]["average_precision"],
        ],
        "Test ROC-AUC": [
            individual["xgboost"]["roc_auc"],
            individual["graphsage"]["roc_auc"],
        ],
    })

    st.subheader("Held-out Test Comparison")

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Average Precision")
        st.bar_chart(
            comparison_df.set_index("Model")[
                ["Test Average Precision"]
            ]
        )

    with col2:
        st.caption("ROC-AUC")
        st.bar_chart(
            comparison_df.set_index("Model")[
                ["Test ROC-AUC"]
            ]
        )

    st.subheader("Selected Experimental Configuration")

    col1, col2 = st.columns(2)

    col1.metric(
        "XGBoost Weight",
        f"{selected['xgb_weight']:.0%}",
    )

    col2.metric(
        "GraphSAGE Weight",
        f"{selected['gnn_weight']:.0%}",
    )

    st.info(interpretation["conclusion"])

    st.caption(interpretation["note"])

    if (
        selected["xgb_weight"] == 1.0
        and selected["gnn_weight"] == 0.0
    ):
        st.warning(
            "The selected configuration is effectively standalone "
            "drift-ablation XGBoost. GraphSAGE contributes no predictions "
            "to the selected blend."
        )

    with st.expander(
        "Selected experimental configuration: test evaluation"
    ):
        test_results = comparison["test_results"]

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Average Precision",
            f"{test_results['average_precision']:.4f}",
        )

        col2.metric(
            "Precision",
            f"{test_results['precision']:.2%}",
        )

        col3.metric(
            "Recall",
            f"{test_results['recall']:.2%}",
        )

        st.write("Confusion Matrix")

        st.dataframe(
            pd.DataFrame(
                test_results["confusion_matrix"],
                index=["Actual licit", "Actual illicit"],
                columns=["Predicted licit", "Predicted illicit"],
            ),
            use_container_width=True,
        )

except requests.RequestException as error:
    st.error(f"Could not load experimental model comparison: {error}")



st.divider()

# ============================================================
# MODEL MONITORING - EVIDENTLY AI
# ============================================================

st.header("Model Monitoring")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALERT_FILE = PROJECT_ROOT / "reports" / "elliptic_drift_alert.json"
DRIFT_REPORT = PROJECT_ROOT / "reports" / "elliptic_drift_report.html"

if ALERT_FILE.exists():
    with open(ALERT_FILE, "r", encoding="utf-8") as f:
        drift_alert = json.load(f)

    status = drift_alert["status"]
    percentage = drift_alert["drift_percentage"]

    if status == "CRITICAL":
        st.error(f"CRITICAL: Dataset drift detected — {percentage:.2f}%")
    elif status == "WARNING":
        st.warning(f"WARNING: Dataset drift detected — {percentage:.2f}%")
    else:
        st.success(f"NORMAL: Dataset drift — {percentage:.2f}%")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Drifted Features",
        f'{drift_alert["drifted_features"]}/{drift_alert["total_features"]}'
    )

    col2.metric(
        "Drift Percentage",
        f'{percentage:.2f}%'
    )

    col3.metric(
        "Monitoring Status",
        status
    )

    st.subheader("Recommended Action")
    st.info(drift_alert["recommended_action"])

    st.caption(
        "Automatic retraining executed: "
        + str(drift_alert["automatic_retraining_executed"])
    )

    if DRIFT_REPORT.exists():
        with open(DRIFT_REPORT, "rb") as f:
            st.download_button(
                label="Download Evidently Drift Report",
                data=f,
                file_name="fraudguard_drift_report.html",
                mime="text/html"
            )

else:
    st.warning(
        "No drift monitoring results found. "
        "Run the Evidently monitoring script first."
    )
# --------------------------------------------------
# MODEL EXPLAINABILITY - SHAP
# --------------------------------------------------

st.divider()
st.header("Model Explainability")
st.caption(
    "SHAP visualisations explain how input features influence "
    "the model's predictions. These are offline model explanations."
)

SHAP_GLOBAL = PROJECT_ROOT / "reports" / "shap_global_importance.png"
SHAP_SUMMARY = PROJECT_ROOT / "reports" / "shap_summary.png"
SHAP_TRANSACTION = PROJECT_ROOT / "reports" / "shap_transaction_explanation.png"

col1, col2 = st.columns(2)

with col1:
    st.subheader("Global Feature Importance")
    if SHAP_GLOBAL.exists():
        st.image(str(SHAP_GLOBAL), use_container_width=True)
    else:
        st.warning("Global SHAP plot not found.")

with col2:
    st.subheader("SHAP Summary Plot")
    if SHAP_SUMMARY.exists():
        st.image(str(SHAP_SUMMARY), use_container_width=True)
    else:
        st.warning("SHAP summary plot not found.")

with st.expander("Transaction-level SHAP explanation"):
    if SHAP_TRANSACTION.exists():
        st.image(str(SHAP_TRANSACTION), use_container_width=True)
    else:
        st.warning("Transaction SHAP plot not found.")


# --------------------------------------------------
# FEATURE DRIFT VISUALISATION
# --------------------------------------------------

st.divider()
st.subheader("Feature Drift Analysis")

DRIFT_CSV = PROJECT_ROOT / "reports" / "elliptic_feature_drift.csv"

if DRIFT_CSV.exists():
    try:
        drift_df = pd.read_csv(DRIFT_CSV)

        drift_chart = (
            drift_df[["feature", "train_vs_validation"]]
            .sort_values("train_vs_validation", ascending=False)
            .head(15)
            .set_index("feature")
        )

        st.caption(
            "Top 15 features by train-versus-validation drift score. "
            "Higher values indicate greater distributional difference."
        )

        st.bar_chart(drift_chart)

    except (OSError, KeyError, ValueError) as error:
        st.error(f"Could not load feature drift data: {error}")
else:
    st.warning("Feature drift CSV not found.")
# --------------------------------------------------
# Transaction investigation
# --------------------------------------------------
st.header("Transaction Investigation")

st.write(
    "Select a transaction from the held-out Elliptic test "
    "dataset to generate a prediction."
)

try:
    samples_response = requests.get(
        f"{API_URL}/samples",
        params={"limit": 1000},
        timeout=10,
    )
    samples_response.raise_for_status()

    samples = samples_response.json()["samples"]

    if samples:
        sample_options = {
            f"{item['tx_id']} | Time step {item['time_step']}": item
            for item in samples
        }

        selected_label = st.selectbox(
            "Select transaction",
            list(sample_options.keys()),
        )

        selected = sample_options[selected_label]

        st.write(
            f"**Transaction ID:** {selected['tx_id']}  \n"
            f"**Time step:** {selected['time_step']}"
        )

        if st.button(
            "Analyse Transaction",
            type="primary",
            use_container_width=True,
        ):
            try:
                response = requests.post(
                    f"{API_URL}/predict/sample/{selected['tx_id']}",
                    timeout=30,
                )
                response.raise_for_status()

                result = response.json()

                st.session_state["latest_prediction"] = result

            except requests.RequestException as error:
                st.error(f"Prediction failed: {error}")

        result = st.session_state.get("latest_prediction")

        if result:
            st.subheader("Investigation Result")

            probability = result["illicit_probability"]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Illicit Probability",
                f"{probability * 100:.4f}%",
            )

            col2.metric(
                "Prediction",
                (
                    "POTENTIALLY ILLICIT"
                    if result["predicted_illicit"]
                    else "LICIT CLASS PREDICTION"
                ),
            )

            col3.metric(
                "Risk Level",
                result["risk_level"],
            )

            st.progress(min(max(probability, 0.0), 1.0))

            if result["predicted_illicit"]:
                st.error(
                    "The model predicts that this transaction "
                    "belongs to the illicit class."
                )
            else:
                st.success(
                    "The model predicts that this transaction "
                    "belongs to the licit class."
                )

            st.caption(
                "This is a model classification, not a verified "
                "determination of criminal activity."
            )

            with st.expander("Prediction details"):
                st.json(result)

    else:
        st.warning("No test transactions are available.")

except requests.RequestException as error:
    st.error(f"Could not load transactions: {error}")


st.divider()

# --------------------------------------------------
# Analyst case review
# --------------------------------------------------

st.divider()
st.header("Analyst Case Review")

try:
    stats_response = requests.get(
        f"{API_URL}/investigations/statistics",
        timeout=10,
    )
    stats_response.raise_for_status()
    stats = stats_response.json()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Cases", stats["total_cases"])
    col2.metric("Open", stats["open_cases"])
    col3.metric("Under Review", stats["under_review"])
    col4.metric("Resolved", stats["resolved_cases"])

    cases_response = requests.get(
        f"{API_URL}/investigations",
        params={"limit": 1000},
        timeout=10,
    )
    cases_response.raise_for_status()

    cases = cases_response.json()["cases"]

    if cases:
        cases_df = pd.DataFrame(cases)

        st.subheader("Investigation Queue")

        st.dataframe(
            cases_df[
                [
                    "case_id",
                    "tx_id",
                    "time_step",
                    "illicit_probability",
                    "risk_level",
                    "status",
                    "analyst_verdict",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Review Investigation")

        case_options = {
            f"Case {case['case_id']} | "
            f"{case['tx_id']} | {case['status']}": case
            for case in cases
        }

        selected_label = st.selectbox(
            "Select investigation case",
            list(case_options.keys()),
        )

        selected_case = case_options[selected_label]

        st.write(
            f"**Transaction:** {selected_case['tx_id']}  \n"
            f"**Probability:** "
            f"{selected_case['illicit_probability']:.4%}  \n"
            f"**Risk:** {selected_case['risk_level']}"
        )

        with st.form("analyst_review_form"):

            new_status = st.selectbox(
                "Investigation status",
                ["Open", "Under Review", "Resolved"],
                index=[
                    "Open",
                    "Under Review",
                    "Resolved",
                ].index(selected_case["status"]),
            )

            new_verdict = st.selectbox(
                "Analyst verdict",
                [
                    "Unconfirmed",
                    "True Positive",
                    "False Positive",
                ],
                index=[
                    "Unconfirmed",
                    "True Positive",
                    "False Positive",
                ].index(selected_case["analyst_verdict"]),
            )

            new_notes = st.text_area(
                "Investigation notes",
                value=selected_case["notes"] or "",
                height=120,
            )

            submitted = st.form_submit_button(
                "Save Investigation",
                type="primary",
            )

            if submitted:
                try:
                    update_response = requests.put(
                        f"{API_URL}/investigations/"
                        f"{selected_case['case_id']}",
                        json={
                            "status": new_status,
                            "analyst_verdict": new_verdict,
                            "notes": new_notes,
                        },
                        timeout=10,
                    )
                    update_response.raise_for_status()

                    st.success("Investigation updated successfully.")
                    st.rerun()

                except requests.RequestException as error:
                    st.error(f"Could not update investigation: {error}")

    else:
        st.info(
            "No investigation cases yet. New transactions "
            "predicted as illicit will automatically enter "
            "the analyst investigation queue."
        )

except requests.RequestException as error:
    st.error(f"Could not load investigation cases: {error}")


# --------------------------------------------------
# Prediction history
# --------------------------------------------------

st.header("Investigation History")

if st.button("Refresh History"):
    st.rerun()

try:
    history_response = requests.get(
        f"{API_URL}/transactions",
        params={"limit": 1000},
        timeout=10,
    )
    history_response.raise_for_status()

    transactions = history_response.json()["transactions"]

    if transactions:
        history_df = pd.DataFrame(transactions)

        total = len(history_df)
        illicit = int(history_df["predicted_illicit"].sum())
        licit = total - illicit

        col1, col2, col3 = st.columns(3)

        col1.metric("Transactions Analysed", total)
        col2.metric("Predicted Illicit", illicit)
        col3.metric("Predicted Licit", licit)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        csv = history_df.to_csv(index=False)

        st.download_button(
            "Download Investigation History",
            data=csv,
            file_name="elliptic_investigation_history.csv",
            mime="text/csv",
        )

    else:
        st.info("No predictions recorded yet.")

except requests.RequestException as error:
    st.error(f"Could not retrieve history: {error}")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "Elliptic Fraud Detection System | "
    "XGBoost | FastAPI | Streamlit | SQLite"
)
