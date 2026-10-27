from pathlib import Path
import json
import pickle

import pandas as pd
import plotly.express as px
import streamlit as st

from src.maintenance import FEATURES, FAILURE_MODES, TARGET, LEAKAGE_COLUMNS

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "best_model.pkl"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
MODE_MODEL_PATH = ROOT / "models" / "failure_mode_model.pkl"
MODE_METADATA_PATH = ROOT / "models" / "failure_mode_metadata.json"
MODE_METRICS_PATH = ROOT / "models" / "failure_mode_metrics.csv"

MODE_NAMES = {
    "TWF": "Tool wear failure",
    "HDF": "Heat dissipation failure",
    "PWF": "Power failure",
    "OSF": "Overstrain failure",
    "RNF": "Random failure",
}

st.set_page_config(page_title="Predictive Maintenance", page_icon="⚙️", layout="wide")
st.title("AI-Based Predictive Maintenance")
st.caption("Educational machine failure screening using the UCI AI4I 2020 dataset")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_mode_model():
    with open(MODE_MODEL_PATH, "rb") as f:
        return pickle.load(f)

def predict_modes(model, row):
    probabilities = model.predict_proba(row)
    return {mode: float(probability[0, 1]) for mode, probability in zip(FAILURE_MODES, probabilities)}

def explain_probability(probability):
    if probability < 0.20:
        return "Lower estimated risk", "This estimate is relatively low within this model's output. It does not guarantee safe operation."
    if probability < 0.50:
        return "Elevated estimated risk", "Consider reviewing the operating conditions and maintenance schedule."
    return "Higher estimated risk", "Prioritize qualified inspection. This model output is not a safety determination."

try:
    data = load_data()
except Exception as exc:
    st.error(f"Could not load dataset at {DATA_PATH}: {exc}")
    st.stop()

page = st.sidebar.radio("Navigate", ["Dashboard", "Predict Machine Failure"])

if page == "Dashboard":
    st.header("Dataset overview")
    st.write("The model uses product type, air and process temperatures, rotational speed, torque, and tool wear. Failure-mode flags and identifiers are excluded.")
    metrics = st.columns(3)
    metrics[0].metric("Records", f"{len(data):,}")
    metrics[1].metric("Input columns", len(FEATURES))
    metrics[2].metric("Failure rate", f"{data[TARGET].mean():.2%}")

    left, right = st.columns(2)
    distribution = data[TARGET].value_counts().rename(index={0: "No failure", 1: "Failure"}).rename_axis("Outcome").reset_index(name="Records")
    left.plotly_chart(px.bar(distribution, x="Outcome", y="Records", color="Outcome", title="Failure distribution", color_discrete_sequence=["#337ab7", "#d9534f"]), width="stretch")

    comparison_path = ROOT / "models" / "model_comparison.csv"
    importance_path = ROOT / "models" / "feature_importance.csv"
    with right:
        st.subheader("Held-out test comparison")
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path, index_col=0)
            st.dataframe(comparison.style.format("{:.3f}"), width="stretch")
        else:
            st.info("Model comparison will appear after running the notebook.")
    st.subheader("Final model feature importance")
    if importance_path.exists():
        importance = pd.read_csv(importance_path)
        st.plotly_chart(px.bar(importance.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title="Random Forest impurity importance"), width="stretch")
    else:
        st.info("Feature importance will appear after running the notebook.")
    st.caption("Feature importance reflects model associations and does not establish causality.")

    st.subheader("Failure mode analysis")
    st.write("A separate multi-label model estimates five failure modes from the same sensor inputs. These are learned estimates, not copied dataset flags.")
    mode_counts = data[FAILURE_MODES].sum().sort_values(ascending=False).rename_axis("Mode").reset_index(name="Records")
    mode_left, mode_right = st.columns(2)
    mode_left.plotly_chart(px.bar(mode_counts, x="Mode", y="Records", color="Mode", title="Failure mode counts"), width="stretch")
    with mode_right:
        if MODE_METRICS_PATH.exists():
            mode_metrics = pd.read_csv(MODE_METRICS_PATH, index_col=0)
            st.dataframe(mode_metrics.round(3), width="stretch")
        else:
            st.info("Failure mode metrics will appear after running the notebook.")
    st.caption("Mode probabilities are independent and do not need to add to 100%. Rare modes such as TWF and RNF are harder to estimate.")

else:
    st.header("Estimate failure risk")
    st.warning("Educational estimate only. Do not use this model as a certified industrial safety assessment.")
    try:
        model = load_model()
        metadata = json.loads(METADATA_PATH.read_text())
        if metadata.get("feature_order") != FEATURES:
            raise ValueError("App feature schema does not match the saved model metadata.")
        if set(LEAKAGE_COLUMNS).intersection(metadata.get("features", [])):
            raise ValueError("Saved model metadata contains excluded failure-mode features.")
    except Exception as exc:
        st.error(f"Trained model is unavailable or incompatible: {exc}")
        st.info("Run all cells in notebooks/Predictive_Maintenance_ML.ipynb to train and export models/best_model.pkl.")
        st.stop()

    mode_model = None
    mode_error = None
    try:
        mode_model = load_mode_model()
        mode_metadata = json.loads(MODE_METADATA_PATH.read_text())
        if mode_metadata.get("feature_order") != FEATURES:
            raise ValueError("Failure mode feature order does not match the app.")
        if mode_metadata.get("target_order") != FAILURE_MODES:
            raise ValueError("Failure mode target order does not match the app.")
    except Exception as exc:
        mode_error = exc

    presets = {
        "Demo preset": {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1500, "Torque [Nm]": 45.0, "Tool wear [min]": 180},
        "High-risk preset": {"Type": "M", "Air temperature [K]": 300.0, "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1200, "Torque [Nm]": 70.0, "Tool wear [min]": 240},
    }
    selected = st.selectbox("Example values", [*presets.keys(), "Custom values"])
    preset = presets.get(selected, {})
    key_suffix = selected.replace(" ", "_").lower()
    with st.form("prediction_form"):
        st.caption("Enter sensor values in the dataset's units (unrestricted input).")
        product_type = st.selectbox("Product type", ["L", "M", "H"], index=["L", "M", "H"].index(preset.get("Type", "M")), key=f"type_{key_suffix}")
        c1, c2 = st.columns(2)
        with c1:
            air = st.number_input("Air temperature [K]", value=float(preset.get("Air temperature [K]", 300.0)), step=0.1, key=f"air_{key_suffix}")
            process = st.number_input("Process temperature [K]", value=float(preset.get("Process temperature [K]", 310.0)), step=0.1, key=f"process_{key_suffix}")
            rpm = st.number_input("Rotational speed [rpm]", value=int(preset.get("Rotational speed [rpm]", 1500)), step=10, key=f"rpm_{key_suffix}")
        with c2:
            torque = st.number_input("Torque [Nm]", value=float(preset.get("Torque [Nm]", 45.0)), step=0.1, key=f"torque_{key_suffix}")
            wear = st.number_input("Tool wear [min]", value=int(preset.get("Tool wear [min]", 180)), step=1, key=f"wear_{key_suffix}")
        submitted = st.form_submit_button("Estimate failure risk", type="primary")
    if submitted:
        row = pd.DataFrame([{
            "Type": product_type,
            "Air temperature [K]": air,
            "Process temperature [K]": process,
            "Rotational speed [rpm]": rpm,
            "Torque [Nm]": torque,
            "Tool wear [min]": wear,
        }], columns=FEATURES)
        try:
            prediction = int(model.predict(row)[0])
            probability = float(model.predict_proba(row)[0, 1]) if hasattr(model, "predict_proba") else None
            st.subheader("Overall failure risk")
            if probability is None:
                st.metric("Prediction", "Failure" if prediction else "No failure")
                st.info("The selected model does not provide a probability estimate.")
            else:
                title, detail = explain_probability(probability)
                st.metric("Predicted outcome", "Failure" if prediction else "No failure")
                st.metric("Estimated failure probability", f"{probability:.1%}")
                (st.error if prediction else st.success)(title)
                st.write(detail)

            st.subheader("Failure mode analysis")
            st.caption("Learned estimates from sensor inputs — not copied from dataset labels. Probabilities are independent and need not add to 100%.")
            if mode_model is None:
                st.info(f"Failure mode estimates are unavailable: {mode_error}")
            else:
                mode_probabilities = predict_modes(mode_model, row)

                # Identify the dominant failure type (highest probability)
                top_mode = max(mode_probabilities, key=mode_probabilities.get)
                top_prob = mode_probabilities[top_mode]

                # Show a clear "most likely failure type" callout
                if prediction and top_prob >= 0.10:
                    st.error(f"⚠️ Most likely failure type: **{MODE_NAMES[top_mode]} ({top_mode})** — {top_prob:.1%} estimated probability")
                elif top_prob >= 0.10:
                    st.warning(f"📊 Highest estimated mode: **{MODE_NAMES[top_mode]} ({top_mode})** — {top_prob:.1%}")
                else:
                    st.info("No single failure mode stands out. All mode estimates are below 10%.")

                st.write("")  # small spacer
                # Show all five modes with progress bars
                for mode, mode_probability in mode_probabilities.items():
                    col_label, col_bar = st.columns([2, 3])
                    with col_label:
                        st.write(f"**{MODE_NAMES[mode]}** ({mode})")
                        st.caption(f"{mode_probability:.1%}")
                    with col_bar:
                        st.progress(mode_probability)

            with st.expander("Input values and feature order"):
                st.dataframe(row, width="stretch")
                st.code(" → ".join(FEATURES))

            # Physical feasibility verification
            impossibility_reasons = []
            if air <= 0:
                impossibility_reasons.append("Air temperature is ≤ 0 K (violates absolute zero).")
            if process <= 0:
                impossibility_reasons.append("Process temperature is ≤ 0 K (violates absolute zero).")
            if process <= air:
                impossibility_reasons.append(f"Process temperature ({process:.1f} K) is ≤ Air temperature ({air:.1f} K); in active machining, cutting friction physically generates heat above ambient air temperature.")
            if rpm <= 0:
                impossibility_reasons.append(f"Rotational speed ({rpm} rpm) is ≤ 0; an operating milling spindle must have positive rotation.")
            if torque <= 0:
                impossibility_reasons.append(f"Torque ({torque:.1f} Nm) is ≤ 0; active cutting tool resistance cannot be non-positive.")
            if wear < 0:
                impossibility_reasons.append(f"Tool wear ({wear} min) is negative; accumulated cutting time cannot be less than zero.")

            if impossibility_reasons:
                st.warning("⚠️ **Physical feasibility notice:** This operating condition is physically not possible — " + " ".join(impossibility_reasons) + " The model output above should be regarded as physically invalid.")
        except Exception as exc:
            st.error(f"Could not calculate a prediction: {exc}")

st.divider()
st.caption("Dataset: UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset. Model outputs require validation against local equipment and operating costs.")
