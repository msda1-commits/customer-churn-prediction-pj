import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

st.title("🧑‍🤝‍🧑 Customer Churn Predictor")
st.caption("Moses solomon —  Churn Predictor ")
st.write(
    "Enter a customer's details below to estimate their probability of churning, "
    "using a  Forest model trained on the Telco customer dataset."
)

st.divider()

# ---------- Input form ----------
with st.form("churn_form"):
    st.subheader("Customer Profile")
    col1, col2 = st.columns(2)

    with col1:
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure_months = st.slider("Tenure (months)", min_value=0, max_value=72, value=12)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
        )
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    st.subheader("Internet Add-on Services")
    st.caption("Only applicable if the customer has internet service.")
    c3, c4 = st.columns(2)
    with c3:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    with c4:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # Total Charges: derive a sensible default from tenure * monthly, but let user override
    default_total = round(tenure_months * monthly_charges, 2)
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, value=float(default_total), step=1.0,
        help="Defaults to tenure × monthly charges; adjust if you have the actual figure."
    )

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)


def build_feature_row(inputs: dict, feature_columns: list) -> pd.DataFrame:
    """Reproduce the exact encoding pipeline used in training."""
    row = {col: 0 for col in feature_columns}

    # Binary yes/no fields (0/1)
    binary_map = {"Yes": 1, "No": 0}
    row["Senior Citizen"] = binary_map[inputs["senior_citizen"]]
    row["Partner"] = binary_map[inputs["partner"]]
    row["Dependents"] = binary_map[inputs["dependents"]]
    row["Paperless Billing"] = binary_map[inputs["paperless_billing"]]

    # Numeric
    row["Tenure Months"] = inputs["tenure_months"]
    row["Monthly Charges"] = inputs["monthly_charges"]
    row["Total Charges"] = inputs["total_charges"]

    # One-hot (drop_first=True during training, so baseline category = all-zero row)
    def set_onehot(prefix, value, baseline):
        if value != baseline:
            key = f"{prefix}_{value}"
            if key in row:
                row[key] = 1

    set_onehot("Multiple Lines", inputs["multiple_lines"], baseline="No")
    set_onehot("Internet Service", inputs["internet_service"], baseline="DSL")
    set_onehot("Online Security", inputs["online_security"], baseline="No")
    set_onehot("Online Backup", inputs["online_backup"], baseline="No")
    set_onehot("Device Protection", inputs["device_protection"], baseline="No")
    set_onehot("Tech Support", inputs["tech_support"], baseline="No")
    set_onehot("Streaming TV", inputs["streaming_tv"], baseline="No")
    set_onehot("Streaming Movies", inputs["streaming_movies"], baseline="No")
    set_onehot("Contract", inputs["contract"], baseline="Month-to-month")
    set_onehot("Payment Method", inputs["payment_method"], baseline="Bank transfer (automatic)")

    return pd.DataFrame([row], columns=feature_columns)


if submitted:
    inputs = dict(
        senior_citizen=senior_citizen, partner=partner, dependents=dependents,
        tenure_months=tenure_months, paperless_billing=paperless_billing,
        monthly_charges=monthly_charges, total_charges=total_charges,
        contract=contract, internet_service=internet_service,
        payment_method=payment_method, multiple_lines=multiple_lines,
        online_security=online_security, online_backup=online_backup,
        device_protection=device_protection, tech_support=tech_support,
        streaming_tv=streaming_tv, streaming_movies=streaming_movies,
    )

    X_row = build_feature_row(inputs, feature_columns)
    X_scaled = scaler.transform(X_row)

    proba = model.predict_proba(X_scaled)[0, 1]
    pred = model.predict(X_scaled)[0]

    st.divider()
    st.subheader("Prediction")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Churn Probability", f"{proba*100:.1f}%")
    with col_b:
        st.metric("Prediction", "❌ Likely to Churn" if pred == 1 else "✅ Likely to Stay")

    st.progress(min(max(proba, 0.0), 1.0))

    if proba >= 0.7:
        st.error("High churn risk — consider proactive retention outreach (discount, contract upgrade offer, or support call).")
    elif proba >= 0.4:
        st.warning("Moderate churn risk — monitor and consider a check-in or loyalty offer.")
    else:
        st.success("Low churn risk — customer appears stable.")

st.divider()
st.caption("Model: Random Forest · Trained on IBM Telco Customer Churn dataset · ROC-AUC ≈ 0.85")
