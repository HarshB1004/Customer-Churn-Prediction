import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("churn_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.set_page_config(page_title="Churn Prediction",page_icon="📊", layout="wide")

st.markdown(
    "<h1 style='text-align:center;'>Telecom Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)
st.write("Predict whether a customer is likely to churn using ML")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Details")

    tenure = st.slider(
        "Tenure (Months)",
        min_value=0,
        max_value=72,
        value=12
    )

    monthly_charges = st.number_input(
        "Monthly Charges ($)",
        min_value=0.0,
        max_value=200.0,
        value=50.0,
        step=0.5,
        format="%.2f"
    )

    # Auto-calculate total charges
    total_charges = round(tenure * monthly_charges, 2)

    st.number_input(
        "Total Charges ($)",
        value=total_charges,
        format="%.2f",
        disabled=True
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
with col2:
    st.subheader("Prediction Result")

    if st.button("Predict Churn"):
        input_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,  # ← calculated value
            'Contract': contract,
            'InternetService': internet_service,
            'PaymentMethod': payment_method
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(
                f"Customer is likely to churn\n\n"
                f"Churn Probability: {probability:.2%}"
            )
        else:
            st.success(
                f"Customer is likely to stay\n\n"
                f"Churn Probability: {probability:.2%}"
            )


