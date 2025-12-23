import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

df = pd.read_excel(r"C:\Users\ELCOT\Downloads\Employee-Attrition.xlsx")

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("👨‍💼 Employee Attrition Prediction System")

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Home",
    "📊 Model Performance",
    "🧠 Individual Prediction",
    "📈 Future Prediction"
])

with tab1:
    st.subheader("📌 Problem Statement")
    st.write("""
    The model that predicts whether an employee will leave or stay and  helps HR teams proactively identify employees at risk of leaving.
    """)

    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

with tab2:
    st.subheader("📈 Model Metrics")

    st.metric("Accuracy", "88%")
    st.metric("Precision (Attrition)", "74%")
    st.metric("Recall (Attrition)", "36%")
    st.metric("F1 Score", "0.49")

with tab3:
    st.subheader("🧠 Individual Employee Attrition Prediction")

    
    business_travel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently"]
    )

    department = st.selectbox(
        "Department",
        ["Sales", "Research & Development"]
    )

    education_field = st.selectbox(
        "Education Field",
        ["Life Sciences", "Medical", "Other"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Healthcare Representative",
            "Manufacturing Director"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single"]
    )

    overtime = st.selectbox(
        "OverTime",
        ["Yes", "No"]
    )

    age = st.number_input("Age", 18, 65, 30)
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 5000)
    job_satisfaction = st.slider("Job Satisfaction (1–4)", 1, 4, 3)
    work_life_balance = st.slider("Work Life Balance (1–4)", 1, 4, 3)
    total_working_years = st.number_input("Total Working Years", 0, 40, 10)

    if st.button("🔍 Predict Attrition"):
        input_df = pd.DataFrame([{
            "BusinessTravel": business_travel,
            "Department": department,
            "EducationField": education_field,
            "Gender": gender,
            "JobRole": job_role,
            "MaritalStatus": marital_status,
            "OverTime": overtime,
            "Age": age,
            "MonthlyIncome": monthly_income,
            "JobSatisfaction": job_satisfaction,
            "WorkLifeBalance": work_life_balance,
            "TotalWorkingYears": total_working_years
        }])

         # 🔹 Encode
        input_df = pd.get_dummies(input_df)

        # 🔹 Add missing columns
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0

        # 🔹 Correct order
        input_df = input_df[feature_names]

        # 🔹 Scale & Predict
        input_scaled = scaler.transform(input_df)
        prob = model.predict_proba(input_scaled)[0][1]

        decision = "YES (Employee likely to Leave)" if prob >= 0.5 else "NO (Employee likely to Stay)"

        st.success(decision)
        st.write("Attrition Probability:", round(prob, 2))

with tab4:
    st.subheader("📈 Future Employee Attrition Prediction")

    uploaded_file = st.file_uploader(
        "Upload Future Employee Dataset (CSV / Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            future_df = pd.read_csv(uploaded_file)
        else:
            future_df = pd.read_excel(uploaded_file)

        st.write("Dataset Preview")
        st.dataframe(future_df.head())

        # 🔹 Drop Attrition if exists
        if "Attrition" in future_df.columns:
            future_df = future_df.drop("Attrition", axis=1)

        # 🔹 Encode
        future_encoded = pd.get_dummies(future_df)

        # 🔹 Add missing columns
        for col in feature_names:
            if col not in future_encoded.columns:
                future_encoded[col] = 0

        # 🔹 Correct column order
        future_encoded = future_encoded[feature_names]

        if st.button("🔮 Predict Future Attrition"):
            future_scaled = scaler.transform(future_encoded)
            probs = model.predict_proba(future_scaled)[:, 1]

            result_df = future_df.copy()
            result_df["Attrition_Probability"] = probs
            result_df["Prediction"] = result_df["Attrition_Probability"].apply(
                lambda x: "Leave" if x >= 0.5 else "Stay"
            )

            st.subheader("📊 Prediction Results")
            st.dataframe(result_df)
