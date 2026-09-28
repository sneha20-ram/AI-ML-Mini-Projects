import streamlit as st
import sys
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Explainable Credit Risk System",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("💳 Explainable Credit Risk Decision System")

st.write(
    "Educational ML system for predicting credit risk "
    "using an XGBoost model."
)

st.divider()


# ============================================================
# APPLICANT INFORMATION
# ============================================================

st.header("👤 Applicant Information")

col1, col2 = st.columns(2)


with col1:

    attribute1 = st.selectbox(
        "Attribute 1",
        ["A11", "A12", "A13", "A14"]
    )

    attribute2 = st.number_input(
        "Attribute 2 — Loan Duration (months)",
        min_value=4,
        max_value=72,
        value=6
    )

    attribute3 = st.selectbox(
        "Attribute 3",
        ["A30", "A31", "A32", "A33", "A34"]
    )

    attribute4 = st.selectbox(
        "Attribute 4",
        ["A40", "A41", "A410", "A42", "A43",
         "A44", "A45", "A46", "A48", "A49"]
    )

    attribute5 = st.number_input(
        "Attribute 5 — Credit Amount",
        min_value=250,
        max_value=18424,
        value=1169
    )

    attribute6 = st.selectbox(
        "Attribute 6",
        ["A61", "A62", "A63", "A64", "A65"]
    )

    attribute7 = st.selectbox(
        "Attribute 7",
        ["A71", "A72", "A73", "A74", "A75"]
    )

    attribute8 = st.selectbox(
        "Attribute 8",
        [1, 2, 3, 4]
    )

    attribute9 = st.selectbox(
        "Attribute 9",
        ["A91", "A92", "A93", "A94"]
    )

    attribute10 = st.selectbox(
        "Attribute 10",
        ["A101", "A102", "A103"]
    )


with col2:

    attribute11 = st.selectbox(
        "Attribute 11",
        [1, 2, 3, 4]
    )

    attribute12 = st.selectbox(
        "Attribute 12",
        ["A121", "A122", "A123", "A124"]
    )

    attribute13 = st.number_input(
        "Attribute 13 — Age",
        min_value=19,
        max_value=75,
        value=67
    )

    attribute14 = st.selectbox(
        "Attribute 14",
        ["A141", "A142", "A143"]
    )

    attribute15 = st.selectbox(
        "Attribute 15",
        ["A151", "A152", "A153"]
    )

    attribute16 = st.selectbox(
        "Attribute 16",
        [1, 2, 3, 4]
    )

    attribute17 = st.selectbox(
        "Attribute 17",
        ["A171", "A172", "A173", "A174"]
    )

    attribute18 = st.selectbox(
        "Attribute 18",
        [1, 2]
    )

    attribute19 = st.selectbox(
        "Attribute 19",
        ["A191", "A192"]
    )

    attribute20 = st.selectbox(
        "Attribute 20",
        ["A201", "A202"]
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

st.header("🔮 Credit Risk Prediction")


if st.button("Predict Credit Risk", type="primary"):

    try:

        from src.predict import predict_credit_risk

        input_data = {

            "Attribute1": attribute1,
            "Attribute2": attribute2,
            "Attribute3": attribute3,
            "Attribute4": attribute4,
            "Attribute5": attribute5,
            "Attribute6": attribute6,
            "Attribute7": attribute7,
            "Attribute8": attribute8,
            "Attribute9": attribute9,
            "Attribute10": attribute10,
            "Attribute11": attribute11,
            "Attribute12": attribute12,
            "Attribute13": attribute13,
            "Attribute14": attribute14,
            "Attribute15": attribute15,
            "Attribute16": attribute16,
            "Attribute17": attribute17,
            "Attribute18": attribute18,
            "Attribute19": attribute19,
            "Attribute20": attribute20

        }

        prediction, probability = predict_credit_risk(input_data)

        st.subheader("Prediction Result")

        if prediction == 0:

            st.success("🟢 Lower Risk")

        else:

            st.error("🔴 Higher Risk")

        st.metric(
            "Risk Probability",
            f"{probability * 100:.2f}%"
        )

        # Store values for What-If Analysis
        st.session_state["prediction"] = prediction
        st.session_state["probability"] = probability
        st.session_state["input_data"] = input_data

    except Exception as e:

        st.error(f"Prediction failed: {e}")


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

if "input_data" in st.session_state:

    st.divider()

    st.header("🔍 What-If Analysis")

    st.write(
        "Change the loan duration or credit amount "
        "to explore how the predicted risk changes."
    )

    original_data = st.session_state["input_data"]

    col1, col2 = st.columns(2)

    with col1:

        what_if_duration = st.slider(
            "Loan Duration (months)",
            min_value=4,
            max_value=72,
            value=int(original_data["Attribute2"])
        )

    with col2:

        what_if_credit = st.slider(
            "Credit Amount",
            min_value=250,
            max_value=18424,
            value=int(original_data["Attribute5"]),
            step=50
        )

    if st.button("Run What-If Prediction"):

        try:

            from src.predict import predict_credit_risk

            what_if_data = original_data.copy()

            what_if_data["Attribute2"] = what_if_duration
            what_if_data["Attribute5"] = what_if_credit

            what_if_prediction, what_if_probability = (
                predict_credit_risk(what_if_data)
            )

            st.subheader("What-If Result")

            if what_if_prediction == 0:

                st.success("🟢 Lower Risk")

            else:

                st.error("🔴 Higher Risk")

            st.metric(
                "New Risk Probability",
                f"{what_if_probability * 100:.2f}%"
            )

            original_probability = st.session_state["probability"]

            change = (
                what_if_probability - original_probability
            ) * 100

            st.metric(
                "Change in Risk Probability",
                f"{change:+.2f}%"
            )

        except Exception as e:

            st.error(
                f"What-If prediction failed: {e}"
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Educational project — predictions are for demonstration "
    "and should not be used for real financial decisions."
)

