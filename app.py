import streamlit as st
import joblib
import pandas as pd

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tender Industry Sector Classifier",
    page_icon="🏢",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .title {
        font-size: 42px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #ddd;
        text-align: center;
        margin-top: 20px;
    }

    .sector {
        font-size: 32px;
        font-weight: bold;
        margin-top: 10px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    bundle = joblib.load("industry_sector_classifier.pkl")

    model = bundle["best_model"]
    vectorizer = bundle["vectorizer"]
    label_classes = bundle["label_classes"]

    return model, vectorizer, label_classes


try:
    model, vectorizer, label_classes = load_model()

except Exception as e:
    st.error("❌ Unable to load the model.")
    st.exception(e)
    st.stop()


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    '<div class="title">🏢 Tender Industry Sector Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Classify a tender/procurement description into its Industry Sector'
    '</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.header("⚙️ Model Information")

    st.write("**Model:** Linear SVM")
    st.write("**Text Representation:** TF-IDF")
    st.write("**N-grams:** Unigrams + Bigrams")
    st.write("**Maximum Features:** 20,000")

    st.divider()

    st.write("### 📊 Model Performance")

    st.metric(
        "Accuracy",
        "71.4%"
    )

    st.metric(
        "Weighted F1",
        "71.0%"
    )

    st.divider()

    st.write("### 🏷️ Supported Sectors")

    st.write(f"Number of classes: **{len(label_classes)}**")

# --------------------------------------------------
# Main Input
# --------------------------------------------------

st.subheader("📝 Enter Tender Description")

default_text = (
    "Supply and delivery of laboratory reagents "
    "for the regional hospital"
)

tender_description = st.text_area(
    "Tender / Procurement Description",
    value=default_text,
    height=180,
    placeholder="Enter the tender description here..."
)

# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button(
    "🔍 Classify Industry Sector",
    use_container_width=True
):

    if not tender_description.strip():
        st.warning("⚠️ Please enter a tender description.")
        st.stop()

    with st.spinner("Analyzing tender description..."):

        # Convert text into TF-IDF features
        X_new = vectorizer.transform([tender_description])

        # Predict industry sector
        prediction = model.predict(X_new)

        predicted_sector = prediction[0]

    # --------------------------------------------------
    # Result
    # --------------------------------------------------

    st.markdown(
        f"""
        <div class="result-box">
            <div>Predicted Industry Sector</div>
            <div class="sector">🏆 {predicted_sector}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # Input Summary
    # --------------------------------------------------

    st.subheader("📋 Tender Summary")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Description Length",
            f"{len(tender_description.split())} words"
        )

    with col2:
        st.metric(
            "Predicted Sector",
            predicted_sector
        )

    # --------------------------------------------------
    # Process Information
    # --------------------------------------------------

    st.subheader("🔎 Classification Process")

    process_df = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4"
            ],
            "Process": [
                "Tender description received",
                "TF-IDF converts text into numerical features",
                "Linear SVM analyzes the features",
                "Industry Sector is predicted"
            ]
        }
    )

    st.dataframe(
        process_df,
        use_container_width=True,
        hide_index=True
    )

# --------------------------------------------------
# Example Section
# --------------------------------------------------

st.divider()

st.subheader("💡 Example Tender Descriptions")

examples = [
    "Construction of a new government school building",
    "Supply of medicines and medical equipment to hospitals",
    "Procurement of laptops and computer networking equipment",
    "Supply and maintenance of automobiles and spare parts",
    "Manufacturing and supply of industrial machinery"
]

for example in examples:
    st.code(example)