import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


FEATURE_NAMES = [
    "SpMax_L", "J_Dz(e)", "nHM", "F01[N-N]", "F04[C-N]",
    "NssssC", "nCb-", "C%", "nCp", "nO", "F03[C-N]", "SdssC",
    "HyWi_B(m)", "LOC", "SM6_L", "F03[C-O]", "Me", "Mi", "nN-N",
    "nArNO2", "nCRX3", "SpPosA_B(p)", "nCIR", "B01[C-Br]",
    "B03[C-Cl]", "N-073", "SpMax_A", "Psi_i_1d", "B04[C-Br]",
    "SdO", "TI2_L", "nCrt", "C-026", "F02[C-N]", "nHDon",
    "SpMax_B(m)", "Psi_i_A", "nN", "SM6_B(m)", "nArCOOR", "nX"
]

TARGET_NAME = "experimental class"

MODEL_PATHS = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "SVM": "models/svm.pkl"
}

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "Linear probabilistic classifier",
    "Decision Tree": "Rule-based tree classifier",
    "KNN": "Neighbourhood-based classifier",
    "Naive Bayes": "Gaussian probabilistic classifier",
    "SVM": "Maximum-margin classifier"
}


st.set_page_config(
    page_title="QSAR Biodegradation",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #061710;
    }

    .main {
        background-color: #061710;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #dff9eb !important;
    }

    p, label {
        color: #c7e5d4 !important;
    }

    hr {
        border-color: #285c43 !important;
    }

    [data-testid="stMetric"] {
        background-color: #0b291d;
        border: 1px solid #286a4c;
        border-radius: 12px;
        padding: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #83b99b !important;
    }

    [data-testid="stMetricValue"] {
        color: #a8f4c9 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #0b291d;
        border-color: #367d59;
    }

    div[data-testid="stFileUploader"] {
        background-color: #0b291d;
        border: 1px solid #286a4c;
        border-radius: 12px;
        padding: 0.5rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #04130d;
        border-right: 1px solid #1f4d36;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #a6efc5 !important;
    }

    .section-label {
        color: #78dfa7;
        font-size: 0.75rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        color: #94bea5;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1.5rem;
    }

    .class-box {
        background-color: #0b291d;
        border: 1px solid #367d59;
        border-radius: 14px;
        padding: 1rem 1.25rem;
        margin: 1.2rem 0;
    }

    .class-box-title {
        color: #7ee2ad;
        font-weight: 800;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .class-box p {
        margin: 0.35rem 0;
        color: #c7e5d4;
    }

    .model-description {
        color: #8fbda3;
        font-size: 0.9rem;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
    }

    .footer {
        color: #638a74;
        text-align: center;
        font-size: 0.78rem;
        margin-top: 1.5rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧪 QSAR Lab")

    st.write(
        "Machine learning evaluation of molecular descriptors "
        "for biodegradability classification."
    )

    st.divider()

    st.markdown("### Dataset")

    st.write("**1,052** unique observations")
    st.write("**41** molecular descriptors")
    st.write("**2** target classes")

    st.divider()

    st.markdown("### Classifiers")

    st.write("Logistic Regression")
    st.write("Decision Tree")
    st.write("KNN")
    st.write("Naive Bayes")
    st.write("SVM")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="section-label">QSAR • MACHINE LEARNING</div>',
    unsafe_allow_html=True
)

st.title("Biodegradation Prediction System")

st.markdown(
    """
    <div class="hero-subtitle">
        Evaluate the biodegradability of chemical compounds using
        molecular descriptor data and five pre-trained machine learning
        classifiers.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CLASS DEFINITIONS
# ============================================================

st.markdown(
    '<div class="section-label">Biodegradation Classes</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    st.markdown("**RB (1) — Ready Biodegradable**")

    st.write(
        "Compounds classified as ready biodegradable."
    )

    st.markdown("**NRB (0) — Not Ready Biodegradable**")

    st.write(
        "Compounds classified as not ready biodegradable."
    )


# ============================================================
# DATASET INFORMATION
# ============================================================

st.markdown(
    '<div class="section-label">Dataset</div>',
    unsafe_allow_html=True
)

dataset_1, dataset_2, dataset_3 = st.columns(3)

dataset_1.metric(
    "Observations",
    "1,052"
)

dataset_2.metric(
    "Molecular Descriptors",
    "41"
)

dataset_3.metric(
    "Classes",
    "2"
)


# ============================================================
# EVALUATION SETUP
# ============================================================

st.markdown(
    '<div class="section-label">01 / Evaluation Setup</div>',
    unsafe_allow_html=True
)

upload_col, model_col = st.columns([1.2, 1])

with upload_col:

    uploaded_file = st.file_uploader(
        "Upload labelled test dataset",
        type=["csv"],
        help=(
            "CSV must contain all 41 molecular descriptors "
            "and the experimental class column."
        )
    )

with model_col:

    model_name = st.selectbox(
        "Select classifier",
        list(MODEL_PATHS.keys())
    )


if uploaded_file is None:

    st.info(
        "Upload the held-out test_data.csv file to begin evaluation."
    )

    st.markdown(
        '<div class="footer">Pre-trained inference only • '
        'No model retraining is performed on uploaded data.</div>',
        unsafe_allow_html=True
    )

    st.stop()


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

data = pd.read_csv(uploaded_file)

required_columns = FEATURE_NAMES + [TARGET_NAME]

missing_columns = [
    column
    for column in required_columns
    if column not in data.columns
]

if missing_columns:

    st.error(
        "The uploaded CSV is missing required columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


y_test = data[TARGET_NAME]

if not set(y_test.unique()).issubset({0, 1}):

    st.error(
        "The target column must contain only 0 (NRB) and 1 (RB)."
    )

    st.stop()


X_test = data[FEATURE_NAMES]


# ============================================================
# UPLOADED DATA SUMMARY
# ============================================================

st.markdown(
    '<div class="section-label">02 / Uploaded Dataset</div>',
    unsafe_allow_html=True
)

data_1, data_2, data_3, data_4 = st.columns(4)

data_1.metric(
    "Samples",
    len(data)
)

data_2.metric(
    "Features",
    len(FEATURE_NAMES)
)

data_3.metric(
    "Classes",
    y_test.nunique()
)

data_4.metric(
    "Missing Values",
    int(data.isnull().sum().sum())
)


# ============================================================
# MODEL
# ============================================================

st.markdown(
    '<div class="section-label">03 / Selected Model</div>',
    unsafe_allow_html=True
)

st.subheader(model_name)

st.markdown(
    f'<div class="model-description">'
    f'{MODEL_DESCRIPTIONS[model_name]}'
    f'</div>',
    unsafe_allow_html=True
)


# ============================================================
# PREDICTION
# ============================================================

model = joblib.load(
    MODEL_PATHS[model_name]
)

predictions = model.predict(X_test)

if hasattr(model, "predict_proba"):

    scores = model.predict_proba(X_test)[:, 1]

else:

    scores = model.decision_function(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

auc = roc_auc_score(
    y_test,
    scores
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)


st.markdown(
    '<div class="section-label">04 / Model Performance</div>',
    unsafe_allow_html=True
)

metric_1, metric_2, metric_3 = st.columns(3)

metric_1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

metric_2.metric(
    "AUC",
    f"{auc:.4f}"
)

metric_3.metric(
    "Precision",
    f"{precision:.4f}"
)


metric_4, metric_5, metric_6 = st.columns(3)

metric_4.metric(
    "Recall",
    f"{recall:.4f}"
)

metric_5.metric(
    "F1 Score",
    f"{f1:.4f}"
)

metric_6.metric(
    "MCC",
    f"{mcc:.4f}"
)


# ============================================================
# DIAGNOSTIC ANALYSIS
# ============================================================

st.markdown(
    '<div class="section-label">05 / Diagnostic Analysis</div>',
    unsafe_allow_html=True
)

matrix_col, report_col = st.columns(2)


with matrix_col:

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        predictions
    )

    cm_df = pd.DataFrame(
        cm,
        index=["Actual NRB", "Actual RB"],
        columns=["Predicted NRB", "Predicted RB"]
    )

    st.dataframe(
        cm_df,
        width="stretch"
    )


with report_col:

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        target_names=["NRB", "RB"],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(4),
        width="stretch"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    '<div class="footer">'
    'QSAR Biodegradation • Pre-trained model inference • '
    'No model retraining on uploaded test data'
    '</div>',
    unsafe_allow_html=True
)