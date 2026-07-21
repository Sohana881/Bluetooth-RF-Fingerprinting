import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os

from predict import predict

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Bluetooth RF Fingerprinting",
    page_icon="📡",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#f7f9fc;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#0B5394;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:#6c757d;
    margin-bottom:30px;
}

.stDataFrame{
    border-radius:12px;
}

div[data-testid="metric-container"]{
    background:white;
    border:1px solid #eeeeee;
    border-radius:15px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    "<div class='title'>📡 Bluetooth RF Fingerprinting</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>AI-Based Bluetooth Device Identification using RF Fingerprints</div>",
    unsafe_allow_html=True
)

st.divider()

# ==========================================================
# FILE UPLOADER
# ==========================================================

uploaded_files = st.file_uploader(
    "Upload Bluetooth Signal(s)",
    type=["txt"],
    accept_multiple_files=True
)
# ==========================================================
# PREDICT ALL FILES
# ==========================================================

if uploaded_files:

    os.makedirs("uploads", exist_ok=True)

    results = []

    single_result = None
    single_filepath = None

    with st.spinner("Running Bluetooth RF Fingerprinting..."):

        for uploaded_file in uploaded_files:

            filepath = os.path.join(
                "uploads",
                uploaded_file.name
            )

            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            result = predict(filepath)

            # Save first result for graph/features
            if len(uploaded_files) == 1:
                single_result = result
                single_filepath = filepath

            confidence = max(
                result["brand_conf"],
                result["model_conf"],
                result["user_conf"]
            )

            results.append({
                "File": uploaded_file.name,
                "Brand": result["brand"],
                "Model": result["model"],
                "Device ID": result["user"],
                "Confidence (%)": f"{confidence:.2f}"
            })

    st.success(
        f"Prediction completed for {len(uploaded_files)} file(s)."
    )

    # ======================================================
    # PREDICTION TABLE
    # ======================================================

    st.subheader("Prediction Results")

    result_df = pd.DataFrame(results)

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # SINGLE FILE METRICS
    # ======================================================

    if len(uploaded_files) == 1:

        st.divider()

        st.subheader("Prediction Summary")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🏷 Brand",
                single_result["brand"],
                f"{single_result['brand_conf']:.2f}%"
            )

        with col2:
            st.metric(
                "📱 Model",
                single_result["model"],
                f"{single_result['model_conf']:.2f}%"
            )

        with col3:
            st.metric(
                "👤 Device ID",
                single_result["user"],
                f"{single_result['user_conf']:.2f}%"
            )



# ==========================================================
# SINGLE FILE VISUALIZATION
# ==========================================================

if uploaded_files and len(uploaded_files) == 1:

    st.divider()

    signal = single_result["signal"]
    features = single_result["features"]

    # ------------------------------------------------------
    # SIGNAL GRAPH
    # ------------------------------------------------------

    st.subheader("Signal Preview")

    fig = px.line(
        x=np.arange(len(signal)),
        y=signal,
        labels={
            "x": "Samples",
            "y": "Amplitude"
        }
    )

    fig.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------
    # SIGNAL STATISTICS
    # ------------------------------------------------------

    st.subheader("Signal Statistics")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Samples",
        len(signal)
    )

    c2.metric(
        "Mean",
        f"{np.mean(signal):.6f}"
    )

    c3.metric(
        "Std Deviation",
        f"{np.std(signal):.6f}"
    )

    c4.metric(
        "Maximum",
        f"{np.max(signal):.6f}"
    )

    # ------------------------------------------------------
    # FEATURE TABLE
    # ------------------------------------------------------

    st.divider()

    st.subheader("Extracted Features")

    feature_df = pd.DataFrame(
        {
            "Feature": list(features.keys()),
            "Value": list(features.values())
        }
    )

    st.dataframe(
        feature_df,
        use_container_width=True,
        hide_index=True
    )