import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ------------------- Streamlit Config -------------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- Custom Styling -------------------
st.markdown("""
<style>
    body, .stApp {
        background: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00416A, #00B4DB);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: bold;
        color: #00416A;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background: #d1e7dd;
        border-left: 6px solid #0f5132;
        padding: 1rem;
        border-radius: 8px;
    }
    .danger-box {
        background: #f8d7da;
        border-left: 6px solid #842029;
        padding: 1rem;
        border-radius: 8px;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 6px solid #664d03;
        padding: 1rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Main Title -------------------
st.markdown('<div class="main-title">🏦 Loan Approval Prediction</div>', unsafe_allow_html=True)

# ------------------- Helper Functions -------------------
def encode_input(value, column):
    mappings = {
        "education": {"Graduate": 1, "Not Graduate": 0},
        "self_employed": {"Yes": 1, "No": 0}
    }
    if column in mappings:
        if value in mappings[column]:
            return mappings[column][value]
        else:
            st.warning(f"⚠️ Unexpected value '{value}' for {column}, defaulting to 0.")
            return 0
    else:
        return 0

@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('loan_approval_model.pkl')
        scaler = joblib.load('loan_scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        target_encoder = joblib.load('target_encoder.pkl')
        feature_columns = joblib.load('feature_columns.pkl')
        return model, scaler, label_encoders, target_encoder, feature_columns
    except FileNotFoundError as e:
        st.error(f"Model files missing: {e}")
        return None, None, None, None, None

def create_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob*100,
        title={'text': "Approval Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#00416A"},
            'steps': [
                {'range': [0, 40], 'color': "#f8d7da"},
                {'range': [40, 70], 'color': "#fff3cd"},
                {'range': [70, 100], 'color': "#d1e7dd"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

def calculate_ratios(data):
    ratios = {}
    ratios['Loan to Income Ratio'] = data['loan_amount'] / data['income_annum']
    ratios['Total Assets'] = (data['residential_assets_value'] +
                             data['commercial_assets_value'] +
                             data['luxury_assets_value'] +
                             data['bank_asset_value'])
    ratios['Asset to Loan Ratio'] = ratios['Total Assets'] / data['loan_amount']
    ratios['Debt to Income Ratio'] = (data['loan_amount'] / data['income_annum']) * 100
    return ratios

# ------------------- Load Model -------------------
model, scaler, label_encoders, target_encoder, feature_columns = load_model_artifacts()
if model is None:
    st.stop()

# ------------------- Sidebar -------------------
st.sidebar.header("📝 Application Details")
no_of_dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 2)
education = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Employment Type", ["No", "Yes"])
income_annum = st.sidebar.number_input("Annual Income (₹)", 100000, 50000000, 5000000, step=100000)
loan_amount = st.sidebar.number_input("Loan Amount (₹)", 100000, 100000000, 1000000, step=100000)
loan_term = st.sidebar.slider("Loan Term (Years)", 1, 30, 15)
cibil_score = st.sidebar.slider("CIBIL Score", 300, 900, 650)
residential_assets = st.sidebar.number_input("Residential Assets Value (₹)", 0, 100000000, 5000000, step=100000)
commercial_assets = st.sidebar.number_input("Commercial Assets Value (₹)", 0, 100000000, 2000000, step=100000)
luxury_assets = st.sidebar.number_input("Luxury Assets Value (₹)", 0, 100000000, 1000000, step=100000)
bank_assets = st.sidebar.number_input("Bank Assets Value (₹)", 0, 100000000, 3000000, step=100000)
predict_button = st.sidebar.button("🔮 Predict Loan Approval", use_container_width=True)

# ------------------- Input Data -------------------
input_data = {
    'no_of_dependents': no_of_dependents,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets,
    'commercial_assets_value': commercial_assets,
    'luxury_assets_value': luxury_assets,
    'bank_asset_value': bank_assets
}
ratios = calculate_ratios(input_data)
total_assets = ratios['Total Assets']

# ------------------- Tabs Layout -------------------
tab1, tab2, tab3 = st.tabs(["📊 Summary", "🎯 Prediction", "💡 Insights"])

with tab1:
    st.markdown('<div class="section-header">Application Summary</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Annual Income", f"₹{income_annum:,}")
    with col2: st.metric("Loan Amount", f"₹{loan_amount:,}")
    with col3: st.metric("CIBIL Score", cibil_score)
    st.write("### Financial Ratios")
    st.json(ratios)

with tab2:
    st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
    if predict_button:
        prediction_data = input_data.copy()
        prediction_data['education_encoded'] = encode_input(education, "education")
        prediction_data['self_employed_encoded'] = encode_input(self_employed, "self_employed")
        prediction_data['loan_to_income_ratio'] = ratios['Loan to Income Ratio']
        prediction_data['total_assets'] = ratios['Total Assets']
        prediction_data['asset_to_loan_ratio'] = ratios['Asset to Loan Ratio']

        feature_vector = np.array([[prediction_data[col] for col in feature_columns]])
        feature_vector_scaled = scaler.transform(feature_vector)
        prediction = model.predict(feature_vector_scaled)[0]
        prediction_proba = model.predict_proba(feature_vector_scaled)[0]
        prediction_label = target_encoder.inverse_transform([prediction])[0]
        approval_probability = prediction_proba[1]

        if prediction_label == "Approved":
            st.markdown(f"<div class='success-box'><h3>✅ Loan APPROVED</h3><p>Confidence: {approval_probability:.1%}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='danger-box'><h3>❌ Loan REJECTED</h3><p>Confidence: {(1-approval_probability):.1%}</p></div>", unsafe_allow_html=True)

        st.plotly_chart(create_gauge(approval_probability), use_container_width=True)

        prob_df = pd.DataFrame({
            'Outcome': ['Rejected', 'Approved'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        fig_bar = px.bar(prob_df, x='Outcome', y='Probability', color='Outcome',
                         color_discrete_map={'Approved': '#28a745', 'Rejected': '#dc3545'})
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("👆 Fill in the sidebar and click Predict.")

with tab3:
    st.markdown('<div class="section-header">Insights & Recommendations</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if cibil_score >= 750:
            st.success("Excellent CIBIL Score ✅")
        elif cibil_score >= 650:
            st.warning("Good Score, try to reach 750+ ⚠️")
        else:
            st.error("Low Score, needs improvement ❌")
    with col2:
        loan_to_income = loan_amount / income_annum
        if loan_to_income <= 3:
            st.success("Healthy Loan-to-Income Ratio ✅")
        elif loan_to_income <= 5:
            st.warning("Moderate Ratio ⚠️")
        else:
            st.error("High Ratio ❌")
    with col3:
        asset_cov = total_assets / loan_amount
        if asset_cov >= 2:
            st.success("Strong Asset Coverage ✅")
        elif asset_cov >= 1:
            st.warning("Adequate Asset Coverage ⚠️")
        else:
            st.error("Weak Asset Coverage ❌")

# ------------------- Footer -------------------
st.markdown("---")
st.markdown("<p style='text-align:center;color:gray'>🏦 Loan Approval Prediction | Streamlit & ML</p>", unsafe_allow_html=True)
