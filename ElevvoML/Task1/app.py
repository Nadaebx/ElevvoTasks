import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for a professional theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .stSlider>div>div>div>div {
        background-color: #2563eb;
    }
    
    h1 {
        color: #1e293b;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e293b;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #475569;
        font-weight: 500;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .metric-card {
        background-color: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #2563eb;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
        padding: 1.8rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 1.8rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .recommendation-box {
        background-color: #f1f5f9;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
    }
    
    .stSelectbox>div>div {
        border-radius: 6px;
        border: 1px solid #e2e8f0;
    }
    
    .stSelectbox>div>div:hover {
        border-color: #93c5fd;
    }
    
    .stForm {
        border-radius: 8px;
        padding: 1.5rem;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .stRadio>div {
        background-color: white;
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #e2e8f0;
    }
    
    div[data-testid="stHorizontalBlock"] {
        background: white;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        padding: 1.5rem;
        margin-top: 2rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .header-icon {
        font-size: 2.5rem;
        margin-right: 1rem;
        color: #2563eb;
    }
    
    .subheader {
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .tab-container {
        background: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    
    .success-box {
        background-color: #dcfce7;
        color: #166534;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #22c55e;
    }
    
    .warning-box {
        background-color: #fef3c7;
        color: #92400e;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    .info-box {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .error-box {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
        border-left: 4px solid #ef4444;
    }
</style>
""", unsafe_allow_html=True)

# Load models and data
@st.cache_data
def load_models_and_data():
    try:
        lr_model = joblib.load('linear_regression_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        model_info = joblib.load('model_info.pkl')
        return lr_model, rf_model, label_encoders, model_info
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please run the preprocessing notebook first. Error: {e}")
        return None, None, None, None

# Load everything
lr_model, rf_model, label_encoders, model_info = load_models_and_data()

# Main app
def main():
    st.markdown("""
    <div class="header-container">
        <div class="header-icon">🎓</div>
        <div>
            <h1>Student Performance Predictor</h1>
            <p class="subheader">Predict exam scores based on student characteristics and study habits</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if lr_model is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
        <span style="font-size: 2rem; margin-right: 0.5rem;">🎓</span>
        <h2 style="margin: 0; color: #1e293b;">Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.sidebar.radio("", ["🔮 Prediction", "📊 Model Info", "📈 Visualizations"])
    
    # Add info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div class="info-box" style="margin: 0;">
        This app predicts student exam scores using machine learning models trained on historical data.
    </div>
    """, unsafe_allow_html=True)
    
    if page == "🔮 Prediction":
        prediction_page()
    elif page == "📊 Model Info":
        model_info_page()
    elif page == "📈 Visualizations":
        visualization_page()

def prediction_page():
    st.header("📊 Predict Exam Score")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="tab-container">
            <h3>Student Information</h3>
        """, unsafe_allow_html=True)
        
        # Create input form
        with st.form("prediction_form"):
            # Numerical inputs
            hours_studied = st.slider("Hours Studied per Week", 0, 20, 10,
                                     help="Total hours spent studying each week")
            attendance = st.slider("Attendance (%)", 0, 100, 85,
                                  help="Percentage of classes attended")
            previous_scores = st.slider("Previous Scores", 0, 100, 75,
                                       help="Scores from previous exams")
            sleep_hours = st.slider("Sleep Hours per Day", 4, 12, 7,
                                   help="Average hours of sleep per night")
            tutoring_sessions = st.slider("Tutoring Sessions per Month", 0, 10, 2,
                                         help="Number of tutoring sessions attended monthly")
            physical_activity = st.slider("Physical Activity Hours per Week", 0, 10, 3,
                                         help="Hours spent on physical activities weekly")
            
            # Categorical inputs
            st.markdown("### Background Information")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                parental_involvement = st.selectbox("Parental Involvement", 
                    ["Low", "Medium", "High"])
                access_to_resources = st.selectbox("Access to Resources", 
                    ["Low", "Medium", "High"])
                motivation_level = st.selectbox("Motivation Level", 
                    ["Low", "Medium", "High"])
            
            with col_b:
                internet_access = st.selectbox("Internet Access", 
                    ["Yes", "No"])
                teacher_quality = st.selectbox("Teacher Quality", 
                    ["Low", "Medium", "High"])
                peer_influence = st.selectbox("Peer Influence", 
                    ["Negative", "Neutral", "Positive"])
            
            submitted = st.form_submit_button("Predict My Score", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if submitted:
            # Prepare input data
            input_data = prepare_input_data(
                hours_studied, attendance, previous_scores, sleep_hours,
                tutoring_sessions, physical_activity, parental_involvement,
                access_to_resources, motivation_level, internet_access,
                teacher_quality, peer_influence
            )
            
            # Make predictions
            lr_prediction = lr_model.predict([input_data])[0]
            rf_prediction = rf_model.predict([input_data])[0]
            
            with col2:
                st.markdown("""
                <div class="tab-container">
                    <h3>Prediction Results</h3>
                """, unsafe_allow_html=True)
                
                # Display predictions in a more visually appealing way
                st.markdown("<div class='prediction-result'>", unsafe_allow_html=True)
                avg_prediction = (lr_prediction + rf_prediction) / 2
                st.metric("Predicted Score", f"{avg_prediction:.1f}/100")
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Model comparison
                st.write("**Model Comparisons**")
                col_lr, col_rf = st.columns(2)
                
                with col_lr:
                    st.metric("Linear Regression", f"{lr_prediction:.1f}", 
                             delta=f"{lr_prediction - avg_prediction:+.1f}")
                
                with col_rf:
                    st.metric("Random Forest", f"{rf_prediction:.1f}", 
                             delta=f"{rf_prediction - avg_prediction:+.1f}")
                
                # Performance interpretation
                st.write("**Performance Assessment**")
                if avg_prediction >= 90:
                    st.markdown("""
                    <div class="success-box">
                        🎉 Excellent! Outstanding performance!
                    </div>
                    """, unsafe_allow_html=True)
                elif avg_prediction >= 80:
                    st.markdown("""
                    <div class="info-box">
                        💪 Very Good! Keep up the good work!
                    </div>
                    """, unsafe_allow_html=True)
                elif avg_prediction >= 70:
                    st.markdown("""
                    <div class="warning-box">
                        📖 Good, but there's room for improvement
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="error-box">
                        😔 Needs improvement - focus on study strategies
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                recommendations = generate_recommendations(
                    hours_studied, attendance, sleep_hours, tutoring_sessions,
                    parental_involvement, access_to_resources, motivation_level
                )
                for rec in recommendations:
                    st.markdown(f"<div class='recommendation-box'>{rec}</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)

def prepare_input_data(hours_studied, attendance, previous_scores, sleep_hours,
                      tutoring_sessions, physical_activity, parental_involvement,
                      access_to_resources, motivation_level, internet_access,
                      teacher_quality, peer_influence):
    
    # Encode categorical variables
    encoded_values = []
    categorical_inputs = {
        'Parental_Involvement': parental_involvement,
        'Access_to_Resources': access_to_resources,
        'Motivation_Level': motivation_level,
        'Internet_Access': internet_access,
        'Teacher_Quality': teacher_quality,
        'Peer_Influence': peer_influence
    }
    
    # Create input array in the correct order
    input_data = [
        hours_studied, attendance, previous_scores, sleep_hours,
        tutoring_sessions, physical_activity
    ]
    
    # Add encoded categorical variables
    for col_name in ['Parental_Involvement', 'Access_to_Resources', 'Motivation_Level',
                     'Internet_Access', 'Teacher_Quality', 'Peer_Influence']:
        if col_name in label_encoders:
            try:
                encoded_val = label_encoders[col_name].transform([categorical_inputs[col_name]])[0]
                input_data.append(encoded_val)
            except ValueError:
                # Handle unseen labels
                input_data.append(0)
    
    return input_data

def generate_recommendations(hours_studied, attendance, sleep_hours, tutoring_sessions,
                           parental_involvement, access_to_resources, motivation_level):
    recommendations = []
    
    if hours_studied < 10:
        recommendations.append("Consider increasing study hours to at least 10 hours per week for better performance")
    elif hours_studied > 15:
        recommendations.append("You're studying a lot! Make sure to take breaks to avoid burnout")
    
    if attendance < 85:
        recommendations.append("Regular attendance is crucial - aim for at least 90% attendance")
    
    if sleep_hours < 7:
        recommendations.append("Get adequate sleep (7-8 hours) to improve learning capacity and memory retention")
    elif sleep_hours > 9:
        recommendations.append("While sleep is important, too much can be counterproductive. Aim for 7-8 hours.")
    
    if tutoring_sessions < 2:
        recommendations.append("Consider adding tutoring sessions to strengthen challenging subjects")
    
    if parental_involvement == "Low":
        recommendations.append("Increased parental involvement has been shown to improve academic performance")
    
    if access_to_resources == "Low":
        recommendations.append("Seek out additional educational resources from your school or community")
    
    if motivation_level == "Low":
        recommendations.append("Try setting specific academic goals to increase motivation")
    
    if len(recommendations) == 0:
        recommendations.append("Great habits! Keep maintaining this balanced routine for continued success")
    
    return recommendations

def model_info_page():
    st.header("📊 Model Information")
    
    if model_info:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="tab-container">
                <h3>Model Performance</h3>
            """, unsafe_allow_html=True)
            
            # Linear Regression metrics
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            lr_metrics = model_info['model_performance']['linear_regression']
            st.write("**Linear Regression**")
            st.write(f"• R² Score: {lr_metrics['r2']:.3f}")
            st.write(f"• RMSE: {lr_metrics['rmse']:.3f}")
            st.write(f"• MAE: {lr_metrics['mae']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            # Random Forest metrics
            rf_metrics = model_info['model_performance']['random_forest']
            st.write("**Random Forest**")
            st.write(f"• R² Score: {rf_metrics['r2']:.3f}")
            st.write(f"• RMSE: {rf_metrics['rmse']:.3f}")
            st.write(f"• MAE: {rf_metrics['mae']:.3f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="tab-container">
                <h3>Dataset Statistics</h3>
            """, unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            target_stats = model_info['target_stats']
            st.write("**Exam Score Distribution**")
            st.write(f"• Minimum: {target_stats['min']:.0f}")
            st.write(f"• Maximum: {target_stats['max']:.0f}")
            st.write(f"• Average: {target_stats['mean']:.1f}")
            st.write(f"• Std Dev: {target_stats['std']:.1f}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.write("**Features Used**")
            for i, feature in enumerate(model_info['feature_columns'][:6], 1):
                st.write(f"{i}. {feature}")
            if len(model_info['feature_columns']) > 6:
                with st.expander("See all features"):
                    for i, feature in enumerate(model_info['feature_columns'][6:], 7):
                        st.write(f"{i}. {feature}")
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Model explanation
    st.markdown("""
    <div class="tab-container">
        <h3>How It Works</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**Linear Regression**")
        st.write("""
        • Uses linear relationships between features
        • Simple and interpretable
        • Assumes linear relationship with exam scores
        • Good baseline model
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("**Random Forest**")
        st.write("""
        • Ensemble of decision trees
        • Captures non-linear relationships
        • More complex but often more accurate
        • Less prone to overfitting
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def visualization_page():
    st.header("📈 Data Visualizations")
    
    # Try to load and display saved plots
    try:
        # Feature importance plot
        st.markdown("""
        <div class="tab-container">
            <h3>Feature Importance</h3>
        """, unsafe_allow_html=True)
        
        feature_img = Image.open('feature_importance.png')
        st.image(feature_img, caption="Feature Importance Comparison", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tab-container">
            <h3>Feature Correlations</h3>
        """, unsafe_allow_html=True)
        
        corr_img = Image.open('correlation_matrix.png')
        st.image(corr_img, caption="Correlation Matrix", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning("Visualization files not found. Please run the preprocessing notebook first.")
    
    # Sample data visualization
    try:
        sample_data = pd.read_csv('sample_data.csv')
        
        st.markdown("""
        <div class="tab-container">
            <h3>Sample Data Exploration</h3>
        """, unsafe_allow_html=True)
        
        # Interactive scatter plot
        fig = px.scatter(sample_data, x='Hours_Studied', y='Exam_Score', 
                        color='Attendance', size='Previous_Scores',
                        title="Hours Studied vs Exam Score",
                        labels={'Hours_Studied': 'Hours Studied per Week',
                               'Exam_Score': 'Exam Score'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution plot
        fig2 = px.histogram(sample_data, x='Exam_Score', nbins=20,
                           title="Distribution of Exam Scores")
        st.plotly_chart(fig2, use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.info("Sample data not available. Run preprocessing notebook to generate sample data.")

# Footer
def add_footer():
    st.markdown("---")
    st.markdown(
        "<div class='footer'>"
        "🎓 Student Performance Predictor | Built with Streamlit & Machine Learning"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
    add_footer()