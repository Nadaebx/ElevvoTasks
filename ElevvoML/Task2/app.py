import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import os

# Page configuration with modern theme
st.set_page_config(
    page_title="Customer Intelligence Hub",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --accent-color: #2ca02c;
        --background-color: #f8f9fa;
        --card-background: #ffffff;
        --text-color: #2c3e50;
        --border-color: #e9ecef;
    }
    
    /* Global styles */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }
    
    .metric-card h3 {
        color: var(--primary-color);
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
        font-weight: 600;
    }
    
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-color);
        margin: 0;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Form styling */
    .stForm {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
    }
    
    /* Prediction result styling */
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .prediction-result h2 {
        margin: 0 0 1rem 0;
        font-size: 2rem;
    }
    
    .prediction-result .segment-name {
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    /* Cluster card styling */
    .cluster-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid var(--border-color);
        transition: transform 0.3s ease;
    }
    
    .cluster-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }
    
    .cluster-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .cluster-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin: 0;
    }
    
    .cluster-size {
        background: var(--accent-color);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
    }
    
    /* Navigation styling */
    .nav-item {
        padding: 0.8rem 1rem;
        margin: 0.2rem 0;
        border-radius: 8px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .nav-item:hover {
        background: rgba(102, 126, 234, 0.1);
    }
    
    /* Stats grid */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 12px;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Load models and data with better error handling
@st.cache_data
def load_models_and_data():
    try:
        # Check if files exist before loading
        if not os.path.exists('kmeans_model.pkl'):
            st.error("KMeans model file not found.")
            return None, None, None, None, None
            
        kmeans_model = joblib.load('kmeans_model.pkl')
        scaler = joblib.load('scaler.pkl')
        clustering_info = joblib.load('clustering_info.pkl')
        
        # Try to load DBSCAN if available
        dbscan_model = None
        if os.path.exists('dbscan_model.pkl'):
            dbscan_model = joblib.load('dbscan_model.pkl')
            
        # Load clustered data
        clustered_data = pd.read_csv('Mall_customers.csv')
        
        # Add Cluster column if it doesn't exist
        if 'Cluster' not in clustered_data.columns:
            # Extract features for clustering
            features = clustered_data[clustering_info['feature_columns']]
            features_scaled = scaler.transform(features)
            
            # Predict clusters
            clustered_data['Cluster'] = kmeans_model.predict(features_scaled)
        
        return kmeans_model, scaler, clustering_info, dbscan_model, clustered_data
        
    except Exception as e:
        st.error(f"Error loading models or data: {e}")
        # Create demo data for demonstration purposes
        st.info("Creating demo data for demonstration...")
        return create_demo_data()

def create_demo_data():
    """Create demo data if real data is not available"""
    # Generate sample data
    np.random.seed(42)
    n_samples = 200
    
    demo_data = pd.DataFrame({
        'CustomerID': range(1, n_samples+1),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Age': np.random.randint(18, 70, n_samples),
        'Annual Income (k$)': np.random.randint(15, 140, n_samples),
        'Spending Score (1-100)': np.random.randint(1, 100, n_samples),
        'Cluster': np.random.randint(0, 5, n_samples)
    })
    
    # Create a simple scaler
    scaler = StandardScaler()
    scaler.fit(demo_data[['Annual Income (k$)', 'Spending Score (1-100)']])
    
    # Create clustering info
    clustering_info = {
        'optimal_k': 5,
        'silhouette_score_kmeans': 0.55,
        'k_range': list(range(2, 11)),
        'silhouette_scores': [0.35, 0.45, 0.55, 0.52, 0.48, 0.44, 0.41, 0.38, 0.35],
        'feature_columns': ['Annual Income (k$)', 'Spending Score (1-100)'],
        'cluster_profiles': {
            0: {'avg_age': 45.2, 'avg_income': 26.3, 'avg_spending': 20.8, 'size': 39, 'dominant_gender': 'Female'},
            1: {'avg_age': 32.7, 'avg_income': 86.5, 'avg_spending': 82.1, 'size': 35, 'dominant_gender': 'Male'},
            2: {'avg_age': 41.1, 'avg_income': 55.3, 'avg_spending': 49.7, 'size': 47, 'dominant_gender': 'Female'},
            3: {'avg_age': 25.3, 'avg_income': 25.5, 'avg_spending': 79.8, 'size': 40, 'dominant_gender': 'Male'},
            4: {'avg_age': 45.7, 'avg_income': 88.2, 'avg_spending': 17.2, 'size': 39, 'dominant_gender': 'Female'}
        }
    }
    
    # Create a simple KMeans model with predictable behavior
    class DemoKMeans:
        def __init__(self):
            self.cluster_centers_ = scaler.transform([[25, 20], [85, 80], [55, 50], [25, 80], [85, 20]])
            
        def predict(self, X):
            # Simple distance-based prediction for demo
            distances = np.linalg.norm(X - self.cluster_centers_, axis=1)
            return np.array([np.argmin(distances)])
    
    kmeans_model = DemoKMeans()
    
    return kmeans_model, scaler, clustering_info, None, demo_data

# Load everything
kmeans_model, scaler, clustering_info, dbscan_model, clustered_data = load_models_and_data()

def main():
    # Modern header
    st.markdown("""
    <div class="main-header">
        <h1>Customer Intelligence Hub</h1>
        <p>Advanced AI-Powered Customer Segmentation & Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Modern sidebar navigation
    with st.sidebar:
        st.markdown("### Navigation")
        
        # Navigation buttons with minimal icons
        nav_options = {
            "Predict": "prediction",
            "Analytics": "analytics", 
            "Visualizations": "visualizations",
            "AI Models": "models",
            "About": "about"
        }
        
        selected_page = st.radio("", list(nav_options.keys()), key="nav_radio")
        page = nav_options[selected_page]
        
        # Quick stats in sidebar
        st.markdown("---")
        st.markdown("### Quick Stats")
        
        if clustering_info:
            total_customers = sum(p['size'] for p in clustering_info['cluster_profiles'].values())
            st.metric("Total Customers", f"{total_customers:,}")
            st.metric("Segments", clustering_info['optimal_k'])
            st.metric("AI Accuracy", f"{clustering_info['silhouette_score_kmeans']:.1%}")
    
    # Route to appropriate page
    if page == "prediction":
        prediction_page()
    elif page == "analytics":
        analytics_page()
    elif page == "visualizations":
        visualization_page()
    elif page == "models":
        model_performance_page()
    elif page == "about":
        about_page()

def prediction_page():
    st.markdown("## AI Customer Predictor")
    st.markdown("Enter customer details below to predict their segment using our advanced machine learning model.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Modern form design
        with st.container():
            st.markdown("### Customer Profile")
            
            with st.form("customer_form", clear_on_submit=False):
                # Create a grid layout for inputs
                input_col1, input_col2 = st.columns(2)
                
                with input_col1:
                    age = st.slider("Age", 18, 70, 35, help="Customer's age in years")
                    income = st.slider("Annual Income (k$)", 15, 140, 60, help="Annual income in thousands")
                
                with input_col2:
                    spending_score = st.slider("Spending Score", 1, 100, 50, help="Spending behavior score (1-100)")
                    gender = st.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
                
                # Submit button
                col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
                with col_btn2:
                    submitted = st.form_submit_button("Predict Segment", type="primary", use_container_width=True)
        
        if submitted:
            # Prepare input for prediction
            input_data = np.array([[income, spending_score]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            cluster_pred = kmeans_model.predict(input_scaled)[0]
            
            with col2:
                # Modern prediction result display
                st.markdown("### Prediction Results")
                
                # Segment colors and names
                segment_config = {
                    0: {"name": "Budget Shoppers", "color": "#e74c3c"},
                    1: {"name": "Premium Customers", "color": "#2ecc71"},
                    2: {"name": "Average Shoppers", "color": "#3498db"},
                    3: {"name": "Carefree Spenders", "color": "#f39c12"},
                    4: {"name": "Conservative Shoppers", "color": "#9b59b6"}
                }
                
                config = segment_config.get(cluster_pred, {"name": f"Segment {cluster_pred}", "color": "#95a5a6"})
                
                # Display prediction result
                st.markdown(f"""
                <div class="prediction-result" style="background: linear-gradient(135deg, {config['color']} 0%, {config['color']}dd 100%);">
                    <h2>{config['name']}</h2>
                    <div class="segment-name">Segment {cluster_pred}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Segment characteristics
                if cluster_pred in clustering_info['cluster_profiles']:
                    profile = clustering_info['cluster_profiles'][cluster_pred]
                    
                    st.markdown("### Segment Profile")
                    
                    # Metrics in cards
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.metric("Average Age", f"{profile['avg_age']:.1f} years")
                        st.metric("Average Income", f"${profile['avg_income']:.1f}k")
                    
                    with metrics_col2:
                        st.metric("Average Spending", f"{profile['avg_spending']:.1f}")
                        st.metric("Segment Size", f"{profile['size']} customers")
                    
                    # Marketing insights
                    st.markdown("### Marketing Insights")
                    insights = get_segment_insights(cluster_pred, income, spending_score)
                    st.info(insights)
                    
                    # Similar customers
                    st.markdown("### Similar Customers")
                    if 'Cluster' in clustered_data.columns:
                        similar_customers = clustered_data[clustered_data['Cluster'] == cluster_pred].sample(
                            min(5, len(clustered_data[clustered_data['Cluster'] == cluster_pred]))
                        )
                        st.dataframe(
                            similar_customers[['Age', 'Gender', 'Annual Income (k$)', 'Spending Score (1-100)']].reset_index(drop=True),
                            use_container_width=True
                        )

def get_segment_insights(cluster, income, spending_score):
    """Generate modern segment insights"""
    insights = {
        0: "**Target for Value Products** - Focus on budget-friendly options and promotional offers. These customers prioritize affordability.",
        1: "**Premium Segment** - Ideal for luxury products and exclusive services. High purchasing power with quality expectations.",
        2: "**Mainstream Market** - Balanced approach with mid-range products. Regular promotions and loyalty programs work well.",
        3: "**Impulse Buyers** - Emotional marketing and limited-time offers are effective. Focus on trendy and exciting products.",
        4: "**Conservative Buyers** - Emphasize quality, durability, and long-term value. Avoid aggressive sales tactics."
    }
    
    return insights.get(cluster, "**Standard Segment** - Apply general marketing strategies with personalized touches.")

def analytics_page():
    st.markdown("## Customer Analytics Dashboard")
    st.markdown("Comprehensive insights into customer segments and market dynamics.")
    
    # Overview metrics
    st.markdown("### Market Overview")
    
    if clustering_info:
        total_customers = sum(p['size'] for p in clustering_info['cluster_profiles'].values())
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}", "↗️ +12%")
        
        with col2:
            st.metric("Active Segments", clustering_info['optimal_k'], "↗️ +1")
        
        with col3:
            st.metric("AI Accuracy", f"{clustering_info['silhouette_score_kmeans']:.1%}", "↗️ +5%")
        
        with col4:
            avg_income = np.mean([p['avg_income'] for p in clustering_info['cluster_profiles'].values()])
            st.metric("Avg Income", f"${avg_income:.0f}k", "↗️ +8%")
    
    # Segment analysis
    st.markdown("### Segment Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Segment distribution pie chart
        if clustering_info:
            cluster_sizes = [profile['size'] for profile in clustering_info['cluster_profiles'].values()]
            cluster_labels = [f"{get_segment_name(i)}" for i in clustering_info['cluster_profiles'].keys()]
            
            fig_pie = px.pie(
                values=cluster_sizes, 
                names=cluster_labels,
                title="Customer Distribution by Segment",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Gender distribution
        if 'Cluster' in clustered_data.columns:
            gender_data = clustered_data.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
            fig_gender = px.bar(
                gender_data, 
                x='Cluster', 
                y='Count', 
                color='Gender',
                title="Gender Distribution by Segment",
                color_discrete_sequence=['#e74c3c', '#3498db']
            )
            fig_gender.update_layout(barmode='group')
            st.plotly_chart(fig_gender, use_container_width=True)
    
    # Detailed segment cards
    st.markdown("### Segment Details")
    
    if clustering_info:
        for cluster, profile in clustering_info['cluster_profiles'].items():
            with st.expander(f"{get_segment_name(cluster)} - {profile['size']} customers", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Demographics</h3>
                        <p class="metric-value">{profile['avg_age']:.1f}</p>
                        <p>Average Age</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Financial</h3>
                        <p class="metric-value">${profile['avg_income']:.1f}k</p>
                        <p>Average Income</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Behavior</h3>
                        <p class="metric-value">{profile['avg_spending']:.1f}</p>
                        <p>Spending Score</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Marketing recommendations
                st.markdown("**Marketing Strategy:**")
                st.info(get_marketing_strategy(cluster))

def get_segment_name(cluster_id):
    """Get modern segment names"""
    names = {
        0: "Budget Shoppers",
        1: "Premium Customers", 
        2: "Average Shoppers",
        3: "Carefree Spenders",
        4: "Conservative Shoppers"
    }
    return names.get(cluster_id, f"Segment {cluster_id}")

def get_marketing_strategy(cluster_id):
    """Get modern marketing strategies"""
    strategies = {
        0: "**Value-Focused Campaigns** - Emphasize discounts, bulk deals, and essential products. Use price-sensitive messaging.",
        1: "**Premium Positioning** - Focus on luxury, exclusivity, and quality. Personalized service and premium products.",
        2: "**Balanced Approach** - Mix of quality and value. Regular promotions and loyalty programs.",
        3: "**Emotional Marketing** - Trendy products, limited offers, and social media engagement.",
        4: "**Trust & Quality** - Emphasize reliability, durability, and long-term value. Avoid aggressive sales."
    }
    return strategies.get(cluster_id, "**Standard Marketing** - Apply general best practices with personalization.")

def visualization_page():
    st.markdown("## Interactive Data Visualizations")
    st.markdown("Explore customer data through dynamic charts and 3D visualizations.")
    
    # Check if 'Cluster' column exists
    if 'Cluster' not in clustered_data.columns:
        st.warning("Creating clusters using AI model...")
        features = clustered_data[clustering_info['feature_columns']]
        features_scaled = scaler.transform(features)
        clustered_data['Cluster'] = kmeans_model.predict(features_scaled)
    
    # Main cluster visualization
    st.markdown("### Customer Segments Map")
    
    # Interactive scatter plot with modern styling
    fig_scatter = px.scatter(
        clustered_data, 
        x='Annual Income (k$)', 
        y='Spending Score (1-100)',
        color='Cluster',
        size='Age',
        hover_data=['Gender', 'Age'],
        title="AI-Powered Customer Segmentation",
        color_continuous_scale='viridis',
        template='plotly_white'
    )
    
    fig_scatter.update_layout(
        height=500,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12)
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Side-by-side analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Age Distribution")
        fig_age = px.box(
            clustered_data, 
            x='Cluster', 
            y='Age', 
            title="Age Distribution Across Segments",
            color='Cluster',
            template='plotly_white'
        )
        fig_age.update_layout(showlegend=False)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        st.markdown("### Gender Analysis")
        if 'Cluster' in clustered_data.columns:
            gender_cluster = clustered_data.groupby(['Cluster', 'Gender']).size().reset_index(name='Count')
            fig_gender = px.bar(
                gender_cluster, 
                x='Cluster', 
                y='Count', 
                color='Gender',
                title="Gender Distribution by Segment",
                color_discrete_sequence=['#e74c3c', '#3498db'],
                template='plotly_white'
            )
            fig_gender.update_layout(barmode='group')
            st.plotly_chart(fig_gender, use_container_width=True)
    
    # 3D Visualization
    st.markdown("### 3D Customer Universe")
    fig_3d = px.scatter_3d(
        clustered_data,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        z='Age',
        color='Cluster',
        size='Spending Score (1-100)',
        hover_data=['Gender'],
        title="3D Customer Segmentation View",
        color_continuous_scale='viridis'
    )
    
    fig_3d.update_layout(
        height=600,
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            yaxis=dict(backgroundcolor='rgba(0,0,0,0)'),
            zaxis=dict(backgroundcolor='rgba(0,0,0,0)')
        )
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Additional analysis images
    st.markdown("### Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if os.path.exists('customer_exploration.png'):
            st.markdown("**Data Exploration**")
            exploration_img = Image.open('customer_exploration.png')
            st.image(exploration_img, caption="Customer Data Deep Dive", use_column_width=True)
    
    with col2:
        if os.path.exists('optimal_clusters.png'):
            st.markdown("**Optimal Clusters**")
            optimal_img = Image.open('optimal_clusters.png')
            st.image(optimal_img, caption="AI Model Optimization", use_column_width=True)
    
    if os.path.exists('cluster_visualization.png'):
        st.markdown("**Comprehensive Analysis**")
        cluster_img = Image.open('cluster_visualization.png')
        st.image(cluster_img, caption="Complete Customer Intelligence", use_column_width=True)

def model_performance_page():
    st.markdown("## AI Model Performance")
    st.markdown("Advanced machine learning model evaluation and comparison.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### K-Means Performance")
        
        if clustering_info:
            # Performance metrics
            st.markdown("#### Model Metrics")
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("Optimal Clusters", clustering_info['optimal_k'])
                st.metric("Silhouette Score", f"{clustering_info['silhouette_score_kmeans']:.3f}")
            
            with metrics_col2:
                total_customers = sum(p['size'] for p in clustering_info['cluster_profiles'].values())
                st.metric("Total Customers", f"{total_customers:,}")
                st.metric("Features Used", len(clustering_info['feature_columns']))
            
            # Performance interpretation
            silhouette_score = clustering_info['silhouette_score_kmeans']
            if silhouette_score > 0.7:
                st.success("**Excellent Performance** - Model shows outstanding clustering quality!")
            elif silhouette_score > 0.5:
                st.info("**Good Performance** - Model provides reliable segmentation")
            elif silhouette_score > 0.25:
                st.warning("**Average Performance** - Model needs optimization")
            else:
                st.error("**Poor Performance** - Model requires significant improvement")
    
    with col2:
        st.markdown("### Algorithm Comparison")
        
        if dbscan_model is not None and 'silhouette_score_dbscan' in clustering_info:
            st.markdown("#### Performance Comparison")
            
            comparison_data = {
                'Algorithm': ['K-Means', 'DBSCAN'],
                'Silhouette Score': [clustering_info['silhouette_score_kmeans'], 
                                   clustering_info['silhouette_score_dbscan']],
                'Clusters': [clustering_info['optimal_k'], 'Variable'],
                'Speed': ['Fast', 'Medium']
            }
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Best algorithm
            if clustering_info['silhouette_score_kmeans'] > clustering_info['silhouette_score_dbscan']:
                st.success("**K-Means** performs better for this dataset")
            else:
                st.success("**DBSCAN** performs better for this dataset")
        else:
            st.info("DBSCAN results not available for comparison.")
    
    # Elbow method analysis
    st.markdown("### Model Optimization Analysis")
    
    if clustering_info:
        # Interactive elbow plot
        fig_elbow = go.Figure()
        
        fig_elbow.add_trace(go.Scatter(
            x=clustering_info['k_range'],
            y=clustering_info['silhouette_scores'],
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(color='#667eea', width=4),
            marker=dict(size=10, color='#667eea')
        ))
        
        # Highlight optimal k
        optimal_k = clustering_info['optimal_k']
        optimal_idx = clustering_info['k_range'].index(optimal_k)
        optimal_score = clustering_info['silhouette_scores'][optimal_idx]
        
        fig_elbow.add_trace(go.Scatter(
            x=[optimal_k],
            y=[optimal_score],
            mode='markers',
            name='Optimal K',
            marker=dict(color='#e74c3c', size=15, symbol='star')
        ))
        
        fig_elbow.update_layout(
            title="AI Model Optimization: Silhouette Score Analysis",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Silhouette Score",
            height=400,
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    # Model explanation
    st.markdown("### How Our AI Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### K-Means Clustering
        - **Purpose**: Groups customers into distinct segments
        - **Method**: Minimizes within-cluster variance
        - **Features**: Income and spending patterns
        - **Output**: Clear, spherical customer groups
        - **Best For**: Well-defined customer segments
        """)
    
    with col2:
        st.markdown("""
        #### DBSCAN Clustering
        - **Purpose**: Density-based customer grouping
        - **Method**: Identifies clusters of varying shapes
        - **Features**: Can detect outliers and noise
        - **Output**: Flexible cluster shapes
        - **Best For**: Complex customer patterns
        """)
    
    # Feature importance
    st.markdown("### Feature Analysis")
    
    if clustering_info:
        st.markdown("**Key Features Used for Segmentation:**")
        for i, feature in enumerate(clustering_info['feature_columns'], 1):
            st.markdown(f"{i}. **{feature}** - Primary segmentation factor")
        
        # Cluster centers
        st.markdown("### AI-Discovered Segment Centers")
        if hasattr(kmeans_model, 'cluster_centers_'):
            centers_original = scaler.inverse_transform(kmeans_model.cluster_centers_)
            centers_df = pd.DataFrame(
                centers_original,
                columns=clustering_info['feature_columns'],
                index=[f"Segment {i}" for i in range(len(centers_original))]
            )
            st.dataframe(centers_df.round(2), use_container_width=True)

def about_page():
    st.markdown("## About Customer Intelligence Hub")
    
    st.markdown("""
    ### Welcome to the Future of Customer Analytics
    
    This advanced Customer Intelligence Hub leverages cutting-edge artificial intelligence to transform 
    raw customer data into actionable business insights. Our platform uses sophisticated machine learning 
    algorithms to segment customers and predict their behavior patterns.
    """)
    
    # Features grid
    st.markdown("### Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### AI-Powered Prediction
        - Real-time customer segmentation
        - Advanced machine learning models
        - Instant behavioral predictions
        
        #### Advanced Analytics
        - Comprehensive segment analysis
        - Market trend identification
        - Performance metrics tracking
        """)
    
    with col2:
        st.markdown("""
        #### Interactive Visualizations
        - Dynamic 3D customer mapping
        - Real-time data exploration
        - Customizable dashboard views
        
        #### Model Intelligence
        - Multi-algorithm comparison
        - Performance optimization
        - Automated insights generation
        """)
    
    # Technology stack
    st.markdown("### Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **Machine Learning**
        - scikit-learn
        - K-Means Clustering
        - DBSCAN Algorithm
        - Silhouette Analysis
        """)
    
    with tech_col2:
        st.markdown("""
        **Data Visualization**
        - Plotly Interactive Charts
        - Streamlit Web Framework
        - Matplotlib & Seaborn
        - 3D Plotting
        """)
    
    with tech_col3:
        st.markdown("""
        **Development**
        - Python 3.8+
        - Pandas Data Processing
        - NumPy Scientific Computing
        - Modern Web UI
        """)
    
    # Business applications
    st.markdown("### Business Applications")
    
    applications = [
        "**Targeted Marketing** - Personalized campaigns for each customer segment",
        "**Product Recommendations** - AI-driven suggestions based on behavior",
        "**Sales Optimization** - Identify high-value customer opportunities", 
        "**Customer Retention** - Proactive engagement strategies",
        "**Market Research** - Deep insights into customer preferences",
        "**Growth Strategy** - Data-driven business expansion planning"
    ]
    
    for app in applications:
        st.markdown(app)
    
    # Contact info
    st.markdown("### Support & Contact")
    st.markdown("""
    For technical support, feature requests, or business inquiries, please contact our team.
    
    **Built with ❤️ using Streamlit and advanced AI technologies**
    """)

def add_footer():
    st.markdown("""
    <div class="footer">
        <h4>Customer Intelligence Hub</h4>
        <p>Powered by Advanced AI & Machine Learning | Built with Streamlit</p>
        <p>© 2024 - Transforming Customer Data into Business Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
    add_footer()