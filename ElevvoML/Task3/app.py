

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle
import io

# Set page config
st.set_page_config(
    page_title="Forest Cover Type Classification",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Forest cover type names
COVER_TYPES = {
    1: 'Spruce/Fir',
    2: 'Lodgepole Pine', 
    3: 'Ponderosa Pine',
    4: 'Cottonwood/Willow',
    5: 'Aspen',
    6: 'Douglas-fir',
    7: 'Krummholz'
}

# Feature names
FEATURE_NAMES = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

# Add wilderness area and soil type columns
WILDERNESS_AREAS = [f'Wilderness_Area_{i}' for i in range(1, 5)]
SOIL_TYPES = [f'Soil_Type_{i}' for i in range(1, 41)]
FEATURE_NAMES.extend(WILDERNESS_AREAS)
FEATURE_NAMES.extend(SOIL_TYPES)

@st.cache_data
def load_sample_data():
    """Load the sample dataset."""
    try:
        data = pd.read_csv('covtype.data.gz', header=None, compression='gzip')
        columns = FEATURE_NAMES + ['Cover_Type']
        data.columns = columns
        return data
    except FileNotFoundError:
        st.error("Sample dataset 'covtype.data.gz' not found. Please upload a CSV file.")
        return None

@st.cache_resource
def train_models(X_train, y_train, X_train_scaled):
    """Train all models and cache them."""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=20,  # Ultra fast training
            max_depth=10,      # Limit depth for speed
            random_state=42, 
            n_jobs=-1,
            class_weight='balanced'
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=20,  # Ultra fast training
            max_depth=5,       # Limit depth for speed
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        ),
        'Logistic Regression': LogisticRegression(
            random_state=42,
            max_iter=500,      # Reduced iterations
            class_weight='balanced'
        ),
        'SVM': SVC(
            random_state=42,
            class_weight='balanced',
            probability=False,  # Disable probability for speed
            kernel='linear'     # Use linear kernel for speed
        )
    }
    
    trained_models = {}
    for name, model in models.items():
        if name in ['Logistic Regression', 'SVM']:
            X_use = X_train_scaled
        else:
            X_use = X_train
        
        model.fit(X_use, y_train)
        trained_models[name] = model
    
    return trained_models

def preprocess_data(data):
    """Preprocess the data for machine learning."""
    X = data.drop('Cover_Type', axis=1)
    y = data['Cover_Type']
    
    # Encode target variable to start from 0 (required by XGBoost)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Scale the features (only quantitative features)
    scaler = StandardScaler()
    quantitative_features = FEATURE_NAMES[:10]
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[quantitative_features] = scaler.fit_transform(X_train[quantitative_features])
    X_test_scaled[quantitative_features] = scaler.transform(X_test[quantitative_features])
    
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoder

def plot_confusion_matrix(y_true, y_pred, model_name, accuracy, label_encoder=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    
    # Get unique labels for axis labels
    if label_encoder is not None:
        unique_labels = sorted(label_encoder.classes_)
        label_names = [COVER_TYPES[i] for i in unique_labels]
    else:
        unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
        label_names = [COVER_TYPES[i] for i in unique_labels]
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=label_names,
               yticklabels=label_names)
    
    plt.title(f'{model_name}\nAccuracy: {accuracy:.4f}', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return plt

def plot_feature_importance(model, model_name, top_n=15):
    """Plot feature importance for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance = model.feature_importances_
    
    # Create feature importance dataframe
    feature_importance_df = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance_df, x='importance', y='feature', palette='viridis')
    plt.title(f'{model_name} - Top {top_n} Features', fontsize=14, fontweight='bold')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    
    return plt

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("🌲 Forest Cover Type Classification")
    st.markdown("**Predict forest cover types based on cartographic and environmental features**")
    
    # Sidebar
    st.sidebar.header("📊 Data Source")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Sample Dataset", "Upload CSV File"]
    )
    
    data = None
    
    if data_source == "Sample Dataset":
        st.sidebar.info("Using the UCI Covertype dataset")
        data = load_sample_data()
    else:
        st.sidebar.info("Upload your own CSV file")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload a CSV file with the same format as the UCI Covertype dataset"
        )
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success("File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
    
    if data is not None:
        # Main content
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Exploration", "🤖 Model Training", "📊 Predictions", "📋 Results"])
        
        with tab1:
            st.header("Data Exploration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Dataset Information")
                st.write(f"**Total samples:** {len(data):,}")
                st.write(f"**Total features:** {len(FEATURE_NAMES)}")
                st.write(f"**Missing values:** {data.isnull().sum().sum()}")
                
                # Class distribution
                st.subheader("Class Distribution")
                class_counts = data['Cover_Type'].value_counts().sort_index()
                for cover_type, count in class_counts.items():
                    percentage = (count / len(data)) * 100
                    st.write(f"**{cover_type}** ({COVER_TYPES[cover_type]}): {count:,} ({percentage:.2f}%)")
            
            with col2:
                st.subheader("Data Preview")
                st.dataframe(data.head(10))
            
            # Visualizations
            st.subheader("Data Visualizations")
            
            # Class distribution plot
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Bar plot
            axes[0].bar(class_counts.index, class_counts.values, color='skyblue', edgecolor='black')
            axes[0].set_title('Forest Cover Type Distribution', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Cover Type')
            axes[0].set_ylabel('Count')
            axes[0].set_xticks(class_counts.index)
            axes[0].set_xticklabels([f'{i}\n({COVER_TYPES[i]})' for i in class_counts.index], rotation=45)
            
            # Pie chart
            axes[1].pie(class_counts.values, labels=[COVER_TYPES[i] for i in class_counts.index], 
                       autopct='%1.1f%%', startangle=90)
            axes[1].set_title('Cover Type Distribution (Percentage)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Elevation distribution
            st.subheader("Elevation Distribution by Cover Type")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for cover_type in sorted(data['Cover_Type'].unique()):
                subset = data[data['Cover_Type'] == cover_type]['Elevation']
                ax.hist(subset, alpha=0.6, label=f'{cover_type} ({COVER_TYPES[cover_type]})', bins=30)
            
            ax.set_xlabel('Elevation (meters)')
            ax.set_ylabel('Frequency')
            ax.set_title('Elevation Distribution by Cover Type', fontsize=14, fontweight='bold')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            st.header("Model Training")
            
            # Add option to use smaller dataset for faster training
            use_fast_mode = st.checkbox("🚀 Fast Mode (Use smaller dataset for faster training)", value=True)
            
            # Add ultra fast mode
            ultra_fast_mode = st.checkbox("⚡ Ultra Fast Mode (Use very small dataset for instant training)", value=False)
            
            if st.button("🚀 Train Models", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    # Use smaller dataset if fast mode is enabled
                    if ultra_fast_mode and len(data) > 10000:
                        data_sample = data.sample(10000, random_state=42)
                        st.info(f"Using {len(data_sample):,} samples for ultra fast training")
                    elif use_fast_mode and len(data) > 20000:
                        data_sample = data.sample(20000, random_state=42)
                        st.info(f"Using {len(data_sample):,} samples for faster training")
                    else:
                        data_sample = data
                    
                    # Preprocess data
                    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler, label_encoder = preprocess_data(data_sample)
                    
                    # Train models
                    trained_models = train_models(X_train, y_train, X_train_scaled)
                    
                    # Store in session state
                    st.session_state['models'] = trained_models
                    st.session_state['X_test'] = X_test
                    st.session_state['X_test_scaled'] = X_test_scaled
                    st.session_state['y_test'] = y_test
                    st.session_state['scaler'] = scaler
                    st.session_state['label_encoder'] = label_encoder
                    
                    st.success("✅ Models trained successfully!")
            
            if 'models' in st.session_state:
                st.subheader("Model Performance")
                
                models = st.session_state['models']
                X_test = st.session_state['X_test']
                X_test_scaled = st.session_state['X_test_scaled']
                y_test = st.session_state['y_test']
                
                results = {}
                
                for name, model in models.items():
                    if name in ['Logistic Regression', 'SVM']:
                        X_use = X_test_scaled
                    else:
                        X_use = X_test
                    
                    y_pred = model.predict(X_use)
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = {'predictions': y_pred, 'accuracy': accuracy}
                
                # Display results
                st.subheader("Accuracy Scores")
                for name, result in results.items():
                    st.metric(name, f"{result['accuracy']:.4f}")
                
                # Best model
                best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
                st.success(f"🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        with tab3:
            st.header("Make Predictions")
            
            if 'models' in st.session_state:
                st.subheader("Single Prediction")
                
                # Create input form
                with st.form("prediction_form"):
                    st.write("Enter feature values for prediction:")
                    
                    # Quantitative features
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        elevation = st.number_input("Elevation (meters)", min_value=0, max_value=5000, value=2500)
                        aspect = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=180)
                        slope = st.number_input("Slope (degrees)", min_value=0, max_value=90, value=15)
                        h_dist_hydrology = st.number_input("Horizontal Distance to Hydrology (meters)", min_value=0, value=250)
                        v_dist_hydrology = st.number_input("Vertical Distance to Hydrology (meters)", min_value=-200, value=50)
                    
                    with col2:
                        h_dist_roadways = st.number_input("Horizontal Distance to Roadways (meters)", min_value=0, value=2000)
                        hillshade_9am = st.number_input("Hillshade 9am (0-255)", min_value=0, max_value=255, value=200)
                        hillshade_noon = st.number_input("Hillshade Noon (0-255)", min_value=0, max_value=255, value=220)
                        hillshade_3pm = st.number_input("Hillshade 3pm (0-255)", min_value=0, max_value=255, value=150)
                        h_dist_fire = st.number_input("Horizontal Distance to Fire Points (meters)", min_value=0, value=2000)
                    
                    # Wilderness area
                    wilderness_area = st.selectbox("Wilderness Area", [1, 2, 3, 4])
                    
                    # Soil type
                    soil_type = st.selectbox("Soil Type", list(range(1, 41)))
                    
                    submitted = st.form_submit_button("🔮 Predict")
                    
                    if submitted:
                        # Create input array
                        input_data = np.array([[
                            elevation, aspect, slope, h_dist_hydrology, v_dist_hydrology,
                            h_dist_roadways, hillshade_9am, hillshade_noon, hillshade_3pm, h_dist_fire
                        ]])
                        
                        # Add wilderness area and soil type (one-hot encoded)
                        wilderness_encoded = np.zeros(4)
                        wilderness_encoded[wilderness_area - 1] = 1
                        
                        soil_encoded = np.zeros(40)
                        soil_encoded[soil_type - 1] = 1
                        
                        # Combine all features
                        full_input = np.concatenate([input_data, wilderness_encoded.reshape(1, -1), soil_encoded.reshape(1, -1)], axis=1)
                        
                        # Make predictions with all models
                        st.subheader("Predictions")
                        
                        models = st.session_state['models']
                        scaler = st.session_state['scaler']
                        label_encoder = st.session_state['label_encoder']
                        
                        # Scale quantitative features
                        full_input_scaled = full_input.copy()
                        full_input_scaled[:, :10] = scaler.transform(full_input[:, :10])
                        
                        for name, model in models.items():
                            if name in ['Logistic Regression', 'SVM']:
                                pred_encoded = model.predict(full_input_scaled)[0]
                            else:
                                pred_encoded = model.predict(full_input)[0]
                            
                            # Decode prediction back to original label
                            pred = label_encoder.inverse_transform([pred_encoded])[0]
                            st.write(f"**{name}:** {pred} ({COVER_TYPES[pred]})")
                
                # Batch prediction
                st.subheader("Batch Prediction")
                st.write("Upload a CSV file with the same format for batch predictions:")
                
                batch_file = st.file_uploader("Upload CSV for batch prediction", type="csv")
                
                if batch_file is not None:
                    try:
                        batch_data = pd.read_csv(batch_file)
                        
                        if st.button("🔮 Predict Batch"):
                            models = st.session_state['models']
                            scaler = st.session_state['scaler']
                            label_encoder = st.session_state['label_encoder']
                            
                            # Scale quantitative features
                            batch_scaled = batch_data.copy()
                            batch_scaled.iloc[:, :10] = scaler.transform(batch_data.iloc[:, :10])
                            
                            predictions = {}
                            for name, model in models.items():
                                if name in ['Logistic Regression', 'SVM']:
                                    pred_encoded = model.predict(batch_scaled)
                                else:
                                    pred_encoded = model.predict(batch_data)
                                
                                # Decode predictions back to original labels
                                pred = label_encoder.inverse_transform(pred_encoded)
                                predictions[name] = pred
                            
                            # Create results dataframe
                            results_df = batch_data.copy()
                            for name, pred in predictions.items():
                                results_df[f'{name}_Prediction'] = pred
                                results_df[f'{name}_Cover_Type'] = [COVER_TYPES[p] for p in pred]
                            
                            st.subheader("Batch Prediction Results")
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results",
                                data=csv,
                                file_name="forest_cover_predictions.csv",
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"Error processing batch file: {str(e)}")
            
            else:
                st.warning("⚠️ Please train models first in the 'Model Training' tab.")
        
        with tab4:
            st.header("Results and Analysis")
            
            if 'models' in st.session_state:
                models = st.session_state['models']
                X_test = st.session_state['X_test']
                X_test_scaled = st.session_state['X_test_scaled']
                y_test = st.session_state['y_test']
                label_encoder = st.session_state['label_encoder']
                
                # Model comparison
                st.subheader("Model Comparison")
                
                results = {}
                for name, model in models.items():
                    if name in ['Logistic Regression', 'SVM']:
                        X_use = X_test_scaled
                    else:
                        X_use = X_test
                    
                    y_pred = model.predict(X_use)
                    accuracy = accuracy_score(y_test, y_pred)
                    results[name] = {'predictions': y_pred, 'accuracy': accuracy}
                
                # Accuracy comparison
                accuracies = [results[name]['accuracy'] for name in results.keys()]
                model_names = list(results.keys())
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightyellow'])
                ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
                ax.set_ylabel('Accuracy')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, acc in zip(bars, accuracies):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
                
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
                # Confusion matrices
                st.subheader("Confusion Matrices")
                
                for name, result in results.items():
                    st.write(f"**{name}**")
                    fig = plot_confusion_matrix(y_test, result['predictions'], name, result['accuracy'], label_encoder)
                    st.pyplot(fig)
                
                # Feature importance
                st.subheader("Feature Importance")
                
                for name, model in models.items():
                    if hasattr(model, 'feature_importances_'):
                        st.write(f"**{name}**")
                        fig = plot_feature_importance(model, name)
                        if fig:
                            st.pyplot(fig)
                        
                        # Top features table
                        importance = model.feature_importances_
                        feature_importance_df = pd.DataFrame({
                            'Feature': FEATURE_NAMES,
                            'Importance': importance
                        }).sort_values('Importance', ascending=False).head(10)
                        
                        st.dataframe(feature_importance_df)
                
                # Classification reports
                st.subheader("Detailed Classification Reports")
                
                for name, result in results.items():
                    with st.expander(f"{name} - Classification Report"):
                        # Decode y_test back to original labels for classification report
                        y_test_decoded = label_encoder.inverse_transform(y_test)
                        y_pred_decoded = label_encoder.inverse_transform(result['predictions'])
                        
                        report = classification_report(
                            y_test_decoded, y_pred_decoded,
                            target_names=[COVER_TYPES[i] for i in sorted(COVER_TYPES.keys())],
                            output_dict=True
                        )
                        
                        # Convert to dataframe for better display
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df)
            
            else:
                st.warning("⚠️ Please train models first in the 'Model Training' tab.")
    
    else:
        st.info("👆 Please select a data source from the sidebar to get started.")

if __name__ == "__main__":
    main()