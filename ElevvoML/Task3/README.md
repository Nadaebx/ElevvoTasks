# Forest Cover Type Classification

A comprehensive machine learning system for predicting forest cover types based on cartographic and environmental features using the UCI Covertype dataset. This project includes both a Jupyter notebook for detailed analysis and a Streamlit web application for interactive predictions.

## Overview

This project implements a multi-class classification system that predicts forest cover types from 54 features including elevation, aspect, slope, distances to hydrology/roadways/fire points, hillshade indices, wilderness areas, and soil types.

## Forest Cover Types

The system predicts 7 different forest cover types:
1. **Spruce/Fir** - High elevation coniferous forests
2. **Lodgepole Pine** - Common pine species in the study area
3. **Ponderosa Pine** - Lower elevation pine forests
4. **Cottonwood/Willow** - Riparian vegetation
5. **Aspen** - Deciduous tree species
6. **Douglas-fir** - Large coniferous trees
7. **Krummholz** - Stunted, wind-shaped trees at high elevations

## Features

### Data Analysis
- **Dataset Loading**: Handles compressed data files efficiently
- **Exploratory Data Analysis**: Comprehensive data exploration with visualizations
- **Data Preprocessing**: Feature scaling, train-test splitting, and categorical handling

### Machine Learning Models
- **Random Forest**: Ensemble method with feature importance analysis
- **XGBoost**: Gradient boosting with advanced hyperparameter tuning
- **Logistic Regression**: Linear baseline model
- **Support Vector Machine**: Non-linear classification with RBF kernel

### Model Evaluation
- **Confusion Matrices**: Visual representation of classification performance
- **Feature Importance**: Analysis of most predictive features
- **Cross-Validation**: Robust model evaluation with 5-fold CV
- **Hyperparameter Tuning**: Grid search optimization for best models

### Visualizations
- Class distribution analysis
- Elevation patterns by cover type
- Confusion matrices for all models
- Feature importance rankings
- Cross-validation score comparisons

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Jupyter Notebook Analysis
For detailed machine learning analysis and exploration:
```bash
jupyter notebook forest_cover_classification.ipynb
```

### Streamlit Web Application
For interactive predictions and data exploration:
```bash
streamlit run app.py
```

### Features of the Streamlit App
- **Data Exploration**: Interactive data visualization and statistics
- **Model Training**: Train multiple ML models with one click
- **Single Predictions**: Make predictions for individual samples
- **Batch Predictions**: Upload CSV files for bulk predictions
- **Results Analysis**: View confusion matrices, feature importance, and classification reports
- **CSV Import/Export**: Upload your own data and download prediction results

## Dataset Information

- **Source**: UCI Machine Learning Repository - Covertype Dataset
- **Samples**: 581,012 observations
- **Features**: 54 (10 quantitative + 4 wilderness areas + 40 soil types)
- **Classes**: 7 forest cover types
- **Missing Values**: None

### Feature Categories

**Quantitative Features (10):**
- Elevation (meters)
- Aspect (degrees azimuth)
- Slope (degrees)
- Horizontal Distance to Hydrology (meters)
- Vertical Distance to Hydrology (meters)
- Horizontal Distance to Roadways (meters)
- Hillshade 9am (0-255 index)
- Hillshade Noon (0-255 index)
- Hillshade 3pm (0-255 index)
- Horizontal Distance to Fire Points (meters)

**Categorical Features (44):**
- 4 Wilderness Areas (binary)
- 40 Soil Types (binary)

## Model Performance

The system typically achieves:
- **Random Forest**: ~95% accuracy
- **XGBoost**: ~96% accuracy
- **Logistic Regression**: ~70% accuracy
- **SVM**: ~85% accuracy

## Key Insights

1. **Elevation** is typically the most important feature for forest cover prediction
2. **Distance to hydrology** and **aspect** are also highly predictive
3. **Soil types** provide important categorical information
4. **Tree-based models** (Random Forest, XGBoost) significantly outperform linear models
5. The dataset is well-balanced, making it suitable for classification tasks

## Technical Details

### Data Preprocessing
- Standard scaling for quantitative features
- Stratified train-test split (80/20)
- No missing value handling required

### Model Training
- Class weight balancing for imbalanced datasets
- Cross-validation for robust evaluation
- Grid search for hyperparameter optimization

### Evaluation Metrics
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrices
- Feature importance rankings

## File Structure

```
├── forest_cover_classification.ipynb  # Jupyter notebook for ML analysis
├── app.py                             # Streamlit web application
├── requirements.txt                   # Python dependencies
├── README.md                         # This file
├── covtype.data.gz                   # Compressed dataset
├── covtype.info                      # Dataset documentation
└── old_covtype.info                  # Original documentation
```

## Requirements

- Python 3.7+
- pandas >= 1.5.0
- numpy >= 1.21.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scikit-learn >= 1.1.0
- xgboost >= 1.6.0
- streamlit >= 1.28.0

## Contributing

Feel free to contribute by:
- Adding new models
- Improving feature engineering
- Enhancing visualizations
- Optimizing hyperparameters

## License

This project uses the UCI Covertype dataset, which is available for unlimited reuse with retention of copyright notice for Jock A. Blackard and Colorado State University.
