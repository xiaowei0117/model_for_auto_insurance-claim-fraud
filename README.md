# Insurance Fraud Claims - Exploratory Data Analysis
     
Comprehensive EDA on insurance fraud claims dataset using Python.
     
## Dataset
Download the dataset from Kaggle: [Auto Insurance Claims Data](https://www.kaggle.com/datasets/...)
     
**Important**: Place `insurance_fraud_claims.csv` in the same directory as the Python script.
     
## Installation
     
Install required packages:
```bash
pip install -r requirements.txt
```
     
## Usage
```bash
python insurance_fraud_eda_complete.py
```
     
## Features
     
- Complete statistical analysis of 1,000 insurance claims
- 12+ comprehensive visualizations
- Fraud pattern detection and analysis
- Correlation analysis
- Demographic and geographic insights
     
## Output
     
The script generates:
- `eda_visualization.png` - Main dashboard with 12 plots
- `advanced_analysis.png` - Correlation heatmap and advanced charts
- Console output with detailed statistics
     
## Key Findings
     
- Fraud rate: 24.7%
- Major Damage incidents: 60.5% fraud rate (highest risk indicator)
- Fraudulent claims average $10,000 higher than legitimate claims
     


# Insurance Fraud Detection - Machine Learning Model

A comprehensive machine learning pipeline for detecting fraudulent insurance claims with **automated data cleaning**, **advanced feature engineering**, and **multiple model comparison**.

---

## 🎯 Project Overview

This project builds a production-ready fraud detection system that:
- Achieves **high accuracy** in identifying fraudulent claims
- Uses **SMOTE** to handle class imbalance
- Compares **6+ machine learning algorithms**
- Provides **interpretable results** with feature importance
- Includes **complete preprocessing pipeline** for deployment

---

## 📊 Dataset

**Source**: [Kaggle - Auto Insurance Claims Data](https://www.kaggle.com/datasets/...)

**Important**: Download the CSV file and place it in the project directory.

**Dataset Characteristics**:
- 1,000 insurance claims
- 40 original features
- Target: Fraud detection (24.7% fraud rate)
- Mix of numerical and categorical features

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 2. Download Dataset

Download `insurance_fraud_claims.csv` from Kaggle and place it in the project folder.

### 3. Run the Pipeline

```bash
python fraud_detection_model.py
```

The script will:
- ✅ Clean the data
- ✅ Engineer 30+ new features
- ✅ Train 6 different models
- ✅ Generate performance visualizations
- ✅ Save the best model

---

## 📁 Project Structure

```
insurance-fraud-detection/
├── fraud_detection_model.py       # Main ML pipeline
├── insurance_fraud_claims.csv     # Dataset (download from Kaggle)
├── requirements_ml.txt             # Python dependencies
├── README.md                       # This file
│
# Generated outputs:
├── model_results.png              # Model comparison visualizations
├── fraud_model_*.pkl              # Trained model (saved automatically)
```

---

## 🔧 Pipeline Components

### **1. Data Cleaning**
- Removes empty columns
- Handles missing values intelligently
- Converts target to binary (0/1)
- Validates and corrects data types

### **2. Feature Engineering** (30+ New Features)

**Temporal Features**:
- Days between policy and incident
- Incident month, day of week
- Policy age at incident
- Time of day categories (Night/Morning/Afternoon/Evening)

**Claim Features**:
- Claim ratios (injury/property/vehicle vs total)
- High claim flags
- Severity score (1-4 ordinal encoding)

**Context Features**:
- Vehicle age
- Customer tenure categories
- Multi-vehicle involvement flag
- Witness presence indicator
- Documentation completeness score

**Location Features**:
- State mismatch (policy vs incident location)
- Coverage level encoding

**Financial Features**:
- Net capital (gains - losses)
- Capital gains/loss indicators

### **3. Models Trained**

| Model | Type | Use Case |
|-------|------|----------|
| **Logistic Regression** | Linear | Baseline, interpretable |
| **Decision Tree** | Tree-based | Simple, interpretable |
| **Random Forest** | Ensemble | High accuracy, robust |
| **Gradient Boosting** | Ensemble | High performance |
| **XGBoost** | Gradient Boosting | State-of-the-art |
| **Naive Bayes** | Probabilistic | Fast, simple |

### **4. Class Imbalance Handling**

Uses **SMOTE** (Synthetic Minority Over-sampling Technique):
- Balances fraud vs non-fraud classes
- Prevents model bias toward majority class
- Improves recall for fraud detection

---

## 📈 Model Evaluation Metrics

Each model is evaluated on:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted frauds are actually frauds
- **Recall**: How many actual frauds are detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve

**Key Metric**: **F1-Score** (balances precision and recall)

---

## 🎨 Generated Visualizations

The pipeline generates `model_results.png` containing:

1. **Model Performance Comparison** - Bar chart of all metrics
2. **ROC Curves** - For all models with AUC scores
3. **Confusion Matrices** - For top 3 models
4. **Feature Importance** - Top 20 most important features

---

## 💾 Model Deployment

### Saving the Model

The best model is automatically saved as a pickle file:
```python
fraud_model_random_forest.pkl  # Example filename
```

### Loading and Using the Model

```python
import pandas as pd
from fraud_detection_model import load_model, predict_fraud

# Load the trained model
model_package = load_model('fraud_model_random_forest.pkl')

# Prepare new data (must have same format as training data)
new_claims = pd.read_csv('new_claims.csv')

# Make predictions
predictions, probabilities = predict_fraud(model_package, new_claims)

# View results
results = pd.DataFrame({
    'Claim_ID': new_claims['policy_number'],
    'Fraud_Prediction': predictions,
    'Fraud_Probability': probabilities,
    'Risk_Level': ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.4 else 'LOW' 
                   for p in probabilities]
})

print(results)
```

---

## ⚙️ Configuration Options

Edit these variables at the bottom of `fraud_detection_model.py`:

```python
FILE_PATH = 'insurance_fraud_claims.csv'  # Path to your data
USE_SMOTE = True                           # Enable SMOTE resampling
TUNE_HYPERPARAMETERS = False               # Enable grid search (slow)
```

**Note**: Setting `TUNE_HYPERPARAMETERS = True` will significantly increase runtime but may improve performance.

---

## 🔍 Key Findings from EDA

From the exploratory data analysis:

### **Strongest Fraud Indicators**:
1. **Incident Severity = "Major Damage"** (60.5% fraud rate)
2. **High claim amounts** (avg $60k vs $50k)
3. **Missing documentation** (no police report)
4. **Single/Multi-vehicle collisions** (vs theft)
5. **State mismatch** (policy state ≠ incident state)

### **Feature Importance** (from Random Forest):
Top features typically include:
- `severity_score`
- `total_claim_amount`
- `vehicle_claim`
- `days_policy_to_incident`
- `incident_severity`
- `policy_annual_premium`

---

## 📊 Expected Performance

Typical results (may vary based on data split):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | ~85-90% | ~80-85% | ~75-80% | ~77-82% |
| **XGBoost** | ~85-90% | ~80-85% | ~75-80% | ~77-82% |
| **Gradient Boosting** | ~83-88% | ~78-83% | ~73-78% | ~75-80% |
| **Logistic Regression** | ~78-83% | ~70-75% | ~68-73% | ~69-74% |

**Note**: With SMOTE, recall typically improves at the cost of slight precision decrease.

---

## 🛠️ Troubleshooting

### Import Errors
```bash
# Install missing packages
pip install scikit-learn xgboost imbalanced-learn
```

### Memory Issues
```python
# Reduce dataset size for testing
df = df.sample(n=500, random_state=42)
```

### Slow Training
```python
# Disable hyperparameter tuning
TUNE_HYPERPARAMETERS = False

# Or reduce cross-validation folds
cv=3  # Instead of 5
```

---

## 📚 Advanced Usage

### Custom Feature Engineering

Add your own features in the `engineer_features()` function:

```python
def engineer_features(df):
    df_eng = df.copy()
    
    # Your custom feature
    df_eng['my_feature'] = df_eng['col1'] * df_eng['col2']
    
    return df_eng
```

### Try Different Resampling Strategies

```python
# In handle_imbalance() function
X_train, y_train = handle_imbalance(X_train, y_train, method='combined')
# Options: 'smote', 'undersample', 'combined', None
```

### Add More Models

```python
# In train_models() function
from sklearn.neighbors import KNeighborsClassifier

models['KNN'] = KNeighborsClassifier(n_neighbors=5)
```

---

## 📝 Output Files

After running the pipeline, you'll have:

1. **model_results.png** - Comprehensive visualizations
2. **fraud_model_*.pkl** - Trained model package containing:
   - Trained classifier
   - Feature scaler
   - Label encoders
   - Feature names
   - Timestamp

---

## 🎓 Learning Resources

**Key Concepts Used**:
- SMOTE for imbalanced data
- Ensemble learning (Random Forest, Gradient Boosting)
- Cross-validation
- Hyperparameter tuning with GridSearchCV
- Feature engineering
- Model evaluation metrics

**Recommended Reading**:
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Guide](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

## 🤝 Contributing

Feel free to:
- Add new features
- Try different models
- Improve hyperparameter tuning
- Enhance visualizations

---

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

## 🙏 Acknowledgments

- Dataset: Kaggle Auto Insurance Claims Data
- Libraries: scikit-learn, XGBoost, imbalanced-learn, pandas, matplotlib, seaborn

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments
3. Open an issue on GitHub

---

**Happy Fraud Detection! 🕵️‍♂️**
