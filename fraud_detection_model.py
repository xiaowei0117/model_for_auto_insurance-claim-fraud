"""
Insurance Fraud Detection - Machine Learning Pipeline
======================================================
Complete pipeline for fraud detection including:
- Data cleaning
- Feature engineering
- Model training and evaluation
- Hyperparameter tuning

Author: Claude
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                            roc_curve, precision_recall_curve, f1_score, 
                            accuracy_score, precision_score, recall_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL INSPECTION
# ============================================================================

def load_data(filepath):
    """
    Load the insurance fraud dataset
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """
    print("="*80)
    print("LOADING DATA")
    print("="*80)
    
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nFirst few rows:")
    print(df.head())
    
    return df


# ============================================================================
# SECTION 2: DATA CLEANING
# ============================================================================

def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicates, and invalid data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataframe
        
    Returns:
    --------
    df_clean : pandas.DataFrame
        Cleaned dataframe
    """
    print("\n" + "="*80)
    print("DATA CLEANING")
    print("="*80)
    
    df_clean = df.copy()
    
    # 1. Remove completely empty columns
    print("\n1. Removing empty columns...")
    empty_cols = df_clean.columns[df_clean.isnull().all()].tolist()
    if empty_cols:
        print(f"   Removing columns: {empty_cols}")
        df_clean = df_clean.drop(columns=empty_cols)
    else:
        print("   No empty columns found")
    
    # 2. Handle missing values
    print("\n2. Handling missing values...")
    missing_before = df_clean.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    
    # For 'authorities_contacted' - fill with 'None' (meaning no authority)
    if 'authorities_contacted' in df_clean.columns:
        df_clean['authorities_contacted'].fillna('None', inplace=True)
        print("   - Filled 'authorities_contacted' missing values with 'None'")
    
    # For property_damage and police_report_available - '?' is essentially missing
    # Keep '?' as a separate category since it might be informative for fraud detection
    
    missing_after = df_clean.isnull().sum().sum()
    print(f"   Missing values after: {missing_after}")
    
    # 3. Check for duplicates
    print("\n3. Checking for duplicates...")
    duplicates = df_clean.duplicated().sum()
    if duplicates > 0:
        print(f"   Found {duplicates} duplicate rows - removing...")
        df_clean = df_clean.drop_duplicates()
    else:
        print("   No duplicates found")
    
    # 4. Convert target variable to binary (0/1)
    print("\n4. Converting target variable...")
    if 'fraud_reported' in df_clean.columns:
        df_clean['fraud_reported'] = (df_clean['fraud_reported'] == 'Y').astype(int)
        print("   - Converted 'fraud_reported': Y→1, N→0")
        print(f"   - Fraud distribution: {df_clean['fraud_reported'].value_counts().to_dict()}")
    
    # 5. Data type corrections
    print("\n5. Correcting data types...")
    
    # Convert dates to datetime
    date_columns = ['policy_bind_date', 'incident_date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col])
            print(f"   - Converted '{col}' to datetime")
    
    print(f"\nCleaned dataset shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    
    return df_clean


# ============================================================================
# SECTION 3: FEATURE ENGINEERING
# ============================================================================

def engineer_features(df):
    """
    Create new features from existing data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned dataframe
        
    Returns:
    --------
    df_engineered : pandas.DataFrame
        Dataframe with new features
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df_eng = df.copy()
    
    # 1. Date-based features
    print("\n1. Creating date-based features...")
    if 'policy_bind_date' in df_eng.columns and 'incident_date' in df_eng.columns:
        # Days between policy binding and incident
        df_eng['days_policy_to_incident'] = (df_eng['incident_date'] - 
                                              df_eng['policy_bind_date']).dt.days
        print("   - Created 'days_policy_to_incident'")
        
        # Extract temporal features from incident date
        df_eng['incident_month'] = df_eng['incident_date'].dt.month
        df_eng['incident_day_of_week'] = df_eng['incident_date'].dt.dayofweek
        df_eng['incident_day_of_month'] = df_eng['incident_date'].dt.day
        print("   - Created incident temporal features (month, day of week, day of month)")
        
        # Policy age at incident
        df_eng['policy_age_years'] = df_eng['days_policy_to_incident'] / 365.25
        print("   - Created 'policy_age_years'")
    
    # 2. Claim-based features
    print("\n2. Creating claim-based features...")
    if all(col in df_eng.columns for col in ['total_claim_amount', 'injury_claim', 
                                               'property_claim', 'vehicle_claim']):
        # Claim ratios
        df_eng['injury_claim_ratio'] = df_eng['injury_claim'] / (df_eng['total_claim_amount'] + 1)
        df_eng['property_claim_ratio'] = df_eng['property_claim'] / (df_eng['total_claim_amount'] + 1)
        df_eng['vehicle_claim_ratio'] = df_eng['vehicle_claim'] / (df_eng['total_claim_amount'] + 1)
        print("   - Created claim ratio features")
        
        # High claim flag
        claim_threshold = df_eng['total_claim_amount'].quantile(0.75)
        df_eng['high_claim_flag'] = (df_eng['total_claim_amount'] > claim_threshold).astype(int)
        print(f"   - Created 'high_claim_flag' (threshold: ${claim_threshold:,.2f})")
    
    # 3. Incident severity encoding (ordinal)
    print("\n3. Creating severity score...")
    if 'incident_severity' in df_eng.columns:
        severity_map = {
            'Trivial Damage': 1,
            'Minor Damage': 2,
            'Major Damage': 3,
            'Total Loss': 4
        }
        df_eng['severity_score'] = df_eng['incident_severity'].map(severity_map)
        print("   - Created 'severity_score' (1=Trivial, 2=Minor, 3=Major, 4=Total Loss)")
    
    # 4. Vehicle age
    print("\n4. Creating vehicle age...")
    if 'auto_year' in df_eng.columns:
        current_year = 2015  # Based on incident dates
        df_eng['vehicle_age'] = current_year - df_eng['auto_year']
        print(f"   - Created 'vehicle_age' (current year: {current_year})")
    
    # 5. Customer tenure categories
    print("\n5. Creating customer tenure categories...")
    if 'months_as_customer' in df_eng.columns:
        df_eng['tenure_category'] = pd.cut(df_eng['months_as_customer'], 
                                            bins=[0, 50, 150, 300, 500],
                                            labels=['New', 'Regular', 'Loyal', 'VIP'])
        print("   - Created 'tenure_category' (New/Regular/Loyal/VIP)")
    
    # 6. Incident context features
    print("\n6. Creating incident context features...")
    
    # Time of day category
    if 'incident_hour_of_the_day' in df_eng.columns:
        def categorize_hour(hour):
            if 0 <= hour < 6:
                return 'Night'
            elif 6 <= hour < 12:
                return 'Morning'
            elif 12 <= hour < 18:
                return 'Afternoon'
            else:
                return 'Evening'
        
        df_eng['incident_time_category'] = df_eng['incident_hour_of_the_day'].apply(categorize_hour)
        print("   - Created 'incident_time_category' (Night/Morning/Afternoon/Evening)")
    
    # Multiple parties involved flag
    if 'number_of_vehicles_involved' in df_eng.columns:
        df_eng['multi_vehicle_flag'] = (df_eng['number_of_vehicles_involved'] > 1).astype(int)
        print("   - Created 'multi_vehicle_flag'")
    
    # Witnesses present flag
    if 'witnesses' in df_eng.columns:
        df_eng['witnesses_present'] = (df_eng['witnesses'] > 0).astype(int)
        print("   - Created 'witnesses_present'")
    
    # 7. Policy coverage score
    print("\n7. Creating policy coverage score...")
    if 'policy_csl' in df_eng.columns:
        csl_map = {
            '100/300': 1,
            '250/500': 2,
            '500/1000': 3
        }
        df_eng['coverage_level'] = df_eng['policy_csl'].map(csl_map)
        print("   - Created 'coverage_level' (1=Low, 2=Medium, 3=High)")
    
    # 8. Property damage and police report interaction
    print("\n8. Creating documentation features...")
    if 'property_damage' in df_eng.columns and 'police_report_available' in df_eng.columns:
        df_eng['property_damage_binary'] = (df_eng['property_damage'] == 'YES').astype(int)
        df_eng['police_report_binary'] = (df_eng['police_report_available'] == 'YES').astype(int)
        df_eng['documentation_score'] = df_eng['property_damage_binary'] + df_eng['police_report_binary']
        print("   - Created 'documentation_score' (0=None, 1=One, 2=Both)")
    
    # 9. State mismatch (policy state vs incident state)
    print("\n9. Creating location features...")
    if 'policy_state' in df_eng.columns and 'incident_state' in df_eng.columns:
        df_eng['state_mismatch'] = (df_eng['policy_state'] != df_eng['incident_state']).astype(int)
        print("   - Created 'state_mismatch' flag")
    
    # 10. Capital gains/loss features
    print("\n10. Creating financial features...")
    if 'capital-gains' in df_eng.columns and 'capital-loss' in df_eng.columns:
        df_eng['net_capital'] = df_eng['capital-gains'] - df_eng['capital-loss']
        df_eng['has_capital_gains'] = (df_eng['capital-gains'] > 0).astype(int)
        df_eng['has_capital_loss'] = (df_eng['capital-loss'] > 0).astype(int)
        print("   - Created capital gains/loss features")
    
    print(f"\nEngineered dataset shape: {df_eng.shape[0]} rows × {df_eng.shape[1]} columns")
    print(f"New features added: {df_eng.shape[1] - df.shape[1]}")
    
    return df_eng


# ============================================================================
# SECTION 4: FEATURE SELECTION AND PREPROCESSING
# ============================================================================

def prepare_features(df, target_col='fraud_reported'):
    """
    Prepare features for modeling by encoding categorical variables and selecting features
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Engineered dataframe
    target_col : str
        Name of target column
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    feature_names : list
        List of feature names
    encoders : dict
        Dictionary of label encoders for each categorical column
    """
    print("\n" + "="*80)
    print("FEATURE PREPARATION")
    print("="*80)
    
    df_prep = df.copy()
    
    # Separate target variable
    y = df_prep[target_col]
    df_prep = df_prep.drop(columns=[target_col])
    
    # Remove identifier columns and date columns
    cols_to_drop = ['policy_number', 'policy_bind_date', 'incident_date', 
                    'incident_location', 'insured_zip']
    cols_to_drop = [col for col in cols_to_drop if col in df_prep.columns]
    df_prep = df_prep.drop(columns=cols_to_drop)
    print(f"\nDropped identifier columns: {cols_to_drop}")
    
    # Identify categorical and numerical columns
    categorical_cols = df_prep.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df_prep.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols[:10]}...")
    print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols[:10]}...")
    
    # Encode categorical variables
    print("\nEncoding categorical variables...")
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df_prep[col] = le.fit_transform(df_prep[col].astype(str))
        encoders[col] = le
        print(f"   - Encoded '{col}' ({len(le.classes_)} categories)")
    
    X = df_prep
    feature_names = X.columns.tolist()
    
    print(f"\nFinal feature matrix shape: {X.shape}")
    print(f"Total features: {len(feature_names)}")
    print(f"Target variable distribution:\n{y.value_counts()}")
    
    return X, y, feature_names, encoders


# ============================================================================
# SECTION 5: MODEL TRAINING AND EVALUATION
# ============================================================================

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features
    y : pandas.Series
        Target
    test_size : float
        Proportion of test set
    random_state : int
        Random seed
        
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    print(f"\nTraining set fraud distribution:\n{y_train.value_counts()}")
    print(f"\nTesting set fraud distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test


def handle_imbalance(X_train, y_train, method='smote'):
    """
    Handle class imbalance using SMOTE or undersampling
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : pandas.Series
        Training target
    method : str
        'smote', 'undersample', or 'combined'
        
    Returns:
    --------
    X_resampled, y_resampled
    """
    print("\n" + "="*80)
    print("HANDLING CLASS IMBALANCE")
    print("="*80)
    
    print(f"\nOriginal distribution:\n{y_train.value_counts()}")
    print(f"Imbalance ratio: {y_train.value_counts()[0] / y_train.value_counts()[1]:.2f}:1")
    
    if method == 'smote':
        print("\nApplying SMOTE (Synthetic Minority Over-sampling)...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
    elif method == 'undersample':
        print("\nApplying Random Under-sampling...")
        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
    elif method == 'combined':
        print("\nApplying Combined (SMOTE + Under-sampling)...")
        over = SMOTE(sampling_strategy=0.5, random_state=42)
        under = RandomUnderSampler(sampling_strategy=0.8, random_state=42)
        X_resampled, y_resampled = over.fit_resample(X_train, y_train)
        X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
    else:
        print("\nNo resampling applied")
        X_resampled, y_resampled = X_train, y_train
    
    print(f"\nResampled distribution:\n{pd.Series(y_resampled).value_counts()}")
    print(f"New shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled


def train_models(X_train, y_train, X_test, y_test):
    """
    Train multiple models and compare performance
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Testing data
        
    Returns:
    --------
    models : dict
        Dictionary of trained models
    results : dict
        Dictionary of model performance metrics
    """
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    trained_models = {}
    
    print("\nTraining and evaluating models...\n")
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Train model
        if 'SVM' in name:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.decision_function(X_test_scaled)
        else:
            model.fit(X_train_scaled if name in ['Logistic Regression', 'SVM'] else X_train, 
                     y_train)
            y_pred = model.predict(X_test_scaled if name in ['Logistic Regression', 'SVM'] else X_test)
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(
                    X_test_scaled if name in ['Logistic Regression', 'SVM'] else X_test
                )[:, 1]
            else:
                y_pred_proba = y_pred
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        if hasattr(model, 'predict_proba') or hasattr(model, 'decision_function'):
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            roc_auc = None
        
        # Store results
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        trained_models[name] = model
        
        # Print results
        print(f"\nResults:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Fraud', 'Fraud']))
    
    # Store scaler for later use
    trained_models['scaler'] = scaler
    
    return trained_models, results


def compare_models(results):
    """
    Compare model performance
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    """
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results],
        'ROC-AUC': [results[m]['roc_auc'] if results[m]['roc_auc'] else 0 for m in results]
    })
    
    comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
    
    print("\n" + comparison_df.to_string(index=False))
    
    # Best model
    best_model = comparison_df.iloc[0]['Model']
    best_f1 = comparison_df.iloc[0]['F1-Score']
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"{'='*60}")
    
    return comparison_df


# ============================================================================
# SECTION 6: HYPERPARAMETER TUNING
# ============================================================================

def tune_best_model(X_train, y_train, X_test, y_test, model_name='Random Forest'):
    """
    Perform hyperparameter tuning on the best model
    
    Parameters:
    -----------
    X_train, y_train : Training data
    X_test, y_test : Testing data
    model_name : str
        Name of model to tune
        
    Returns:
    --------
    best_model : sklearn model
        Tuned model
    """
    print("\n" + "="*80)
    print(f"HYPERPARAMETER TUNING - {model_name}")
    print("="*80)
    
    if model_name == 'Random Forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        
    elif model_name == 'XGBoost':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
        
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'min_samples_split': [2, 5, 10]
        }
        model = GradientBoostingClassifier(random_state=42)
    else:
        print(f"Tuning not configured for {model_name}")
        return None
    
    print(f"\nParameter grid: {param_grid}")
    print("\nPerforming Grid Search with 5-fold cross-validation...")
    print("This may take several minutes...\n")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
    
    # Evaluate on test set
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print(f"\nTest set performance:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
    
    return best_model


# ============================================================================
# SECTION 7: VISUALIZATION
# ============================================================================

def plot_model_results(models, results, y_test, feature_names, save_path='model_results.png'):
    """
    Create comprehensive visualization of model results
    
    Parameters:
    -----------
    models : dict
        Trained models
    results : dict
        Model results
    y_test : array
        Test target values
    feature_names : list
        List of feature names
    save_path : str
        Path to save visualization
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Model Comparison - Metrics
    ax1 = fig.add_subplot(gs[0, :2])
    metrics_df = pd.DataFrame({
        'Accuracy': [results[m]['accuracy'] for m in results],
        'Precision': [results[m]['precision'] for m in results],
        'Recall': [results[m]['recall'] for m in results],
        'F1-Score': [results[m]['f1'] for m in results]
    }, index=list(results.keys()))
    
    metrics_df.plot(kind='bar', ax=ax1, width=0.8, edgecolor='black')
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.legend(loc='lower right')
    ax1.set_ylim([0, 1])
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. ROC Curves
    ax2 = fig.add_subplot(gs[0, 2])
    for name in results:
        if results[name]['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
            ax2.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})", linewidth=2)
    
    ax2.plot([0, 1], [0, 1], 'k--', label='Random')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # 3-5. Confusion Matrices for top 3 models
    top_models = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
    
    for idx, (name, result) in enumerate(top_models):
        ax = fig.add_subplot(gs[1, idx])
        cm = confusion_matrix(y_test, result['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=['No Fraud', 'Fraud'],
                   yticklabels=['No Fraud', 'Fraud'])
        ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    # 6. Feature Importance (for tree-based models)
    ax6 = fig.add_subplot(gs[2, :])
    
    # Get the best performing tree-based model
    tree_models = {k: v for k, v in results.items() 
                   if 'Random Forest' in k or 'XGBoost' in k or 'Gradient Boosting' in k}
    if tree_models:
        best_tree_model = max(tree_models.items(), key=lambda x: x[1]['f1'])[0]
        model = models[best_tree_model]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            ax6.barh(range(len(indices)), importances[indices], color='#3498db', edgecolor='black')
            ax6.set_yticks(range(len(indices)))
            ax6.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
            ax6.set_xlabel('Importance')
            ax6.set_title(f'Top 20 Feature Importances - {best_tree_model}', 
                         fontsize=14, fontweight='bold')
            ax6.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Fraud Detection Model Results', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


# ============================================================================
# SECTION 8: MODEL SAVING AND DEPLOYMENT
# ============================================================================

def save_model(model, scaler, encoders, feature_names, filepath='fraud_detection_model.pkl'):
    """
    Save trained model and preprocessing objects
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    scaler : StandardScaler
        Fitted scaler
    encoders : dict
        Label encoders
    feature_names : list
        Feature names
    filepath : str
        Path to save model
    """
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_package, filepath)
    print(f"\nModel saved successfully to: {filepath}")
    print(f"Package includes:")
    print(f"  - Trained model")
    print(f"  - Feature scaler")
    print(f"  - Label encoders ({len(encoders)} encoders)")
    print(f"  - Feature names ({len(feature_names)} features)")
    print(f"  - Timestamp: {model_package['timestamp']}")


def load_model(filepath='fraud_detection_model.pkl'):
    """
    Load saved model
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
        
    Returns:
    --------
    model_package : dict
        Dictionary containing model and preprocessing objects
    """
    print("\nLoading model from:", filepath)
    model_package = joblib.load(filepath)
    print("Model loaded successfully!")
    print(f"Model saved on: {model_package['timestamp']}")
    return model_package


def predict_fraud(model_package, new_data):
    """
    Make predictions on new data
    
    Parameters:
    -----------
    model_package : dict
        Loaded model package
    new_data : pandas.DataFrame
        New data to predict
        
    Returns:
    --------
    predictions : array
        Fraud predictions (0 or 1)
    probabilities : array
        Fraud probabilities
    """
    model = model_package['model']
    scaler = model_package['scaler']
    encoders = model_package['encoders']
    feature_names = model_package['feature_names']
    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in new_data.columns:
            new_data[col] = encoder.transform(new_data[col].astype(str))
    
    # Ensure all features are present
    for feat in feature_names:
        if feat not in new_data.columns:
            new_data[feat] = 0
    
    # Reorder columns
    new_data = new_data[feature_names]
    
    # Scale features
    new_data_scaled = scaler.transform(new_data)
    
    # Predict
    predictions = model.predict(new_data_scaled)
    probabilities = model.predict_proba(new_data_scaled)[:, 1]
    
    return predictions, probabilities


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================

def main(filepath, use_smote=True, tune_hyperparameters=False):
    """
    Main pipeline execution
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    use_smote : bool
        Whether to use SMOTE for class imbalance
    tune_hyperparameters : bool
        Whether to perform hyperparameter tuning
    """
    print("\n" + "="*80)
    print("INSURANCE FRAUD DETECTION - MACHINE LEARNING PIPELINE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # 1. Load data
    df = load_data(filepath)
    
    # 2. Clean data
    df_clean = clean_data(df)
    
    # 3. Engineer features
    df_engineered = engineer_features(df_clean)
    
    # 4. Prepare features
    X, y, feature_names, encoders = prepare_features(df_engineered)
    
    # 5. Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 6. Handle class imbalance
    if use_smote:
        X_train, y_train = handle_imbalance(X_train, y_train, method='smote')
    
    # 7. Train models
    models, results = train_models(X_train, y_train, X_test, y_test)
    
    # 8. Compare models
    comparison_df = compare_models(results)
    
    # 9. Hyperparameter tuning (optional)
    if tune_hyperparameters:
        best_model_name = comparison_df.iloc[0]['Model']
        print(f"\nTuning {best_model_name}...")
        tuned_model = tune_best_model(X_train, y_train, X_test, y_test, best_model_name)
        models['Tuned_' + best_model_name] = tuned_model
    
    # 10. Create visualizations
    plot_model_results(models, results, y_test, feature_names)
    
    # 11. Save best model
    best_model_name = comparison_df.iloc[0]['Model']
    save_model(
        models[best_model_name],
        models['scaler'],
        encoders,
        feature_names,
        f'fraud_model_{best_model_name.replace(" ", "_").lower()}.pkl'
    )
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nGenerated files:")
    print("  - model_results.png (visualizations)")
    print(f"  - fraud_model_{best_model_name.replace(' ', '_').lower()}.pkl (trained model)")
    print("="*80 + "\n")
    
    return models, results, comparison_df


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    FILE_PATH = 'insurance_fraud_claims.csv'  # Update this path
    USE_SMOTE = True  # Set to False to train without SMOTE
    TUNE_HYPERPARAMETERS = False  # Set to True to enable hyperparameter tuning (slower)
    
    # Run pipeline
    models, results, comparison = main(
        filepath=FILE_PATH,
        use_smote=USE_SMOTE,
        tune_hyperparameters=TUNE_HYPERPARAMETERS
    )
    
    print("\n" + "="*80)
    print("To make predictions on new data, use:")
    print("="*80)
    print("""
    # Load the saved model
    model_package = load_model('fraud_model_random_forest.pkl')
    
    # Prepare new data (same format as training data)
    new_data = pd.read_csv('new_claims.csv')
    
    # Make predictions
    predictions, probabilities = predict_fraud(model_package, new_data)
    
    # View results
    results_df = pd.DataFrame({
        'Prediction': predictions,
        'Fraud_Probability': probabilities
    })
    print(results_df)
    """)
