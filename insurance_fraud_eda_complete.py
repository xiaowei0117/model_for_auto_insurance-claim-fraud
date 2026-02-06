"""
Insurance Fraud Claims - Complete Exploratory Data Analysis
============================================================
This script performs comprehensive EDA on insurance fraud claims dataset.
Author: Claude
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING AND INITIAL EXPLORATION
# ============================================================================

def load_and_explore_data(filepath):
    """
    Load the dataset and perform initial exploration
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
        
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """
    # Load the data
    df = pd.read_csv(filepath)
    
    # Basic information
    print("="*80)
    print("DATASET OVERVIEW")
    print("="*80)
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumn Names and Types:")
    print(df.dtypes)
    print("\n" + "="*80)
    
    # First few rows
    print("\nFirst 5 rows:")
    print(df.head())
    
    return df


def check_missing_values(df):
    """
    Analyze missing values in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print(missing_df[missing_df['Missing Count'] > 0])
    
    # Total missing values
    total_missing = missing.sum()
    total_cells = df.shape[0] * df.shape[1]
    print(f"\nTotal missing values: {total_missing} ({total_missing/total_cells*100:.2f}%)")


def basic_statistics(df):
    """
    Display basic statistical summaries
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("BASIC STATISTICS - NUMERICAL FEATURES")
    print("="*80)
    print(df.describe())
    
    print("\n" + "="*80)
    print("BASIC STATISTICS - CATEGORICAL FEATURES")
    print("="*80)
    print(df.describe(include='object'))


def analyze_target_variable(df, target_col='fraud_reported'):
    """
    Analyze the target variable distribution
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    """
    print("\n" + "="*80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("="*80)
    print(df[target_col].value_counts())
    
    fraud_rate = (df[target_col]=='Y').sum() / len(df) * 100
    print(f"\nFraud Rate: {fraud_rate:.2f}%")
    print(f"Fraudulent Cases: {(df[target_col]=='Y').sum()}")
    print(f"Legitimate Cases: {(df[target_col]=='N').sum()}")


# ============================================================================
# SECTION 2: CATEGORICAL FEATURES ANALYSIS
# ============================================================================

def analyze_categorical_features(df):
    """
    Analyze all categorical features in the dataset
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("="*80)
    
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if 'fraud_reported' in categorical_cols:
        categorical_cols.remove('fraud_reported')
    
    for col in categorical_cols:
        if df[col].notna().sum() > 0:  # Only show if there's data
            print(f"\n{col}:")
            print("-" * 60)
            print(df[col].value_counts().head(10))
            print(f"Unique values: {df[col].nunique()}")
            print(f"Missing values: {df[col].isnull().sum()} ({df[col].isnull().sum()/len(df)*100:.1f}%)")


# ============================================================================
# SECTION 3: FRAUD ANALYSIS BY DIMENSIONS
# ============================================================================

def fraud_analysis_by_dimensions(df):
    """
    Analyze fraud rates across different dimensions
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("FRAUD RATE BY KEY DIMENSIONS")
    print("="*80)
    
    # By Incident Type
    print("\nBy Incident Type:")
    print("-" * 60)
    incident_fraud = df.groupby('incident_type').agg({
        'fraud_reported': lambda x: f"{(x=='Y').sum()} / {len(x)} ({(x=='Y').sum()/len(x)*100:.1f}%)"
    })
    print(incident_fraud)
    
    # By Incident Severity
    print("\nBy Incident Severity:")
    print("-" * 60)
    severity_fraud = df.groupby('incident_severity').agg({
        'fraud_reported': lambda x: f"{(x=='Y').sum()} / {len(x)} ({(x=='Y').sum()/len(x)*100:.1f}%)"
    })
    print(severity_fraud)
    
    # By Property Damage
    print("\nBy Property Damage:")
    print("-" * 60)
    prop_fraud = df.groupby('property_damage').agg({
        'fraud_reported': lambda x: f"{(x=='Y').sum()} / {len(x)} ({(x=='Y').sum()/len(x)*100:.1f}%)"
    })
    print(prop_fraud)
    
    # By Police Report Available
    print("\nBy Police Report Available:")
    print("-" * 60)
    police_fraud = df.groupby('police_report_available').agg({
        'fraud_reported': lambda x: f"{(x=='Y').sum()} / {len(x)} ({(x=='Y').sum()/len(x)*100:.1f}%)"
    })
    print(police_fraud)
    
    # By Collision Type
    print("\nBy Collision Type:")
    print("-" * 60)
    collision_fraud = df.groupby('collision_type').agg({
        'fraud_reported': lambda x: f"{(x=='Y').sum()} / {len(x)} ({(x=='Y').sum()/len(x)*100:.1f}%)"
    })
    print(collision_fraud)


def claim_amount_analysis(df):
    """
    Analyze claim amounts for fraud vs non-fraud cases
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("CLAIM AMOUNT ANALYSIS")
    print("="*80)
    
    fraud_claims = df[df['fraud_reported'] == 'Y']
    no_fraud_claims = df[df['fraud_reported'] == 'N']
    
    claim_columns = ['total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim']
    
    print("\nAverage Claim Amounts:")
    print("-" * 60)
    for col in claim_columns:
        fraud_avg = fraud_claims[col].mean()
        no_fraud_avg = no_fraud_claims[col].mean()
        diff = fraud_avg - no_fraud_avg
        pct_diff = (diff / no_fraud_avg) * 100
        
        print(f"\n{col.replace('_', ' ').title()}:")
        print(f"  Fraud:     ${fraud_avg:>12,.2f}")
        print(f"  No Fraud:  ${no_fraud_avg:>12,.2f}")
        print(f"  Difference: ${diff:>11,.2f} ({pct_diff:+.1f}%)")
    
    # Statistical summary
    print("\n" + "-" * 60)
    print("\nTotal Claim Amount Statistics:")
    print("\nFraud Cases:")
    print(fraud_claims['total_claim_amount'].describe())
    print("\nNo Fraud Cases:")
    print(no_fraud_claims['total_claim_amount'].describe())


def demographic_analysis(df):
    """
    Analyze demographic patterns in fraud
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("DEMOGRAPHIC PATTERNS")
    print("="*80)
    
    fraud_claims = df[df['fraud_reported'] == 'Y']
    no_fraud_claims = df[df['fraud_reported'] == 'N']
    
    # Age Analysis
    print("\nAge Analysis:")
    print("-" * 60)
    print(f"Average Age - Fraud: {fraud_claims['age'].mean():.1f} years")
    print(f"Average Age - No Fraud: {no_fraud_claims['age'].mean():.1f} years")
    print(f"Age Range - Fraud: {fraud_claims['age'].min()} to {fraud_claims['age'].max()}")
    print(f"Age Range - No Fraud: {no_fraud_claims['age'].min()} to {no_fraud_claims['age'].max()}")
    
    # Customer Tenure
    print("\nCustomer Tenure Analysis:")
    print("-" * 60)
    print(f"Average Tenure - Fraud: {fraud_claims['months_as_customer'].mean():.1f} months")
    print(f"Average Tenure - No Fraud: {no_fraud_claims['months_as_customer'].mean():.1f} months")
    
    # Gender
    print("\nFraud Rate by Gender:")
    print("-" * 60)
    gender_fraud = df.groupby('insured_sex')['fraud_reported'].apply(lambda x: (x=='Y').sum() / len(x) * 100)
    for gender, rate in gender_fraud.items():
        count = df[df['insured_sex']==gender].shape[0]
        fraud_count = df[(df['insured_sex']==gender) & (df['fraud_reported']=='Y')].shape[0]
        print(f"{gender}: {rate:.2f}% ({fraud_count}/{count})")
    
    # Education Level
    print("\nFraud Rate by Education Level:")
    print("-" * 60)
    edu_fraud = df.groupby('insured_education_level')['fraud_reported'].apply(lambda x: (x=='Y').sum() / len(x) * 100)
    edu_fraud = edu_fraud.sort_values(ascending=False)
    for edu, rate in edu_fraud.items():
        count = df[df['insured_education_level']==edu].shape[0]
        fraud_count = df[(df['insured_education_level']==edu) & (df['fraud_reported']=='Y')].shape[0]
        print(f"{edu}: {rate:.2f}% ({fraud_count}/{count})")


def incident_characteristics_analysis(df):
    """
    Analyze incident characteristics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("INCIDENT CHARACTERISTICS")
    print("="*80)
    
    fraud_claims = df[df['fraud_reported'] == 'Y']
    no_fraud_claims = df[df['fraud_reported'] == 'N']
    
    # Vehicles Involved
    print("\nNumber of Vehicles Involved:")
    print("-" * 60)
    print(f"Fraud: {fraud_claims['number_of_vehicles_involved'].mean():.2f}")
    print(f"No Fraud: {no_fraud_claims['number_of_vehicles_involved'].mean():.2f}")
    
    # Witnesses
    print("\nNumber of Witnesses:")
    print("-" * 60)
    print(f"Fraud: {fraud_claims['witnesses'].mean():.2f}")
    print(f"No Fraud: {no_fraud_claims['witnesses'].mean():.2f}")
    
    # Bodily Injuries
    print("\nNumber of Bodily Injuries:")
    print("-" * 60)
    print(f"Fraud: {fraud_claims['bodily_injuries'].mean():.2f}")
    print(f"No Fraud: {no_fraud_claims['bodily_injuries'].mean():.2f}")
    
    # Hour of Day
    print("\nIncidents by Hour of Day (Top 5 for Fraud):")
    print("-" * 60)
    fraud_hours = fraud_claims['incident_hour_of_the_day'].value_counts().head(5)
    for hour, count in fraud_hours.items():
        print(f"Hour {hour:02d}:00 - {count} incidents")


def geographic_analysis(df):
    """
    Analyze geographic patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("GEOGRAPHIC ANALYSIS")
    print("="*80)
    
    # By Incident State
    print("\nFraud Rate by Incident State:")
    print("-" * 60)
    state_fraud = df.groupby('incident_state').agg({
        'fraud_reported': lambda x: (x=='Y').sum(),
        'policy_number': 'count'
    })
    state_fraud['fraud_rate'] = (state_fraud['fraud_reported'] / state_fraud['policy_number']) * 100
    state_fraud = state_fraud.sort_values('fraud_rate', ascending=False)
    
    for state, row in state_fraud.iterrows():
        print(f"{state}: {row['fraud_rate']:.1f}% ({int(row['fraud_reported'])}/{int(row['policy_number'])})")
    
    # By Policy State
    print("\nPolicy Distribution by State:")
    print("-" * 60)
    policy_states = df['policy_state'].value_counts()
    for state, count in policy_states.items():
        pct = count / len(df) * 100
        print(f"{state}: {count} policies ({pct:.1f}%)")


def vehicle_analysis(df):
    """
    Analyze vehicle-related patterns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("VEHICLE ANALYSIS")
    print("="*80)
    
    fraud_claims = df[df['fraud_reported'] == 'Y']
    
    # Top Auto Makes in Fraud Cases
    print("\nTop 10 Auto Makes in Fraud Cases:")
    print("-" * 60)
    top_fraud_makes = fraud_claims['auto_make'].value_counts().head(10)
    for make, count in top_fraud_makes.items():
        total = df[df['auto_make']==make].shape[0]
        rate = (count / total) * 100
        print(f"{make}: {count} fraud cases out of {total} total ({rate:.1f}%)")
    
    # Auto Year Analysis
    print("\nAuto Year Analysis:")
    print("-" * 60)
    print(f"Average Year - Fraud: {fraud_claims['auto_year'].mean():.0f}")
    print(f"Average Year - No Fraud: {df[df['fraud_reported']=='N']['auto_year'].mean():.0f}")


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def create_main_visualizations(df, output_path='eda_visualization.png'):
    """
    Create comprehensive visualization dashboard
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    output_path : str
        Path to save the visualization
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle('Insurance Fraud Claims - Exploratory Data Analysis', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # 1. Fraud Distribution
    ax = axes[0, 0]
    fraud_counts = df['fraud_reported'].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    ax.bar(fraud_counts.index, fraud_counts.values, color=colors, 
           edgecolor='black', linewidth=1.2)
    ax.set_title('Fraud Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fraud Reported')
    ax.set_ylabel('Count')
    for i, v in enumerate(fraud_counts.values):
        ax.text(i, v + 10, str(v), ha='center', fontweight='bold', fontsize=11)
    
    # 2. Age Distribution by Fraud
    ax = axes[0, 1]
    df_fraud = df[df['fraud_reported'] == 'Y']['age']
    df_no_fraud = df[df['fraud_reported'] == 'N']['age']
    ax.hist([df_no_fraud, df_fraud], bins=20, label=['No Fraud', 'Fraud'], 
            color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
    ax.set_title('Age Distribution by Fraud Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Age')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # 3. Total Claim Amount Distribution
    ax = axes[0, 2]
    ax.hist(df['total_claim_amount'], bins=30, color='#3498db', 
            edgecolor='black', alpha=0.7)
    ax.set_title('Total Claim Amount Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Claim Amount ($)')
    ax.set_ylabel('Frequency')
    ax.axvline(df['total_claim_amount'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: ${df["total_claim_amount"].mean():,.0f}')
    ax.legend()
    
    # 4. Fraud Rate by Incident Type
    ax = axes[1, 0]
    fraud_by_incident = df.groupby('incident_type')['fraud_reported'].apply(
        lambda x: (x=='Y').sum() / len(x) * 100
    )
    fraud_by_incident.sort_values(ascending=True).plot(kind='barh', ax=ax, 
                                                        color='#e67e22', edgecolor='black')
    ax.set_title('Fraud Rate by Incident Type', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fraud Rate (%)')
    ax.set_ylabel('')
    
    # 5. Incident Severity Distribution
    ax = axes[1, 1]
    severity_counts = df['incident_severity'].value_counts()
    ax.bar(range(len(severity_counts)), severity_counts.values, 
           color='#9b59b6', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(severity_counts)))
    ax.set_xticklabels(severity_counts.index, rotation=45, ha='right')
    ax.set_title('Incident Severity Distribution', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    
    # 6. Fraud by Collision Type
    ax = axes[1, 2]
    collision_fraud = pd.crosstab(df['collision_type'], df['fraud_reported'], 
                                   normalize='index') * 100
    collision_fraud.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], 
                         edgecolor='black')
    ax.set_title('Fraud Rate by Collision Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('')
    ax.legend(title='Fraud', labels=['No', 'Yes'])
    ax.tick_params(axis='x', rotation=45)
    
    # 7. Monthly Customer Tenure
    ax = axes[2, 0]
    ax.hist(df['months_as_customer'], bins=30, color='#1abc9c', 
            edgecolor='black', alpha=0.7)
    ax.set_title('Customer Tenure Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Months as Customer')
    ax.set_ylabel('Frequency')
    ax.axvline(df['months_as_customer'].mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {df["months_as_customer"].mean():.0f} months')
    ax.legend()
    
    # 8. Number of Vehicles Involved
    ax = axes[2, 1]
    vehicles_counts = df['number_of_vehicles_involved'].value_counts().sort_index()
    ax.bar(vehicles_counts.index, vehicles_counts.values, 
           color='#f39c12', edgecolor='black', alpha=0.8)
    ax.set_title('Number of Vehicles Involved', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel('Count')
    
    # 9. Fraud by Education Level
    ax = axes[2, 2]
    edu_fraud = df.groupby('insured_education_level')['fraud_reported'].apply(
        lambda x: (x=='Y').sum() / len(x) * 100
    )
    edu_fraud.sort_values(ascending=True).plot(kind='barh', ax=ax, 
                                                color='#c0392b', edgecolor='black')
    ax.set_title('Fraud Rate by Education Level', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fraud Rate (%)')
    ax.set_ylabel('')
    
    # 10. Claims by Hour of Day
    ax = axes[3, 0]
    hour_counts = df['incident_hour_of_the_day'].value_counts().sort_index()
    ax.plot(hour_counts.index, hour_counts.values, marker='o', 
            color='#2980b9', linewidth=2, markersize=6)
    ax.fill_between(hour_counts.index, hour_counts.values, alpha=0.3, color='#2980b9')
    ax.set_title('Incidents by Hour of Day', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Incidents')
    ax.grid(True, alpha=0.3)
    
    # 11. Policy Annual Premium Distribution
    ax = axes[3, 1]
    fraud_premium = df[df['fraud_reported'] == 'Y']['policy_annual_premium']
    no_fraud_premium = df[df['fraud_reported'] == 'N']['policy_annual_premium']
    bp = ax.boxplot([no_fraud_premium, fraud_premium], 
                     tick_labels=['No Fraud', 'Fraud'], 
                     patch_artist=True,
                     boxprops=dict(facecolor='#3498db', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2))
    ax.set_title('Policy Premium by Fraud Status', fontsize=14, fontweight='bold')
    ax.set_ylabel('Annual Premium ($)')
    
    # 12. Property Damage Distribution
    ax = axes[3, 2]
    prop_damage_counts = df['property_damage'].value_counts()
    colors_pie = ['#2ecc71', '#e74c3c', '#95a5a6']
    ax.pie(prop_damage_counts.values, labels=prop_damage_counts.index, 
           autopct='%1.1f%%', colors=colors_pie, startangle=90, 
           textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax.set_title('Property Damage Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def create_advanced_visualizations(df, output_path='advanced_analysis.png'):
    """
    Create advanced analysis visualizations including correlations
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    output_path : str
        Path to save the visualization
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Advanced Analysis - Insurance Fraud Claims', 
                 fontsize=18, fontweight='bold')
    
    # 1. Correlation Heatmap
    ax = axes[0, 0]
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove identifiers
    numerical_cols = [col for col in numerical_cols 
                     if col not in ['policy_number', 'insured_zip', '_c39']]
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, 
                ax=ax, annot_kws={'size': 7})
    ax.set_title('Correlation Matrix - Numerical Features', 
                 fontsize=14, fontweight='bold', pad=10)
    
    # 2. Claim Amount by Fraud Status
    ax = axes[0, 1]
    claim_types = ['total_claim_amount', 'injury_claim', 
                   'property_claim', 'vehicle_claim']
    fraud_means = df[df['fraud_reported'] == 'Y'][claim_types].mean()
    no_fraud_means = df[df['fraud_reported'] == 'N'][claim_types].mean()
    
    x = np.arange(len(claim_types))
    width = 0.35
    bars1 = ax.bar(x - width/2, no_fraud_means, width, label='No Fraud', 
                   color='#2ecc71', edgecolor='black', alpha=0.8)
    bars2 = ax.bar(x + width/2, fraud_means, width, label='Fraud', 
                   color='#e74c3c', edgecolor='black', alpha=0.8)
    
    ax.set_title('Average Claim Amounts by Type', fontsize=14, fontweight='bold')
    ax.set_ylabel('Amount ($)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Total', 'Injury', 'Property', 'Vehicle'], 
                       rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Fraud by State
    ax = axes[1, 0]
    state_fraud = df.groupby('incident_state').agg({
        'fraud_reported': lambda x: (x=='Y').sum(),
        'policy_number': 'count'
    })
    state_fraud['fraud_rate'] = (state_fraud['fraud_reported'] / 
                                  state_fraud['policy_number']) * 100
    state_fraud = state_fraud.sort_values('fraud_rate', ascending=False)
    
    bars = ax.barh(state_fraud.index, state_fraud['fraud_rate'], 
                   color='#e67e22', edgecolor='black', alpha=0.8)
    ax.set_title('Fraud Rate by Incident State', fontsize=14, fontweight='bold')
    ax.set_xlabel('Fraud Rate (%)')
    ax.set_ylabel('State')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', ha='left', va='center', 
                fontweight='bold', fontsize=10)
    
    # 4. Authorities Contacted by Fraud
    ax = axes[1, 1]
    authority_data = pd.crosstab(df['authorities_contacted'].fillna('None'), 
                                  df['fraud_reported'])
    authority_data.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], 
                        edgecolor='black', alpha=0.8)
    ax.set_title('Authorities Contacted by Fraud Status', 
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.set_xlabel('Authority')
    ax.legend(title='Fraud', labels=['No', 'Yes'])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Advanced visualization saved to: {output_path}")
    plt.close()


def create_correlation_analysis(df):
    """
    Perform detailed correlation analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols 
                     if col not in ['policy_number', 'insured_zip', '_c39']]
    
    # Create correlation matrix
    corr_matrix = df[numerical_cols].corr()
    
    # Find high correlations (excluding diagonal)
    print("\nHigh Correlations (|r| > 0.7):")
    print("-" * 60)
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr:
        high_corr_df = pd.DataFrame(high_corr)
        high_corr_df = high_corr_df.sort_values('Correlation', 
                                                 key=abs, ascending=False)
        print(high_corr_df.to_string(index=False))
    else:
        print("No high correlations found.")


# ============================================================================
# SECTION 5: MAIN EXECUTION FUNCTION
# ============================================================================

def main(filepath):
    """
    Main function to run complete EDA
    
    Parameters:
    -----------
    filepath : str
        Path to the insurance fraud CSV file
    """
    print("\n" + "="*80)
    print("INSURANCE FRAUD CLAIMS - EXPLORATORY DATA ANALYSIS")
    print("="*80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Load and explore data
    df = load_and_explore_data(filepath)
    
    # Data quality checks
    check_missing_values(df)
    basic_statistics(df)
    
    # Target variable analysis
    analyze_target_variable(df)
    
    # Categorical features
    analyze_categorical_features(df)
    
    # Fraud analysis
    fraud_analysis_by_dimensions(df)
    claim_amount_analysis(df)
    demographic_analysis(df)
    incident_characteristics_analysis(df)
    geographic_analysis(df)
    vehicle_analysis(df)
    
    # Correlation analysis
    create_correlation_analysis(df)
    
    # Create visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    create_main_visualizations(df, 'eda_visualization.png')
    create_advanced_visualizations(df, 'advanced_analysis.png')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nFiles generated:")
    print("  - eda_visualization.png")
    print("  - advanced_analysis.png")
    print("\nKey Findings:")
    print("  - Total records analyzed: {:,}".format(len(df)))
    print("  - Fraud rate: {:.2f}%".format((df['fraud_reported']=='Y').sum()/len(df)*100))
    print("  - Average fraud claim: ${:,.2f}".format(
        df[df['fraud_reported']=='Y']['total_claim_amount'].mean()))
    print("  - Average legitimate claim: ${:,.2f}".format(
        df[df['fraud_reported']=='N']['total_claim_amount'].mean()))
    

# ============================================================================
# SECTION 6: ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def export_summary_statistics(df, output_file='summary_statistics.csv'):
    """
    Export summary statistics to CSV
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    output_file : str
        Output filename
    """
    summary = df.describe(include='all').T
    summary.to_csv(output_file)
    print(f"Summary statistics exported to: {output_file}")


def detect_outliers(df, column, method='iqr'):
    """
    Detect outliers in a numerical column
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    column : str
        Column name
    method : str
        Method to use ('iqr' or 'zscore')
        
    Returns:
    --------
    outliers : pandas.Series
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column].dropna()))
        outliers = z_scores > 3
    
    return outliers


def analyze_outliers(df):
    """
    Analyze outliers in numerical columns
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    """
    print("\n" + "="*80)
    print("OUTLIER ANALYSIS")
    print("="*80)
    
    numerical_cols = ['total_claim_amount', 'injury_claim', 
                     'property_claim', 'vehicle_claim', 
                     'policy_annual_premium', 'age', 'months_as_customer']
    
    for col in numerical_cols:
        outliers = detect_outliers(df, col, method='iqr')
        n_outliers = outliers.sum()
        pct_outliers = (n_outliers / len(df)) * 100
        
        print(f"\n{col}:")
        print(f"  Number of outliers: {n_outliers} ({pct_outliers:.2f}%)")
        if n_outliers > 0:
            print(f"  Outlier range: ${df[outliers][col].min():,.2f} to ${df[outliers][col].max():,.2f}")


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Update this path to your CSV file location
    FILE_PATH = 'insurance_fraud_claims.csv'
    
    # Run the complete analysis
    main(FILE_PATH)
    
    # Optional: Additional analyses
    # df = pd.read_csv(FILE_PATH)
    # analyze_outliers(df)
    # export_summary_statistics(df)
    
    print("\n" + "="*80)
    print("All analyses completed successfully!")
    print("="*80)
