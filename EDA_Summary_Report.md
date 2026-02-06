# Insurance Fraud Claims - Exploratory Data Analysis Report

## Executive Summary

This report presents a comprehensive exploratory data analysis of an insurance fraud claims dataset containing **1,000 insurance claims** with **40 features**. The analysis reveals key patterns and characteristics associated with fraudulent insurance claims.

---

## Dataset Overview

### Basic Statistics
- **Total Records**: 1,000 claims
- **Total Features**: 40 columns
- **Fraud Rate**: 24.7% (247 fraudulent claims, 753 legitimate claims)
- **Time Period**: Claims from January-February 2015
- **Geographic Coverage**: 7 states (NY, SC, WV, VA, NC, PA, OH)

### Data Quality
- **Missing Values**: 
  - `_c39`: 100% missing (empty column, can be removed)
  - `authorities_contacted`: 9.1% missing (91 records)
- **No duplicates** identified
- **Date formats** consistent across policy and incident dates

---

## Key Findings

### 1. Fraud Patterns by Incident Type

**Highest Fraud Rates**:
- **Single Vehicle Collision**: 29.0% fraud rate (117/403 claims)
- **Multi-vehicle Collision**: 27.2% fraud rate (114/419 claims)

**Lowest Fraud Rates**:
- **Parked Car**: 9.5% fraud rate (8/84 claims)
- **Vehicle Theft**: 8.5% fraud rate (8/94 claims)

**Insight**: Collision-type incidents show significantly higher fraud rates compared to theft or parked car incidents, suggesting fraudsters may prefer scenarios with more ambiguity and potential for fabrication.

---

### 2. Incident Severity - Strong Fraud Indicator

**Critical Finding**: Incident severity shows the **strongest correlation with fraud**:

- **Major Damage**: 60.5% fraud rate (167/276 claims) ⚠️ **HIGHEST RISK**
- **Minor Damage**: 10.7% fraud rate (38/354 claims)
- **Total Loss**: 12.9% fraud rate (36/280 claims)
- **Trivial Damage**: 6.7% fraud rate (6/90 claims)

**Insight**: Claims reporting "Major Damage" are **5-6 times more likely** to be fraudulent compared to minor or trivial damage claims. This is a critical red flag for fraud detection models.

---

### 3. Financial Impact

**Claim Amounts**:
- **Fraudulent Claims Average**: $60,302
- **Legitimate Claims Average**: $50,289
- **Difference**: $10,014 higher for fraud cases (+20%)

**Breakdown by Claim Type**:
| Claim Type | Fraud Average | No Fraud Average | Difference |
|------------|---------------|------------------|------------|
| Total Claim | $60,302 | $50,289 | +$10,013 |
| Vehicle Claim | $43,534 | $36,090 | +$7,444 |
| Injury Claim | $5,483 | $5,024 | +$459 |
| Property Claim | $11,285 | $9,175 | +$2,110 |

**Insight**: Fraudulent claims consistently show higher amounts across all claim types, with vehicle claims showing the largest absolute difference.

---

### 4. Demographic Patterns

**Age Distribution**:
- Fraud Average: 39.1 years
- No Fraud Average: 38.9 years
- **No significant age difference** between fraud and legitimate claims

**Gender**:
- Male Fraud Rate: 26.1%
- Female Fraud Rate: 23.5%
- **Slightly higher fraud rate for males**, but difference is modest

**Education Level Fraud Rates**:
- JD (Law): ~25% fraud rate
- High School: ~25% fraud rate
- PhD: ~24% fraud rate
- **No clear correlation** between education level and fraud propensity

**Customer Tenure**:
- Fraud: 208.1 months average
- No Fraud: 202.6 months average
- **Longer-tenured customers** show slightly higher fraud rates (possibly due to familiarity with claims process)

---

### 5. Geographic Insights

**Incident States by Fraud Rate**:
1. **PA** (Pennsylvania): Highest fraud rate
2. **OH** (Ohio): Second highest
3. **NY** (New York): 262 total incidents
4. **SC** (South Carolina): 248 total incidents
5. **WV** (West Virginia): 217 total incidents

**Policy States**:
- Ohio (OH): 352 policies (35.2%)
- Illinois (IL): 338 policies (33.8%)
- Indiana (IN): 310 policies (31.0%)

**Insight**: There's a **mismatch between policy state and incident state**, suggesting people traveling or fraudulent activity across state lines.

---

### 6. Incident Characteristics

**Vehicles Involved**:
- Fraud: 1.93 vehicles average
- No Fraud: 1.81 vehicles average
- Fraudulent claims involve **slightly more vehicles**

**Witnesses**:
- Fraud: 1.58 witnesses average
- No Fraud: 1.46 witnesses average
- **No significant difference**

**Bodily Injuries**:
- Fraud: 1.04 injuries average
- No Fraud: 0.98 injuries average
- **Minimal difference**

**Time of Day**:
- Incidents occur throughout all 24 hours
- Notable peaks at **hours 0, 3, and 6** for fraudulent claims
- Late night/early morning incidents warrant additional scrutiny

---

### 7. Property Damage & Police Reports

**Property Damage**:
- "?" (Unknown): 28.6% fraud rate
- "YES": 25.8% fraud rate
- "NO": 19.5% fraud rate

**Police Report Availability**:
- Unknown ("?"): 25.9% fraud rate
- "NO": 25.1% fraud rate
- "YES": 22.9% fraud rate

**Insight**: **Missing or unclear information** on property damage and police reports is associated with **higher fraud rates**. Lack of official documentation creates opportunities for fraud.

---

### 8. Vehicle Makes in Fraud Cases

**Top 10 Vehicle Makes in Fraudulent Claims**:
1. Mercedes: 22 cases
2. Ford: 22 cases
3. Audi: 21 cases
4. Chevrolet: 21 cases
5. Dodge: 20 cases
6. BMW: 20 cases
7. Suburu: 19 cases
8. Volkswagen: 19 cases
9. Saab: 18 cases
10. Nissan: 14 cases

**Insight**: Luxury brands (Mercedes, Audi, BMW) appear frequently in fraud cases, potentially due to higher claim values.

---

### 9. Collision Types

**Distribution**:
- Rear Collision: 292 incidents (29.2%)
- Side Collision: 276 incidents (27.6%)
- Front Collision: 254 incidents (25.4%)
- Unknown ("?"): 178 incidents (17.8%)

**Fraud Rates by Collision Type**:
- All collision types show relatively similar fraud rates (20-30%)
- Unknown collision type shows elevated fraud rate

---

### 10. Policy Characteristics

**Policy Coverage Levels (CSL)**:
- 250/500: 351 policies (35.1%)
- 100/300: 349 policies (34.9%)
- 500/1000: 300 policies (30.0%)
- **Even distribution** across coverage levels

**Policy Deductibles**:
- Range: $500 to $2,000
- Most common: $1,000 and $2,000
- Average annual premium: $1,258

**Umbrella Limits**:
- Range: $0 to $6,000,000
- Many policies with $0 umbrella coverage

---

## Critical Fraud Indicators (Ranked by Importance)

### 🔴 High Risk Indicators:
1. **Incident Severity = "Major Damage"** (60.5% fraud rate)
2. **Total claim amount > $60,000** (significantly above average)
3. **Property damage status = Unknown**
4. **Police report not available or unknown**
5. **Single or Multi-vehicle collisions** (vs. theft/parked car)

### 🟡 Moderate Risk Indicators:
6. **Incident in PA or OH states**
7. **Luxury vehicle makes** (Mercedes, BMW, Audi)
8. **Multiple vehicles involved**
9. **Incident during late night/early morning hours** (0-6 AM)
10. **Longer customer tenure** (>200 months)

---

## Data Quality & Missing Values

### Issues Identified:
1. **`_c39` column**: Completely empty (100% missing) - recommend removal
2. **`authorities_contacted`**: 9.1% missing - may indicate unreported incidents
3. **Collision type "?"**: 17.8% of records - ambiguity in incident classification
4. **Property damage "?"**: 36% of records - documentation gaps

### Recommendations:
- Remove `_c39` column from analysis
- Impute or flag missing `authorities_contacted` values
- Investigate why collision type and property damage have high "unknown" rates
- Consider "unknown" as a separate category rather than missing data

---

## Statistical Correlations

### Numerical Feature Correlations:
- **Strong positive correlations**:
  - Total claim ↔ Vehicle claim (0.90+)
  - Total claim ↔ Injury claim
  - Total claim ↔ Property claim
  
- **Weak correlations**:
  - Age ↔ Fraud (minimal correlation)
  - Customer tenure ↔ Fraud (minimal correlation)

**Insight**: Claim components are highly correlated with total claim (as expected), but demographic features show weak predictive power for fraud detection.

---

## Recommendations for Fraud Detection

### 1. **Immediate Red Flags**:
   - Flag all claims with "Major Damage" severity for manual review
   - Scrutinize claims lacking police reports or property damage documentation
   - Review claims from PA and OH with extra diligence

### 2. **Risk Scoring Model**:
   - Assign highest weight to incident severity
   - Include claim amount thresholds
   - Factor in documentation completeness
   - Consider incident type and collision characteristics

### 3. **Data Collection Improvements**:
   - Mandate police reports for all major incidents
   - Reduce "unknown" categories through better data capture
   - Add timestamp precision for incident hour validation
   - Collect additional vehicle damage photos/documentation

### 4. **Further Analysis Needed**:
   - Time series analysis of fraud patterns over months
   - Text analysis of incident descriptions (if available)
   - Network analysis to detect organized fraud rings
   - Machine learning model development for fraud prediction

---

## Conclusion

This dataset reveals clear patterns distinguishing fraudulent from legitimate insurance claims. The most powerful predictor is **incident severity**, with "Major Damage" claims showing a 60.5% fraud rate compared to 10-13% for other severity levels. Combined with claim amount thresholds, documentation gaps, and geographic patterns, insurers can develop robust fraud detection systems.

The 24.7% fraud rate represents significant financial exposure, with fraudulent claims averaging $10,000 higher than legitimate ones. Implementing targeted detection strategies based on these findings could substantially reduce fraud losses.

---

**Analysis Date**: February 2026  
**Dataset**: insurance_fraud_claims.csv  
**Total Records Analyzed**: 1,000 claims  
**Analysis Tools**: Python (pandas, matplotlib, seaborn)
