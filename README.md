# 🏦 CreditWise Loan Prediction

An end-to-end Machine Learning project to predict loan approval status based on applicant demographics, financial histories, and loan properties. Using advanced data preprocessing, feature engineering, and robust classification models, this project identifies creditworthiness and automates the loan evaluation workflow.

---

## 🎯 Project Objectives
- Conduct **Exploratory Data Analysis (EDA)** to understand applicant profiles and key drivers behind loan approvals.
- Build a robust **data preprocessing pipeline** handling missing values, encoding categorical attributes, and scaling features.
- Perform **Feature Engineering** to capture non-linear relationships (polynomial square transformations and log-transforming skewed variables).
- Train, evaluate, and compare multiple classification models including **Logistic Regression**, **K-Nearest Neighbors (KNN)**, and **Naive Bayes**.

---

## 📊 Dataset Overview
The dataset contains **1,000 applicant profiles** with 20 features mapping demographic information, employment histories, credit profiles, and loan characteristics.

### Feature Dictionary

| Attribute | Data Type | Description |
| :--- | :--- | :--- |
| **Applicant_ID** | `float64` | Unique identifier for each applicant |
| **Applicant_Income** | `float64` | Monthly income of the primary applicant |
| **Coapplicant_Income** | `float64` | Monthly income of the co-applicant |
| **Employment_Status** | `Categorical` | Employment type (e.g., *Salaried, Self-employed, Contract, Unemployed*) |
| **Age** | `float64` | Age of the applicant in years |
| **Marital_Status** | `Categorical` | Marital status (*Married, Single*) |
| **Dependents** | `float64` | Number of family dependents |
| **Credit_Score** | `float64` | Credit history score (ranging from 550 to 799) |
| **Existing_Loans** | `float64` | Count of currently active loans |
| **DTI_Ratio** | `float64` | Debt-to-Income Ratio (ranging from 0.10 to 0.60) |
| **Savings** | `float64` | Total liquid savings in accounts |
| **Collateral_Value** | `float64` | Market value of collateral pledged for the loan |
| **Loan_Amount** | `float64` | Requested loan principal amount |
| **Loan_Term** | `float64` | Duration of the loan in months |
| **Loan_Purpose** | `Categorical` | Primary purpose of the loan (*Home, Personal, Education, Car, Business*) |
| **Property_Area** | `Categorical` | Geographic category of property (*Urban, Semiurban, Rural*) |
| **Education_Level** | `Categorical` | Education milestone achieved (*Graduate, Not Graduate*) |
| **Gender** | `Categorical` | Self-identified gender of applicant |
| **Employer_Category** | `Categorical` | Industry of employment (*Government, MNC, Private, Business, Unemployed*) |
| **Loan_Approved** | `Categorical` | **Target Variable:** Status of loan application (*Yes / No*) |

---

## ⚙️ Machine Learning Pipeline

### 1. Data Preprocessing & Cleaning
- **Imputation**: Missing values are imputed using standard statistical practices:
  - *Numerical Features*: Filled with the `mean` strategy using `SimpleImputer`.
  - *Categorical Features*: Filled with the `most_frequent` (mode) strategy.
- **Categorical Encoding**:
  - `Education_Level` and `Loan_Approved` (Target) are converted using `LabelEncoder`.
  - Multiclass categorical columns (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`) are encoded using `OneHotEncoder(drop='first', sparse_output=False)` to prevent multicollinearity.
- **Feature Scaling**: Scaled using `StandardScaler` to bring all numerical variables to a uniform distribution, crucial for distance-based models (KNN) and gradient-descent algorithms (Logistic Regression).

### 2. Feature Engineering
To capture complex patterns, the following feature transformations are performed:
- **DTI_Ratio_sq**: Squared Debt-to-Income ratio to capture compounding debt risk.
- **Credit_Score_sq**: Squared Credit Score to amplify differences at premium credit thresholds.
- **Applicant_Income_log**: Natural logarithm transformation `log(1 + x)` of applicant income to reduce right-skewness and normalize distribution.
- Redundant base features (`Credit_Score` and `DTI_Ratio`) are dropped post-engineering.

---

## 📈 Model Performance & Evaluation

Models are evaluated before and after feature engineering using standard metrics: **Accuracy, Precision, Recall, and F1-Score**.

### 1. Base Model Performance (Without Feature Engineering)

| Model Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **86.5%** | 78.3% | **77.0%** | **77.7%** |
| **Gaussian Naive Bayes** | **86.5%** | **80.4%** | 73.8% | 76.9% |
| **K-Nearest Neighbors (KNN)** | 76.0% | 62.7% | 52.5% | 57.1% |

### 2. Advanced Performance (With Feature Engineering)

| Model Classifier | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** 🚀 | **88.0%** | 78.5% | **83.6%** | **81.0%** |
| **Gaussian Naive Bayes** | 86.0% | **81.1%** | 70.5% | 75.4% |
| **K-Nearest Neighbors (KNN)** | 78.5% | 67.3% | 57.4% | 61.9% |

> [!NOTE]
> **Logistic Regression** combined with **Feature Engineering** yielded the best overall performance, boosting accuracy to **88.0%** and F1-Score to **81.0%**. For risk-averse scenarios prioritizing minimal false approvals, **Naive Bayes** offers the highest **Precision (81.1%)**.

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/vinayakbhat19/creditwise-_loanpredict.git
   cd creditwise-_loanpredict
   ```

2. **Install Dependencies**:
   Ensure you have Python installed, then run:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

3. **Explore the Notebook**:
   Launch the Jupyter notebook interface to interact with the code and visualization pipeline:
   ```bash
   jupyter notebook credit_wise.ipynb
   ```

---

## 📜 License
This project is licensed under the MIT License.
