# 🏦 CreditWise Loan System — Loan Approval Prediction

A supervised machine learning project that predicts whether a loan application should be **Approved** ✅ or **Rejected** ❌, based on an applicant's personal, financial, and credit information.

## 🎯 Problem Statement

**SecureTrust Bank** is a mid-sized financial company offering personal and home loans to customers across urban and rural regions of India. Loan applications have traditionally been reviewed through a manual verification process. This project explores whether a machine learning model can predict loan approval outcomes accurately, quickly, and consistently — supporting (not replacing) that decision process.

> 💡 Given an applicant's personal, financial, and credit details, can we predict whether their loan application will be Approved or Rejected?

This is framed as a **binary classification** problem, with `Loan_Approved` (1 = Approved, 0 = Rejected) as the target variable.

## 📊 Dataset

Each row represents a loan applicant. Key columns include:

| Column | Description |
|---|---|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral offered |
| Gender | Applicant gender |
| Education_Level | Applicant education level |
| Loan_Purpose | Purpose of the loan |
| Property_Area | Urban / Semiurban / Rural |
| Employer_Category | Category of employer |
| Loan_Approved | Target — 1 = Approved, 0 = Rejected |

> ⚠️ Note: the source CSV (`loan_approval_data.csv`) is not included in this repository. Place it in the project root before running the notebook.

## 🔄 Project Workflow

1. 🧹 **Data Cleaning & Preprocessing** — missing numerical values imputed with the column mean; missing categorical values imputed with the mode.
2. 🔍 **Exploratory Data Analysis (EDA)** — target balance, demographic distributions (Gender, Education), income distributions, and comparisons of Income / Credit Score / DTI Ratio / Savings across approval outcomes.
3. 🛠️ **Feature Engineering** — dropped the non-predictive `Applicant_ID` column; encoded categorical features using Label Encoding (`Education_Level`, target) and One-Hot Encoding (`Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`).
4. 🔥 **Correlation Analysis** — heatmap of numerical features to check multicollinearity and relationships with the target.
5. ✂️ **Train/Test Split & Scaling** — 80/20 split (`random_state=42`), features standardized with `StandardScaler`.
6. 🤖 **Model Training & Evaluation** — three baseline classifiers trained and compared:
   - Logistic Regression
   - K-Nearest Neighbors (k = 5)
   - Gaussian Naive Bayes
7. ⚙️ **Iteration 2 — Feature Engineering** — added `DTI_Ratio_sq` and `Credit_Score_sq` (squared terms) to capture non-linear relationships, replacing the original `Credit_Score` and `DTI_Ratio` columns, then retrained all three models.
8. 📈 **Before vs After Comparison** — Precision, Recall, and F1-Score compared across both iterations for all models.

## 🏆 Results

| Model | Precision (Before → After) | Recall (Before → After) | F1-Score (Before → After) |
|---|---|---|---|
| Logistic Regression | 0.783 → 0.790 | 0.770 → 0.803 | 0.777 → 0.797 |
| KNN | 0.627 → 0.620 | 0.525 → 0.508 | 0.571 → 0.559 |
| Naive Bayes | 0.804 → 0.783 | 0.738 → 0.770 | 0.769 → 0.777 |

**Key takeaways:**
- 🚀 **Logistic Regression** improved the most after feature engineering (Recall 0.770 → 0.803, F1 0.777 → 0.797), making it the best overall model post feature engineering.
- 🥈 **Naive Bayes** was the strongest baseline model, though its Precision dropped slightly after feature engineering.
- 🐢 **KNN** performed worst across both iterations and is not recommended for deployment.

Precision and Recall were prioritized alongside Accuracy since both error types carry real business cost — a false negative means a good applicant is wrongly rejected, while a false positive means a risky applicant is wrongly approved.

## 🧰 Tech Stack

- 🐍 Python 3
- 🐼 pandas, numpy
- 🔬 scikit-learn (`SimpleImputer`, `LabelEncoder`, `OneHotEncoder`, `StandardScaler`, `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`, metrics)
- 📉 matplotlib, seaborn (visualization)

## 🚀 Getting Started

1. Clone/download this repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Place `loan_approval_data.csv` in the project root.
4. Launch the notebook:
   ```bash
   jupyter notebook Loan_Prediction.ipynb
   ```
5. Run all cells in order. ✅

## 📁 Repository Contents

- `Loan_Prediction.ipynb` — full analysis notebook (data cleaning, EDA, feature engineering, model training and evaluation).
- `README.md` — this file.
- `LICENSE` — project license.

## 👤 Author

**Aniket Patel**

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
