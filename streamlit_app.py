## Step 00 - Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, classification_report

st.set_page_config(
    page_title="Telco Customer Churn Analysis 📡",
    layout="centered",
    page_icon="📡",
)

## Step 01 - Setup
st.sidebar.title("Telco Churn Analysis 📡")
page = st.sidebar.selectbox("Select Page", ["Business Case 📘", "Visualization 📊", "Prediction 🤖", "Insights and Recommendations 🧠"])

st.write("   ")

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['SeniorCitizen'] = df['SeniorCitizen'].astype(int)

## Step 02 - Pages
if page == "Business Case 📘":

    st.subheader("Telco Customer Churn Dashboard")

    st.markdown("""
    ## 🎯 Business Problem

    Customer churn in the telecom industry leads to:

    - **Revenue loss** from departing customers
    - **High acquisition costs** to replace churned customers
    - **Reduced customer lifetime value**
    - **Competitive disadvantage** in a saturated market

    Understanding churn drivers allows proactive retention strategies.
    """)

    st.markdown("""
    ## Our Solution

    1. **Data Analysis:** Identify key factors driving customer churn
    2. **Visualization:** Interactive dashboards to explore churn patterns
    3. **Predictive Modeling:** Logistic regression to predict churn probability
    """)

    st.markdown("##### Data Preview")
    rows = st.slider("Select number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### Missing Values")
    missing_values = df.isnull().sum()
    st.write(missing_values)

    st.markdown("##### Data Shape")
    st.write("Telco Churn Data:", df.shape)

    if missing_values.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning(f"⚠️ {missing_values.sum()} missing values found")

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())


elif page == "Visualization 📊":

    st.subheader("02 Data Visualization")

    tab1, tab2, tab3, tab4 = st.tabs(["Churn Distribution 📊", "Churn by Contract 📋", "Correlation Matrix 🔥", "Churn by Service 📡"])

    with tab1:
        st.subheader("Customer Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 14})
        ax1.set_title("Churn vs No Churn", fontsize=16)
        st.pyplot(fig1)

        col1, col2 = st.columns(2)
        col1.metric("Total Customers", f"{len(df):,}")
        col2.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")

    with tab2:
        st.subheader("Churn Rate by Contract Type")
        churn_by_contract = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        churn_by_contract.columns = ['Contract', 'Churn Rate (%)']
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=churn_by_contract, x='Contract', y='Churn Rate (%)', ax=ax2, palette='Reds_r')
        ax2.set_title("Churn Rate by Contract Type", fontsize=16)
        ax2.set_ylabel("Churn Rate (%)")
        ax2.set_xlabel("Contract Type")
        st.pyplot(fig2)

    with tab3:
        st.subheader("Correlation Matrix")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='Blues', fmt=".2f", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

    with tab4:
        st.subheader("Churn by Internet Service")
        churn_by_internet = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
        churn_by_internet.columns = ['InternetService', 'Churn Rate (%)']
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.barplot(data=churn_by_internet, x='InternetService', y='Churn Rate (%)', ax=ax4, palette='coolwarm')
        ax4.set_title("Churn Rate by Internet Service", fontsize=16)
        ax4.set_ylabel("Churn Rate (%)")
        ax4.set_xlabel("Internet Service")
        st.pyplot(fig4)


elif page == "Prediction 🤖":
    st.subheader("Churn Prediction — Logistic Regression")

    df2 = df.drop(columns=['customerID']).dropna().copy()

    # Encode categorical variables
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                        'PaperlessBilling', 'PaymentMethod']

    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df2[col] = le.fit_transform(df2[col])
        le_dict[col] = le

    le_churn = LabelEncoder()
    df2['Churn'] = le_churn.fit_transform(df2['Churn'])

    feature_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges'] + categorical_cols
    X = df2[feature_cols]
    y = df2['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    st.markdown("### Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
    col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")

    st.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob):.3f}")

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    labels = le_churn.classes_
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=labels, yticklabels=labels)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Churn Prediction Confusion Matrix')
    st.pyplot(fig1)

    # Feature Importance
    st.markdown("### Feature Importance")
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', key=abs, ascending=False)

    fig2, ax2 = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in importance['Coefficient']]
    sns.barplot(data=importance, x='Coefficient', y='Feature', palette=colors, ax=ax2)
    ax2.set_title("Feature Importance (Logistic Regression Coefficients)", fontsize=16)
    ax2.set_xlabel("Coefficient Value")
    ax2.set_ylabel("")
    st.pyplot(fig2)

    # Classification Report
    st.markdown("### Classification Report")
    report = classification_report(y_test, y_pred, target_names=le_churn.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())


elif page == "Insights and Recommendations 🧠":
    st.subheader("Insights and Recommendations")

    col1, col2, col3 = st.columns(3)
    col1.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.1f}%")
    col2.metric("Avg Tenure", f"{df['tenure'].mean():.0f} months")
    col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")

    st.markdown("""
    ## Key Insights

    1. **Contract Type Drives Churn:** Month-to-month customers churn at significantly higher rates than those on 1-year or 2-year contracts.
    2. **Tenure Matters:** Customers with shorter tenure are far more likely to churn — the first 12 months are critical.
    3. **Fiber Optic Risk:** Fiber optic internet customers show higher churn rates, suggesting possible service quality or pricing issues.
    4. **Payment Method:** Electronic check users have notably higher churn compared to automatic payment methods.
    5. **Senior Citizens:** Higher churn rate compared to non-senior customers.

    ## Recommendations

    1. **Incentivize Long-Term Contracts:** Offer discounts or perks for customers switching from month-to-month to annual contracts.
    2. **Early Retention Programs:** Focus retention efforts on new customers within their first 12 months.
    3. **Investigate Fiber Optic Service:** Review fiber optic pricing, speed, and reliability to address higher churn.
    4. **Promote Auto-Pay:** Encourage customers to switch from electronic check to automatic bank transfer or credit card payments.
    5. **Senior-Specific Plans:** Create tailored plans and dedicated support for senior citizen customers.
    """)

    st.info("🎯 Targeting the right segments with proactive retention can reduce churn by 15-25%.")
