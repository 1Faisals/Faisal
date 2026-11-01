import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise HR Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Corporate CSS (Exact same layout + Complete Fix for Black Input Boxes)
st.markdown("""
<style>
    /* Force clean corporate light background and typography */
    .stApp {
        background-color: #F4F6F9;
        font-family: 'Segoe UI', Roboto, Helvetica, sans-serif;
    }
    
    /* Fix labels and text */
    label, .stMarkdown, p, h1, h2, h3, h4, span {
        color: #1E293B !important;
    }

    /* COMPLETE FIX FOR BLACK INPUT BOXES, STEPPERS (+/-) & DROPDOWNS */
    div[data-baseweb="input"], 
    div[data-baseweb="input"] > div, 
    div[data-baseweb="base-input"],
    div[data-baseweb="select"],
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        border-color: #CBD5E1 !important;
        border-radius: 6px !important;
    }
    
    /* Fix text inside number inputs and select boxes */
    input[type="number"], input[type="text"], div[data-baseweb="select"] span {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        -webkit-text-fill-color: #0F172A !important;
    }

    /* Fix the +/- buttons on number inputs */
    button[kind="header"], div[data-baseweb="input"] button {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        border-left: 1px solid #CBD5E1 !important;
    }
    div[data-baseweb="input"] button svg {
        fill: #0F172A !important;
    }

    /* Fix the dropdown menu list when clicked */
    div[data-baseweb="popover"] div, ul[role="listbox"], li[role="option"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    li[role="option"]:hover {
        background-color: #EFF6FF !important;
    }

    /* Professional Header Banner */
    .corporate-header {
        background-color: #0F172A;
        padding: 24px 32px;
        border-radius: 8px;
        margin-bottom: 24px;
        border-left: 6px solid #2563EB;
    }
    .corporate-header h1 {
        color: #FFFFFF !important;
        font-size: 26px;
        font-weight: 600;
        margin: 0;
    }
    .corporate-header p {
        color: #94A3B8 !important;
        font-size: 14px;
        margin: 6px 0 0 0;
    }

    /* KPI Metric Cards */
    .kpi-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .kpi-title {
        color: #64748B !important;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    .kpi-value {
        color: #0F172A !important;
        font-size: 28px;
        font-weight: 700;
        margin-top: 6px;
    }

    /* Professional Submit Button */
    .stButton > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        transition: background-color 0.2s;
    }
    .stButton > button:hover {
        background-color: #1D4ED8 !important;
    }
    .stButton > button p {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. Header Section
st.markdown("""
<div class="corporate-header">
    <h1>Workforce Attrition & Retention Intelligence</h1>
    <p>Predictive Machine Learning Decision Support System for Human Resources</p>
</div>
""", unsafe_allow_html=True)

# 4. Load Dataset & Train Model
@st.cache_data
def load_and_train():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    
    features = [
        'Age', 'MonthlyIncome', 'DistanceFromHome', 
        'YearsAtCompany', 'YearsSinceLastPromotion', 
        'JobSatisfaction', 'WorkLifeBalance'
    ]
    
    X = df[features].copy()
    X['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
    features.append('OverTime')
    
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X, y)
    
    return df, model, features

df, model, features = load_and_train()

# 5. Top Row: Company KPIs
total_emp = len(df)
attrition_rate = (df['Attrition'].value_counts().get('Yes', 0) / total_emp) * 100
avg_income = df['MonthlyIncome'].mean()
avg_tenure = df['YearsAtCompany'].mean()

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Total Workforce</div><div class="kpi-value">{total_emp:,}</div></div>', unsafe_allow_html=True)
with k2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Historical Attrition Rate</div><div class="kpi-value">{attrition_rate:.1f}%</div></div>', unsafe_allow_html=True)
with k3:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Avg Monthly Compensation</div><div class="kpi-value">${avg_income:,.0f}</div></div>', unsafe_allow_html=True)
with k4:
    st.markdown(f'<div class="kpi-card"><div class="kpi-title">Average Tenure</div><div class="kpi-value">{avg_tenure:.1f} Years</div></div>', unsafe_allow_html=True)

st.write("")
st.write("")

# 6. Main Dashboard Area (Left: Input Form, Right: Analytics Output)
left_col, right_col = st.columns([1.2, 1], gap="large")

with left_col:
    st.subheader("Employee Profile Parameters")
    st.write("Enter employee record details below to evaluate flight risk.")
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Employee Age", min_value=18, max_value=65, value=32)
        income = st.number_input("Monthly Income (USD)", min_value=1000, max_value=25000, value=4500, step=250)
        distance = st.number_input("Commute Distance (km)", min_value=1, max_value=60, value=8)
        overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
        
    with c2:
        years_company = st.number_input("Tenure at Company (Years)", min_value=0, max_value=40, value=4)
        years_promo = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=1)
        job_sat = st.selectbox("Job Satisfaction Level", [1, 2, 3, 4], index=2, format_func=lambda x: f"Level {x} - " + ["Low", "Medium", "High", "Very High"][x-1])
        wlb = st.selectbox("Work-Life Balance Rating", [1, 2, 3, 4], index=2, format_func=lambda x: f"Rating {x} - " + ["Poor", "Fair", "Good", "Excellent"][x-1])

    st.write("")
    analyze_btn = st.button("Run Attrition Risk Assessment")

with right_col:
    st.subheader("Diagnostic Assessment")
    
    if analyze_btn:
        ot_val = 1 if overtime == "Yes" else 0
        input_df = pd.DataFrame([[
            age, income, distance, years_company, years_promo, job_sat, wlb, ot_val
        ]], columns=features)
        
        prob = model.predict_proba(input_df)[0][1] * 100
        
        if prob >= 50:
            risk_label = "HIGH ATTRITION RISK"
            bg_color = "#FEF2F2"
            border_color = "#DC2626"
            text_color = "#991B1B"
            rec = "Immediate HR intervention recommended. Review compensation alignment, workload distribution, and promotion timeline."
        elif prob >= 25:
            risk_label = "MODERATE ATTRITION RISK"
            bg_color = "#FFFBEB"
            border_color = "#D97706"
            text_color = "#92400E"
            rec = "Monitor employee engagement during next quarterly review. Discuss career progression opportunities."
        else:
            risk_label = "LOW ATTRITION RISK (STABLE)"
            bg_color = "#F0FDF4"
            border_color = "#16A34A"
            text_color = "#166534"
            rec = "Employee metrics indicate high stability. Maintain standard engagement and recognition practices."

        st.markdown(f"""
        <div style="background-color: {bg_color}; border-left: 5px solid {border_color}; padding: 20px; border-radius: 6px; margin-bottom: 20px;">
            <div style="color: {text_color}; font-weight: 700; font-size: 13px; letter-spacing: 0.8px;">{risk_label}</div>
            <div style="color: #0F172A; font-size: 34px; font-weight: 800; margin: 6px 0;">{prob:.1f}% Probability</div>
            <div style="color: #334155; font-size: 14px; line-height: 1.5;"><strong>HR Action Plan:</strong> {rec}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("**Key Model Decision Drivers (Global Weightage)**")
        imp_df = pd.DataFrame({
            'Factor': ['Monthly Income', 'Age', 'Commute Distance', 'Company Tenure', 'Overtime', 'Years Since Promo', 'Job Satisfaction', 'Work-Life Balance'],
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=True)
        
        st.bar_chart(imp_df.set_index('Factor'), height=220)
    else:
        st.info("Configure employee parameters on the left panel and click 'Run Attrition Risk Assessment' to generate the diagnostic report.")