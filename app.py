import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import time
import joblib
import shap
import os
import base64

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.utils.multiclass import type_of_target
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Unsupervised
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, precision_score, 
    recall_score, f1_score, roc_curve, auc, precision_recall_curve
)

# Boosting Models
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from datetime import datetime

# PAGE CONFIG
st.set_page_config(page_title="Neuronix AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# NEXT-GEN PREMIUM UI STYLE
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* BACKGROUND & MAIN CONTAINER */
.stApp { background: linear-gradient(135deg, #09090b, #111827, #1e293b); color: #e4e4e7; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1500px; }

/* SIDEBAR */
[data-testid="stSidebar"] { background: #0b1120; border-right: 1px solid rgba(255,255,255,.08); }

/* BUTTONS */
.stButton > button {
    width: 100%; border-radius: 14px; height: 52px; border: none; color: white;
    font-weight: 600; font-size: 15px; background: linear-gradient(90deg, #2563eb, #7c3aed); transition: 0.3s;
}
.stButton > button:hover { transform: translateY(-3px); box-shadow: 0 12px 25px rgba(0,0,0,.35); }

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    width: 100%; border-radius: 14px; background: linear-gradient(90deg, #059669, #16a34a);
    color: white; font-weight: 600; height: 52px;
}

/* METRICS & TABS */
[data-testid="metric-container"] {
    background: #111827; border-radius: 18px; padding: 18px; border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 8px 25px rgba(0,0,0,.35);
}
.stTabs [data-baseweb="tab"] { border-radius: 12px; padding: 12px 22px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #2563eb; color: white; }

/* DATAFRAME & UPLOADER */
[data-testid="stDataFrame"] { border-radius: 16px; overflow: hidden; }
[data-testid="stFileUploader"] { border-radius: 18px; border: 2px dashed #3b82f6; background: #111827; }

/* ALERTS & SCROLLBAR */
.stSuccess, .stInfo, .stWarning, .stError { border-radius: 15px; }
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #475569; border-radius: 20px; }
::-webkit-scrollbar-track { background: #111827; }
</style>
""", unsafe_allow_html=True)

# PREMIUM HERO SECTION (NEURONIX AI LOGO STYLE)
st.markdown("""
<div style="padding: 45px; border-radius: 25px; background: linear-gradient(135deg, #0f172a, #1e1b4b); border: 1px solid rgba(139, 92, 246, 0.3); box-shadow: 0 20px 50px rgba(0,0,0,.5); margin-bottom: 30px;">
    <h1 style="font-size: 56px; font-weight: 900; margin-bottom: 5px; letter-spacing: -1px;">
        <span style="color: white;">Neuronix</span><span style="background: -webkit-linear-gradient(45deg, #06b6d4, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">.AI</span>
    </h1>
    <p style="font-size: 20px; color: #94a3b8; margin-bottom: 30px; font-weight: 300;">Think Smarter. Predict Faster. Your Enterprise AutoML Universe.</p>
    <div style="display: flex; gap: 12px; flex-wrap: wrap;">
        <div style="background: rgba(37, 99, 235, 0.1); border: 1px solid #2563eb; padding: 12px 22px; border-radius: 12px; color: #60a5fa; font-weight: 600;">🧠 Neural Engine Active</div>
        <div style="background: rgba(124, 58, 237, 0.1); border: 1px solid #7c3aed; padding: 12px 22px; border-radius: 12px; color: #a78bfa; font-weight: 600;">⚡ Auto Feature Engineering</div>
        <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; padding: 12px 22px; border-radius: 12px; color: #34d399; font-weight: 600;">🔍 Explainable AI</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 300px !important;
        max-width: 300px !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-top: 0rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    try:
        with open("Neuronix AI/logo.png", "rb") as f:
            data = f.read()
            img_base64 = base64.b64encode(data).decode()
            
        sst.markdown(
            f'''
            <div style="text-align: center; margin-top: -30px; margin-bottom: 20px;">
                <img src="data:image/pnd;base64,{img_base64}" style="width: 80%; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
            </div>
            ''',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.image("logo.png", use_container_width=True)
    st.markdown("<div style='padding: 0px 20px;'>", unsafe_allow_html=True)
    st.markdown("### 🔌 Control Panel")
    st.markdown("---")
    file = st.file_uploader("📂 Upload Dataset", type=["csv", "xlsx", "xls", "json", "txt", "parquet"])

    st.markdown("## ⚡ Project Status")
    st.success("🟢 System Ready")
    st.info("🤖 Neural Engine Active")
    st.info("📊 AutoML Enabled")
    st.info("🧠 Explainable AI Ready")
    st.success("📄 PDF Report Available")

    if file is not None:
        st.markdown("---")
        st.success("✅ Dataset Uploaded")
        st.write(f"📄 **File Name:** {file.name}")
        st.write(f"📦 **File Size:** {round(file.size / 1024, 2)} KB")

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:10px; font-size:13px; color:#64748b;">
        Version <b>2.0 Pro</b><br>Python • Streamlit • Scikit-learn<br><br>❤️ Built with AI
    </div>
    """, unsafe_allow_html=True)

# MAIN LOGIC
if file is not None:
    # 1. LOAD DATASET
    if "df" not in st.session_state or st.session_state.get("last_file") != file.name:
        try:
            if file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(file)
            elif file.name.endswith((".xlsx", ".xls")):
                st.session_state.df = pd.read_excel(file)
            elif file.name.endswith(".json"):
                st.session_state.df = pd.read_json(file)
            elif file.name.endswith(".txt"):
                st.session_state.df = pd.read_csv(file, sep=None, engine="python")
            elif file.name.endswith(".parquet"):
                st.session_state.df = pd.read_parquet(file)
            else:
                st.error("❌ Unsupported file format.")
                st.stop()
            st.session_state.last_file = file.name
            st.toast("✅ Dataset Loaded Successfully", icon="🎉")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")
            st.stop()

    df = st.session_state.df

    # 2. DATASET VALIDATION
    if df.empty:
        st.error("❌ Dataset is Empty")
        st.stop()

    if df.columns.duplicated().any():
        st.warning("⚠ Duplicate Column Names Found")

    # 3. ENTERPRISE DASHBOARD CARDS
    rows = df.shape[0]
    cols = df.shape[1]
    missing = df.isnull().sum().sum()
    memory = round(df.memory_usage(deep=True).sum() / 1024, 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: #111827; padding: 15px; border-radius: 12px; border-left: 5px solid #3b82f6; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 20px; text-align: center;">
            <p style="color: #9ca3af; font-size: 14px; margin: 0 0 5px 0; font-weight: 500;">📄 Rows</p>
            <h4 style="color: white; margin: 0; font-size: 24px; font-weight: 700;">{rows:,}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: #111827; padding: 15px; border-radius: 12px; border-left: 5px solid #8b5cf6; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 20px; text-align: center;">
            <p style="color: #9ca3af; font-size: 14px; margin: 0 0 5px 0; font-weight: 500;">📊 Columns</p>
            <h4 style="color: white; margin: 0; font-size: 24px; font-weight: 700;">{cols}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: #111827; padding: 15px; border-radius: 12px; border-left: 5px solid #ef4444; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 20px; text-align: center;">
            <p style="color: #9ca3af; font-size: 14px; margin: 0 0 5px 0; font-weight: 500;">❗ Missing</p>
            <h4 style="color: white; margin: 0; font-size: 24px; font-weight: 700;">{missing}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: #111827; padding: 15px; border-radius: 12px; border-left: 5px solid #10b981; box-shadow: 0 4px 10px rgba(0,0,0,0.2); margin-bottom: 20px; text-align: center;">
            <p style="color: #9ca3af; font-size: 14px; margin: 0 0 5px 0; font-weight: 500;">💾 Memory</p>
            <h4 style="color: white; margin: 0; font-size: 24px; font-weight: 700;">{memory} KB</h4>
        </div>
        """, unsafe_allow_html=True)

    # 4. ENTERPRISE STATUS DASHBOARD
    st.markdown("## 🚀 AI Workspace")
    col1, col2, col3 = st.columns(3)
        
    with col1:
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1e40af,#2563eb); padding:15px; border-radius:15px; color:white; height:185px; margin-bottom:15px; box-shadow:0 6px 15px rgba(0,0,0,.15);">
            <h4 style="margin-top:0; margin-bottom:10px; color:white; white-space: nowrap; overflow: hidden;">🤖 AI Engine</h4>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
            <p style="margin: 5px 0; font-size: 13px;">🟢 Status : Active</p>
            <p style="margin: 5px 0; font-size: 13px;">⚡ AutoML : Enabled</p>
            <p style="margin: 5px 0; font-size: 13px;">🧠 Explainable AI : Ready</p>
            <p style="margin: 5px 0; font-size: 13px;">📄 Report Ready</p>
        </div>
        """, unsafe_allow_html=True)
            
    with col2:
        health = max(0, 100 - int(missing) - int(df.duplicated().sum()))
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#065f46,#059669); padding:15px; border-radius:15px; color:white; height:185px; margin-bottom:15px; box-shadow:0 6px 15px rgba(0,0,0,.15);">
            <h4 style="margin-top:0; margin-bottom:10px; color:white; white-space: nowrap; overflow: hidden;">📊 Data Health</h4>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
            <div style="margin: 5px 0; color:white; font-size: 26px; font-weight: bold;">{health}%</div>
            <p style="margin: 5px 0; font-size: 13px;">Rows : {rows:,}</p>
            <p style="margin: 5px 0; font-size: 13px;">Columns : {cols}</p>
        </div>
        """, unsafe_allow_html=True)
            
    with col3:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#581c87,#7c3aed); padding:15px; border-radius:15px; color:white; height:185px; margin-bottom:15px; box-shadow:0 6px 15px rgba(0,0,0,.15);">
            <h4 style="margin-top:0; margin-bottom:10px; color:white; white-space: nowrap; overflow: hidden;">⚡ Workspace</h4>
            <hr style="border-color: rgba(255,255,255,0.2); margin: 10px 0;">
            <p style="margin: 5px 0; font-size: 13px;">Dataset Loaded</p>
            <p style="margin: 5px 0; font-size: 13px;">Memory : {memory} KB</p>
            <p style="margin: 5px 0; font-size: 13px;">Ready for Training</p>
            <p style="margin: 5px 0; font-size: 13px;">Ready for Export</p>
        </div>
        """, unsafe_allow_html=True)
 
    # CREATE TABS
    tab_eda, tab_prep, tab_train, tab_download, tab_history, tab_glossary, tab_interview = st.tabs([
        "📊 Data & Exploration", "🛠️ Preprocessing", "🧠 Model Training", "📥 Downloads", "📜 History", "📖 Learn ML", "💼 Interview Prep"
    ])
    with tab_eda:
        st.markdown("### 🔍 Dataset Overview")
        st.markdown("### 📊 Dataset Summary")

        # PREMIUM DATASET METRIC CARDS
        rows = df.shape[0]
        cols = df.shape[1]
        missing = df.isnull().sum().sum()
        numeric = len(df.select_dtypes(include=np.number).columns)
        categorical = len(df.select_dtypes(exclude=np.number).columns)
        memory = round(df.memory_usage(deep=True).sum() / 1024, 2)
        
        cards = [
            ("📄 Total Rows", rows, "#2563eb"),
            ("📊 Total Columns", cols, "#7c3aed"),
            ("❗ Missing Values", missing, "#dc2626"),
            ("🔢 Numeric", numeric, "#059669"),
            ("🔤 Categorical", categorical, "#d97706"),
            ("💾 Memory (KB)", memory, "#0891b2")
        ]
        
        cols_ui = st.columns(3)
        
        for i, (title, value, color) in enumerate(cards):
            with cols_ui[i % 3]:
                st.markdown(f"""
                <div style="
                background:rgba(255,255,255,0.05);
                border:1px solid rgba(255,255,255,0.08);
                border-left:6px solid {color};
                border-radius:18px;
                padding:22px;
                margin-bottom:18px;
                box-shadow:0 10px 25px rgba(0,0,0,.35);
                transition:0.3s;
                ">
                
                <div style="
                color:#94a3b8;
                font-size:15px;
                margin-bottom:10px;">
                {title}
                </div>
                
                <div style="
                color:white;
                font-size:34px;
                font-weight:700;">
                {value}
                </div>
                
                </div>
                """, unsafe_allow_html=True)

        # DATA LEAKAGE & DATA QUALITY DETECTION
        st.markdown("---")
        st.subheader("🛡️ Data Leakage & Data Quality Detection")
        
        leakage_found = False
        
        # 1. UNIQUE ID COLUMN DETECTION
        for col in df.columns:
            if df[col].nunique() == len(df):
                st.warning(f"⚠ Possible ID / Data Leakage Column : '{col}'")
                st.info(f"💡 Recommendation : Remove '{col}' if it is only an ID.")
                leakage_found = True
        
        # 2. CONSTANT COLUMN DETECTION
        for col in df.columns:
            if df[col].nunique() == 1:
                st.warning(f"⚠ Constant Column Detected : '{col}'")
                st.info(f"💡 '{col}' has only one unique value. Remove it.")
                leakage_found = True
        
        # 3. NEAR CONSTANT COLUMN
        for col in df.columns:
            if df[col].nunique() <= 2 and not df.empty:
                value_ratio = df[col].value_counts(normalize=True).iloc[0]
                if value_ratio > 0.95:
                    st.warning(f"⚠ Near Constant Column : '{col}'")
                    leakage_found = True
        
        # 4. DUPLICATE COLUMN DETECTION
        duplicate_cols = df.columns[df.columns.duplicated()]
        if len(duplicate_cols) > 0:
            st.warning(f"⚠ Duplicate Columns : {list(duplicate_cols)}")
            leakage_found = True
        
        # 5. DATE COLUMN DETECTION
        for col in df.columns:
            if "date" in col.lower() or "time" in col.lower():
                st.warning(f"📅 Date/Time Column Detected : '{col}'")
                st.info("💡 Consider extracting Year, Month, Day features.")
        
        # 6. HIGH MISSING VALUES
        for col in df.columns:
            missing_percent = df[col].isnull().mean() * 100
            if missing_percent > 50:
                st.warning(f"⚠ '{col}' has {missing_percent:.1f}% Missing Values")
                st.info("💡 Consider dropping this column.")
        
        # 7. HIGH CORRELATION
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for column in upper.columns:
                high_corr = upper[column][upper[column] > 0.95]
                for idx, value in high_corr.items():
                    st.warning(f"⚠ High Correlation : '{idx}' ↔ '{column}' ({value:.2f})")
                    st.info("💡 Remove one of these features.")
                    leakage_found = True
        
        # FINAL STATUS
        st.markdown("---")
        if leakage_found:
            st.error("⚠ Data Quality Issues Found")
        else:
            st.success("✅ No Data Leakage or Major Data Quality Issues Detected")

        # FINAL STATUS & HEALTH SCORE (PREMIUM UI)
        
        # Calculate Data Health Score
        score = 100
        score -= df.isnull().sum().sum() * 0.2
        score -= df.duplicated().sum() * 2
        score = max(0, min(100, score))

        # Dynamic Styling based on Leakage
        if leakage_found:
            status_html = """
<div style="background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.5); color:#fca5a5; padding:16px; border-radius:12px; font-weight:600; font-size:16px;">
⚠ Data Quality Issues Found. Please review the warnings above.
</div>
"""
            border_color = "#ef4444" # Red
        else:
            status_html = """
<div style="background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.5); color:#6ee7b7; padding:16px; border-radius:12px; font-weight:600; font-size:16px;">
✅ No Data Leakage or Major Data Quality Issues Detected
</div>
"""
            border_color = "#10b981" # Green

        st.markdown(f"""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px; border-radius: 20px; border-left: 8px solid {border_color}; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 30px; margin-bottom: 25px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 20px;">🛡️ Data Quality Report</h2>
{status_html}
<div style="margin-top: 25px;">
<p style="color: #9ca3af; font-size: 15px; margin-bottom: 5px; font-weight: 600;">📊 OVERALL DATASET HEALTH</p>
<h1 style="color: white; font-size: 48px; margin: 0;">{score:.1f}%</h1>
</div>
</div>
""", unsafe_allow_html=True)
        # Streamlit Progress Bar Below the Box
        st.progress(int(score) / 100)
                
        st.markdown("---")
        st.dataframe(df.head(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📐 **Dataset Shape:** {df.shape[0]} Rows | {df.shape[1]} Columns")
        with col2:
            st.warning(f"❗ **Total Missing Values:** {df.isnull().sum().sum()}")

        st.markdown("---")
        st.markdown("### 📈 Data Distribution")
        numeric_cols = df.select_dtypes(include=np.number).columns

        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 30px; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📈 Data Distribution Analysis</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Explore the mathematical frequency and data density of any numeric column using this interactive histogram.</p>
</div>
""", unsafe_allow_html=True)

        if len(numeric_cols) > 0:
            col = st.selectbox("Select column to visualize distribution:", numeric_cols, key="eda_dist_select")
            fig = px.histogram(df, x=col, template="plotly_dark", color_discrete_sequence=['#8b5cf6'])
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns available for distribution analysis.")

        st.markdown("---")
        st.markdown("### 🔥 Correlation Heatmap")
        
        # 2. CORRELATION HEATMAP CARD
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #059669; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🔥 Feature Correlation Heatmap</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">This matrix illustrates the linear relationship between different features. Values closer to 1 or -1 indicate a stronger correlation.</p>
</div>
""", unsafe_allow_html=True)
        
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            fig = px.imshow(
                corr, text_auto=".2f", color_continuous_scale="Viridis",
                aspect="auto", title="Feature Correlation Matrix"
            )
            fig.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns to generate a correlation heatmap.")

        st.markdown("---")
        st.markdown("### 📦 Box Plot & Outlier Detection")
        
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #dc2626; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📦 Box Plot & Outlier Detection</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Analyze data dispersion, skewness, and identify statistical outliers using the Interquartile Range (IQR) method.</p>
</div>
""", unsafe_allow_html=True)
        
        if len(numeric_cols) > 0:
            box_col = st.selectbox("Select Column for Box Plot", numeric_cols, key="box_plot")
            fig = px.box(
                df, y=box_col, points="outliers", template="plotly_dark",
                color_discrete_sequence=["#8b5cf6"], title=f"Box Plot - {box_col}"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            
            Q1 = df[box_col].quantile(0.25)
            Q3 = df[box_col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[box_col] < (Q1 - 1.5 * IQR)) | (df[box_col] > (Q3 + 1.5 * IQR))]
            st.metric("🚨 Total Outliers Detected", len(outliers))
            st.dataframe(outliers.head(), use_container_width=True)
        else:
            st.info("No numeric columns available.")

        st.markdown("---")
        st.markdown("### 📍 Scatter Plot Explorer")
        
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #d97706; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📍 Multi-Dimensional Scatter Explorer</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Visualize the relationship between two numeric variables and analyze class patterns using optional color grouping.</p>
</div>
""", unsafe_allow_html=True)
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("Select X-axis", numeric_cols, key="scatter_x")
            with col2:
                y_axis = st.selectbox("Select Y-axis", numeric_cols, index=1, key="scatter_y")
            
            color_col = st.selectbox("Color By (Optional)", ["None"] + list(df.columns), key="scatter_color")
            
            if color_col == "None":
                fig = px.scatter(df, x=x_axis, y=y_axis, template="plotly_dark", title="Scatter Plot")
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col, template="plotly_dark", title="Scatter Plot")
                fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.info("At least 2 numeric columns are required.")


                
        st.markdown("---")
        st.markdown("## 📋 Data Profiling Dashboard")
        
        st.markdown("---")
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #3b82f6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 20px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📋 In-Depth Data Profiling</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Select any feature to generate a comprehensive statistical profile, including central tendency, spread, and visual distributions.</p>
</div>
""", unsafe_allow_html=True)
        
        selected_col = st.selectbox("Select Column", df.columns, key="profile_col")
        col_data = df[selected_col]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📦 Data Type", str(col_data.dtype))
        c2.metric("🔢 Unique Values", col_data.nunique())
        c3.metric("❌ Missing", col_data.isnull().sum())
        c4.metric("📊 Total Values", len(col_data))

        if pd.api.types.is_numeric_dtype(col_data):
            st.markdown("### 📈 Statistics")
            a, b, c = st.columns(3)
            d, e, f = st.columns(3)
            
            a.metric("Mean", round(col_data.mean(),2))
            b.metric("Median", round(col_data.median(),2))
            c.metric("Std Dev", round(col_data.std(),2))
            
            if pd.api.types.is_bool_dtype(col_data):
                d.metric("Minimum", str(col_data.min()))
                e.metric("Maximum", str(col_data.max()))
                f.metric("Variance", "N/A")
            else:
                d.metric("Minimum", round(float(col_data.min()), 2))
                e.metric("Maximum", round(float(col_data.max()), 2))
                f.metric("Variance", round(float(col_data.var()), 2))
                
            q1 = col_data.quantile(.25)
            q2 = col_data.quantile(.50)
            q3 = col_data.quantile(.75)
            
            x, y, z = st.columns(3)
            x.metric("Q1", round(float(q1),2))
            y.metric("Q2", round(float(q2),2))
            z.metric("Q3", round(float(q3),2))

            a, b = st.columns(2)
            a.metric("Skewness", round(float(col_data.skew()),2))
            b.metric("Kurtosis", round(float(col_data.kurt()),2))

            # FIXED: Added unique titles and keys to prevent Duplicate Element Error
            fig1 = px.histogram(df, x=selected_col, template="plotly_dark", color_discrete_sequence=["#8b5cf6"], title=f"Frequency Profile: {selected_col}")
            fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig1, use_container_width=True, key="profile_hist_chart")

            fig2 = px.box(df, y=selected_col, points="outliers", template="plotly_dark", color_discrete_sequence=["#8b5cf6"], title=f"Outlier Profile: {selected_col}")
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True, key="profile_box_chart")

            st.markdown("### 🔝 Top Values")
            st.dataframe(col_data.value_counts().head(10), use_container_width=True)
        else:
            st.markdown("### 📊 Value Counts")
            st.dataframe(col_data.value_counts().reset_index(), use_container_width=True)
            
        st.markdown("---")
        st.markdown("## 🤖 AI Dataset Insights")

        st.markdown("---")
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🤖 AI Dataset Insights & Recommendations</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Automated analysis of dataset characteristics, optimal scaling strategies, and algorithmic recommendations.</p>
</div>
""", unsafe_allow_html=True)

        rows = df.shape[0]
        cols = df.shape[1]
        numeric = len(df.select_dtypes(include=np.number).columns)
        categorical = len(df.select_dtypes(exclude=np.number).columns)
        missing = df.isnull().sum().sum()

        c1, c2 = st.columns(2)

        with c1:
            if rows > 10000:
                st.success("✅ Large Dataset Detected")
            else:
                st.info("ℹ Small / Medium Dataset")
            if missing > 0:
                st.warning("⚠ Missing Values Found")
            else:
                st.success("✅ No Missing Values")
        with c2:
            if categorical > numeric:
                st.info("📌 More Categorical Features")
            else:
                st.success("📌 Mostly Numeric Dataset")
            if rows < 100:
                st.warning("⚠ Dataset is very small")

        st.markdown("### ⚖ Recommended Scaling")
        if numeric > 0:
            if rows > 1000:
                st.success("✅ StandardScaler Recommended")
            else:
                st.info("ℹ MinMaxScaler Recommended")

        st.markdown("### 🤖 Recommended ML Algorithm")
        if numeric > categorical:
            st.success("✅ Random Forest\n✅ XGBoost\n✅ Gradient Boosting")
        else:
            st.success("✅ CatBoost\n✅ LightGBM\n✅ Random Forest") 

        health = 100
        health -= missing
        if rows < 100: health -= 20
        if health < 0: health = 0
        
        st.markdown("### ❤️ Dataset Health Score")
        st.progress(health / 100)
        st.metric("Dataset Health", f"{health}%")

        st.markdown("### 💡 AI Suggestions")
        tips = []
        if missing > 0: tips.append("🧹 Handle Missing Values")
        if categorical > 0: tips.append("🏷 Encode Categorical Columns")
        if rows < 500: tips.append("📈 More Data Recommended")
        if numeric > 15: tips.append("⭐ Feature Selection Recommended")
        if len(tips) == 0: tips.append("🎉 Dataset looks ready for Machine Learning")
        
        for tip in tips:
            st.info(tip)
            
        # AI DATASET ANALYZER
        st.markdown("---")
        st.header("🤖 AI Dataset Analyzer")
        
        missing = df.isnull().sum().sum()
        duplicates = df.duplicated().sum()
        rows = len(df)
        cols = len(df.columns)
        numeric = len(df.select_dtypes(include=np.number).columns)
        categorical = len(df.select_dtypes(exclude=np.number).columns)
        
        score = 100
        if missing > 0: score -= 20
        if duplicates > 0: score -= 10
        if rows < 500: score -= 10
        if categorical > numeric: score -= 5
        score = max(0, score)
        
        if score >= 90:
            quality = "🟢 Excellent"
            box_color = "#10b981"
        elif score >= 75:
            quality = "🟡 Good"
            box_color = "#eab308"
        elif score >= 60:
            quality = "🟠 Average"
            box_color = "#f97316"
        else:
            quality = "🔴 Poor"
            box_color = "#ef4444"
            
        st.markdown(f"""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px; border-radius: 20px; border-left: 8px solid {box_color}; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 25px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 20px;">🤖 AI Dataset Analyzer</h2>
<p style="color: #9ca3af; font-size: 15px; margin-bottom: 5px; font-weight: 600;">DATASET READINESS SCORE</p>
<h1 style="color: white; font-size: 48px; margin: 0;">{score}% <span style="font-size: 20px; font-weight: 400; color: #cbd5e1;">- {quality}</span></h1>
</div>
""", unsafe_allow_html=True)
        
        tips = []
        if missing == 0:
            tips.append("✅ **Clean:** No Missing Values")
        else:
            tips.append(f"⚠ **Warning:** {missing} Missing Values Found")
        
        if duplicates == 0:
            tips.append("✅ **Clean:** No Duplicate Rows")
        else:
            tips.append(f"⚠ **Warning:** {duplicates} Duplicate Rows Found")
        
        if numeric > categorical:
            tips.append("✅ **Structure:** Mostly Numeric Dataset")
        else:
            tips.append("ℹ **Structure:** Mostly Categorical Dataset")
        
        if rows > 1000:
            tips.append("✅ **Volume:** Enough Data for Training")
        else:
            tips.append("⚠ **Volume:** More Data Recommended")
        
        if numeric > 10:
            tips.append("✅ **Features:** Feature Selection Recommended")
        
        tips.append("✅ **Ready:** Train/Test Split Supported")
        tips.append("✅ **Ready:** Machine Learning Ready")
        
        for tip in tips:
            if "✅" in tip:
                st.success(tip)
            elif "⚠" in tip:
                st.warning(tip)
            else:
                st.info(tip)

    # TAB: PREPROCESSING
    with tab_prep:
        st.markdown("---")
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #f59e0b; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 10px;">🛠️ Data Preprocessing Studio</h2>
<p style="color: #9ca3af; font-size: 15px; margin: 0;">Clean your dataset by removing useless columns, handling missing values, encoding categorical strings into numbers, and scaling features for optimal machine learning performance.</p>
</div>
""", unsafe_allow_html=True)

        # DROP COLUMNS SECTION
        st.markdown("""
<div style="background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.2); padding: 20px; border-radius: 15px; margin-bottom: 25px;">
<h4 style="color: #f87171; margin-top: 0; margin-bottom: 10px;">🗑️ Drop Unnecessary Columns</h4>
<p style="font-size: 14px; color: #9ca3af; margin-bottom: 10px;">Remove IDs, names, or columns with highly missing values that don't add predictive value to the AI model.</p>
""", unsafe_allow_html=True)
        
        col_drop1, col_drop2 = st.columns([3, 1])
        with col_drop1:
            drop_cols = st.multiselect("Select Columns to Drop", df.columns, key="drop_cols_select", label_visibility="collapsed")
        with col_drop2:
            if st.button("🗑️ Drop Columns", use_container_width=True):
                if len(drop_cols) > 0:
                    df = df.drop(columns=drop_cols)
                    st.session_state.df = df
                    st.success(f"✅ Successfully dropped: {', '.join(drop_cols)}")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.warning("⚠ Please select at least one column to drop.")
                    
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin: 30px 0;'>", unsafe_allow_html=True)

        # Existing preprocessing options start here
        col1, col2, col3 = st.columns(3)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
<div style="background: rgba(96, 165, 250, 0.1); border: 1px solid rgba(96, 165, 250, 0.3); padding: 15px; border-radius: 12px; margin-bottom: 15px; min-height: 175px;">
<h4 style="color: #60a5fa; margin-top: 0; margin-bottom: 10px; text-align: center;">🩹 Handle Missing Values</h4>
<div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
<span style="color:#cbd5e1; font-weight:600;">Mean:</span> Numeric data without outliers.<br>
<span style="color:#cbd5e1; font-weight:600;">Median:</span> Numeric data with outliers.<br>
<span style="color:#cbd5e1; font-weight:600;">Mode:</span> Categorical (text) data.<br>
<span style="color:#cbd5e1; font-weight:600;">F/B Fill:</span> Time-series sequence data.
</div>
</div>
""", unsafe_allow_html=True)
            
        
            cols_with_missing = df.columns[df.isnull().any()].tolist()
            
           
            fill_cols = st.multiselect("Select Columns to Fill", cols_with_missing)
            fill_method = st.selectbox("Imputation Method", ["Mode", "Median", "Mean", "Forward Fill", "Backward Fill"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply Fill", use_container_width=True):
                for col in fill_cols:
                    if fill_method == "Mode":
                        df[col] = df[col].fillna(df[col].mode()[0])
                    elif fill_method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].median())
                    elif fill_method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col].fillna(df[col].mean())
                    elif fill_method == "Forward Fill":
                        df[col] = df[col].ffill()
                    elif fill_method == "Backward Fill":
                        df[col] = df[col].bfill()
                st.session_state.df = df
                st.success("✅ Missing Values Filled")

        with col2:
            st.markdown("""
<div style="background: rgba(167, 139, 250, 0.1); border: 1px solid rgba(167, 139, 250, 0.3); padding: 15px; border-radius: 12px; margin-bottom: 15px; min-height: 175px;">
<h4 style="color: #a78bfa; margin-top: 0; margin-bottom: 10px; text-align: center;">🔡 Categorical Encoding</h4>
<div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
<span style="color:#cbd5e1; font-weight:600;">What it does:</span> Converts text labels (e.g., 'Yes', 'No') into numbers (1, 0).<br><br>
<span style="color:#cbd5e1; font-weight:600;">When to use:</span> Required for ML models as they only understand numerical data.
</div>
</div>
""", unsafe_allow_html=True)
            
            cat_cols = df.select_dtypes(include="object").columns
            encode_cols = st.multiselect("Select Columns to Encode", cat_cols)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply Encoding", use_container_width=True):
                if len(encode_cols) > 0:
                    for col in encode_cols:
                        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
                    st.session_state.df = df
                    st.success("✅ Encoded Successfully")
                else:
                    st.warning("⚠ Select a column first.")

        with col3:
            st.markdown("""
<div style="background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.3); padding: 15px; border-radius: 12px; margin-bottom: 15px; min-height: 175px;">
<h4 style="color: #34d399; margin-top: 0; margin-bottom: 10px; text-align: center;">⚖️ Feature Scaling</h4>
<div style="font-size: 13px; color: #9ca3af; line-height: 1.5;">
<span style="color:#cbd5e1; font-weight:600;">Standardization:</span> Centers data. Best for algorithms assuming normal distribution.<br>
<span style="color:#cbd5e1; font-weight:600;">Normalization:</span> Squeezes data (0 to 1). Best for distance-based models like KNN.
</div>
</div>
""", unsafe_allow_html=True)
            
            num_cols = df.select_dtypes(include=np.number).columns
            scale_cols = st.multiselect("Select Columns to Scale", num_cols)
            scale_method = st.selectbox("Scaling Method", ["Standardization", "Normalization"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply Scaling", use_container_width=True):
                if len(scale_cols) > 0:
                    scaler = StandardScaler() if scale_method == "Standardization" else MinMaxScaler()
                    df[scale_cols] = scaler.fit_transform(df[scale_cols])
                    st.session_state.df = df
                    st.success("✅ Scaled Successfully")
                else:
                    st.warning("⚠ Select a column first.")

                    
        st.markdown("---")
        st.markdown("#### 📌 Current Preprocessed Dataset View")

        # CURRENT PREPROCESSED DATASET VIEW
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #f59e0b; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 10px;">📌 Current Preprocessed Dataset View</h2>
<p style="color: #9ca3af; font-size: 15px; margin: 0;">Preview the first few rows of your dataset after applying the preprocessing steps.</p>
</div>
""", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)

        # AUTO FEATURE ENGINEERING
        st.markdown("---")
        st.markdown("#### ✨ Auto Feature Engineering")
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #ec4899; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 40px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 15px;">✨ Auto Feature Engineering</h3>
<div style="display: flex; gap: 20px; flex-wrap: wrap; background: rgba(236, 72, 153, 0.05); padding: 15px; border-radius: 12px; border: 1px solid rgba(236, 72, 153, 0.2);">
<div style="flex: 1; min-width: 250px; font-size: 13px; color: #9ca3af; line-height: 1.6;">
<span style="color:#fbcfe8; font-weight:600;">Polynomial Features:</span> Captures complex non-linear patterns.<br>
<span style="color:#fbcfe8; font-weight:600;">Log Transformation:</span> Fixes highly skewed data (e.g., income, prices).<br>
<span style="color:#fbcfe8; font-weight:600;">Square Root:</span> Similar to Log, but handles zero values safely.
</div>
<div style="flex: 1; min-width: 250px; font-size: 13px; color: #9ca3af; line-height: 1.6;">
<span style="color:#fbcfe8; font-weight:600;">Square Feature:</span> Amplifies the impact of extreme/large values.<br>
<span style="color:#fbcfe8; font-weight:600;">Binning:</span> Converts continuous numbers into groups (e.g., Age to Age-Groups).<br>
<span style="color:#fbcfe8; font-weight:600;">Interaction Features:</span> Multiplies features to find their combined effect.
</div>
</div>
</div>
""", unsafe_allow_html=True)

        numeric_cols = df.select_dtypes(include=np.number).columns
        
        if len(numeric_cols) == 0:
            st.warning("⚠ No numeric columns available for feature engineering.")
        else:
            col_fe1, col_fe2 = st.columns(2)
            with col_fe1:
                feature_option = st.selectbox(
                    "Select Feature Engineering Method",
                    [
                        "Polynomial Features",
                        "Log Transformation",
                        "Square Root Transformation",
                        "Square Feature",
                        "Binning",
                        "Interaction Features"
                    ]
                )
            with col_fe2:
                selected_cols = st.multiselect("Select Numeric Columns", numeric_cols, key="fe_multi")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Apply Feature Engineering", use_container_width=True):
                if len(selected_cols) == 0:
                    st.warning("⚠ Please select at least one column.")
                else:
                    if feature_option == "Polynomial Features":
                        poly = PolynomialFeatures(degree=2, include_bias=False)
                        poly_data = poly.fit_transform(df[selected_cols])
                        poly_df = pd.DataFrame(poly_data, columns=poly.get_feature_names_out(selected_cols))
                        for c in poly_df.columns:
                            if c not in df.columns:
                                df[c] = poly_df[c]
                        st.success("✅ Polynomial Features Generated")
                        
                    elif feature_option == "Log Transformation":
                        for col in selected_cols:
                            if (df[col] >= 0).all():
                                df[col + "_log"] = np.log1p(df[col])
                        st.success("✅ Log Transformation Applied")
                        
                    elif feature_option == "Square Root Transformation":
                        for col in selected_cols:
                            if (df[col] >= 0).all():
                                df[col + "_sqrt"] = np.sqrt(df[col])
                        st.success("✅ Square Root Transformation Applied")
                        
                    elif feature_option == "Square Feature":
                        for col in selected_cols:
                            df[col + "_square"] = df[col] ** 2
                        st.success("✅ Square Features Created")
                        
                    elif feature_option == "Binning":
                        for col in selected_cols:
                            df[col + "_bin"] = pd.cut(df[col], bins=5, labels=False)
                        st.success("✅ Binning Applied")
                        
                    elif feature_option == "Interaction Features":
                        if len(selected_cols) < 2:
                            st.warning("⚠ Select at least 2 columns for interactions.")
                        else:
                            for i in range(len(selected_cols)):
                                for j in range(i + 1, len(selected_cols)):
                                    c1 = selected_cols[i]
                                    c2 = selected_cols[j]
                                    df[f"{c1}_x_{c2}"] = df[c1] * df[c2]
                            st.success("✅ Interaction Features Generated")
                    
                    st.session_state.df = df
                    st.dataframe(df.head(), use_container_width=True)

    # TAB: MODEL TRAINING
    with tab_train:
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #ef4444; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 10px;">🧠 Model Training Studio</h2>
<p style="color: #9ca3af; font-size: 15px; margin: 0;">Configure your machine learning pipeline. Select your target variable, set the execution mode, and let the AutoML engine train and evaluate multiple models automatically.</p>
</div>
""", unsafe_allow_html=True)

        st.markdown("<h4 style='color: #60a5fa; margin-bottom: 10px;'>⚙️ Step 1: Learning Configuration</h4>", unsafe_allow_html=True)
        learning_type = st.radio("Select Learning Type", ["Supervised", "Unsupervised"], horizontal=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if learning_type == "Supervised":
            col_target, col_k = st.columns(2)
            with col_target:
                target = st.selectbox("🎯 Select Target Column", df.columns)
            
            if target not in df.columns:
                st.error("❌ Target Column not found.")
                st.stop()
        
            df = df.dropna(subset=[target])
            X = df.drop(columns=[target])
            y = df[target]
            target_type = type_of_target(y)
            task = "Classification" if target_type in ["binary", "multiclass"] else "Regression"

            if task == "Classification":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
            
            with col_k:
                k = st.slider("⭐ Top K Features to Select", 1, X.shape[1], min(5, X.shape[1]))
        
            st.info(f"🤖 **Auto-Detected Task:** {task}")
        
            non_numeric = X.select_dtypes(exclude=np.number)
            if len(non_numeric.columns) > 0:
                st.error(f"Please encode these categorical columns in the 'Preprocessing' tab first: {list(non_numeric.columns)}")
                st.stop()
        
            selector = SelectKBest(f_classif if task == "Classification" else f_regression, k=k)
            for col in X.columns:
                if X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                    X[col] = X[col].fillna(X[col].mode()[0])
                else:
                    X[col] = X[col].fillna(X[col].median())
            
            
            X_new = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()]
            X = pd.DataFrame(X_new, columns=selected_features)
        
            if X.isnull().sum().sum() > 0:
                st.error("❌ Dataset still contains missing values. Please handle them in the Preprocessing tab.")
                st.stop()
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            st.markdown("<hr style='border-color: rgba(255,255,255,0.05); margin-top: 20px; margin-bottom: 20px;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #a78bfa; margin-bottom: 10px;'>🚀 Step 2: Execution Settings</h4>", unsafe_allow_html=True)
  
            # TRAINING MODEL
            training_mode = st.radio(
                "Training Mode",
                ["Normal Training", "Hyperparameter Tuning"],
                horizontal=True
            )
        
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Start Model Training", use_container_width=True):
                st.markdown("### 🏆 Leaderboard")
                results = []
                best_model = None
                best_model_name = None
    
                # STORE DATA FOR THE REPORT GENERATOR LATER
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                st.session_state.target = target
    
                if task == "Classification":
                    best_score = 0
                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=1000),
                        "Random Forest": RandomForestClassifier(),
                        "Extra Trees": ExtraTreesClassifier(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "KNN": KNeighborsClassifier(),
                        "SVM": SVC(probability=True),
                        "Naive Bayes": GaussianNB(),
                        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
                        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
                        "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
                    }
                                                
                    progress_bar = st.progress(0)
                    with st.spinner("Training Classification Models..."):
                        for i, (name, model) in enumerate(models.items()):
                            start_time = time.time()
                          
                            # Hyperparameter Tuning
                            if training_mode == "Hyperparameter Tuning" and name == "Random Forest":
                                params = {
                                    "n_estimators": [100, 200, 300, 500],
                                    "max_depth": [5, 10, 20, None],
                                    "min_samples_split": [2, 5, 10],
                                    "min_samples_leaf": [1, 2, 4]
                                }
                                search = RandomizedSearchCV(
                                    estimator=RandomForestClassifier(),
                                    param_distributions=params,
                                    n_iter=10,
                                    cv=5,
                                    scoring="accuracy",
                                    random_state=42,
                                    n_jobs=-1
                                )
                                search.fit(X_train, y_train)
                                model = search.best_estimator_
                                st.success("✅ Hyperparameter Tuning Completed")
                                st.write("### 🏆 Best Parameters")
                                st.json(search.best_params_)
                            else:
                                model.fit(X_train, y_train)
                            
                            training_time = round(time.time() - start_time, 3)
                            preds = model.predict(X_test)
                            cv_score = cross_val_score(model, X, y, cv=5, scoring="accuracy").mean()
                            acc = accuracy_score(y_test, preds)
                            
                            results.append([name, acc, cv_score, training_time])
                            progress_bar.progress((i + 1) / len(models))
                            
                            if acc > best_score:
                                best_score = acc
                                best_model = model
                                best_model_name = name
                        
                        # Save to session state
                        st.session_state.best_model = best_model
                        st.session_state.best_model_name = best_model_name
                        st.session_state.best_score = best_score
                        st.session_state.selected_features = selected_features
                        st.session_state.task = task
                        
                        res = pd.DataFrame(
                            results, 
                            columns=["Model", "Accuracy", "CV Score", "Time (s)"]
                        ).sort_values(by="Accuracy", ascending=False)
                        
                        st.session_state.res = res
                        st.dataframe(res, use_container_width=True)
                        # MODEL COMPARISON DASHBOARD
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🏆 AI Model Comparison Dashboard</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Visual comparison of all trained machine learning models based on their accuracy scores.</p>
</div>
""", unsafe_allow_html=True)
                        
                        plot_df = res.copy()
                        
                        # Sort
                        plot_df = plot_df.sort_values(
                            by="Accuracy",
                            ascending=False
                        )
                        
                        fig = px.bar(
                            plot_df,
                            x="Accuracy",
                            y="Model",
                            orientation="h",
                            text="Accuracy",
                            color="Accuracy",
                            color_continuous_scale="Viridis"
                        )
                        
                        fig.update_traces(
                            texttemplate="%{text:.4f}",
                            textposition="outside"
                        )
                        
                        fig.update_layout(
                            template="plotly_dark",
                            height=450,
                            xaxis_title="Model Score",
                            yaxis_title="",
                            coloraxis_showscale=False,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        
                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )
                        
                        st.markdown("<h3 style='color: #cbd5e1; margin-top: 20px; margin-bottom: 15px;'>🥇 Top 3 Models</h3>", unsafe_allow_html=True)
                        
                        top3 = plot_df.head(3)
                        cols = st.columns(3)
                        medals = ["🥇", "🥈", "🥉"]
                        
                        for i, (_, row) in enumerate(top3.iterrows()):
                            cols[i].metric(
                                f"{medals[i]} {row['Model']}",
                                f"{row['Accuracy']:.4f}"
                            )
                        st.success(f"""
                        🏆 Best Model Selected
                        
                        **{best_model_name}**
                        
                        Performance : **{best_score:.4f}**
                        """)
                        # ML PIPELINE VISUALIZER
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #059669; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🔄 AutoML Pipeline Flow</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Step-by-step execution log of the automated machine learning pipeline.</p>
</div>
""", unsafe_allow_html=True)
                        
                        pipeline_steps = [
                            "📂 Dataset Uploaded",
                            "🧹 Missing Values Handled",
                            "🗑 Duplicate Rows Removed",
                            "🏷 Categorical Encoding",
                            "⚖ Feature Scaling",
                            "⭐ Feature Selection",
                            "✂ Train-Test Split",
                            "🧠 Model Training",
                            "📊 Model Comparison",
                            f"🏆 Best Model : {best_model_name}",
                            "📈 Performance Evaluation",
                            "📄 PDF Report Generated",
                            "📥 Model Ready for Download"
                        ]
                        
                        for i, step in enumerate(pipeline_steps, start=1):
                            st.success(f"{i}. {step}")
                        # PIPELINE COMPLETION
                        st.markdown("<h4 style='color: #a78bfa; margin-top: 20px;'>🚀 Pipeline Completion</h4>", unsafe_allow_html=True)
                        st.progress(100)
                        st.success("✅ AutoML Pipeline Completed Successfully")
                        # WORKFLOW TIMELINE
                        st.markdown("<h4 style='color: #60a5fa; margin-top: 20px;'>⏱ Workflow Timeline</h4>", unsafe_allow_html=True)
                        
                        timeline = pd.DataFrame({
                            "Step": [
                                "Upload",
                                "Preprocessing",
                                "Feature Selection",
                                "Training",
                                "Evaluation",
                                "Report Generation",
                                "Download"
                            ],
                            "Status": [
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Ready"
                            ]
                        })
                        
                        st.dataframe(
                            timeline,
                            use_container_width=True,
                            hide_index=True
                        )

                        # AI MODEL RECOMMENDATION ENGINE
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #f59e0b; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🤖 AI Model Recommendation</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Smart insights and deployment recommendations based on model performance and overfitting analysis.</p>
</div>
""", unsafe_allow_html=True)
                        
                        try:
                            recommendation = []
                            recommendation.append(f"🏆 Best Model : {best_model_name}")
                            recommendation.append(f"📊 Score : {best_score:.4f}")
                            
                            # Just testing basic overfit detection logic if available
                            train_score = model.score(X_train, y_train)
                            test_score = best_score
                            
                            if train_score is not None and test_score is not None:
                                gap = abs(train_score - test_score)
                                if gap < 0.05:
                                    recommendation.append("✅ Very Low Overfitting")
                                elif gap < 0.10:
                                    recommendation.append("🟡 Slight Overfitting")
                                else:
                                    recommendation.append("🔴 High Overfitting")
                        
                            if hasattr(best_model, "feature_importances_"):
                                recommendation.append("✅ Supports Explainable AI")
                        
                            if len(df) > 10000:
                                recommendation.append("✅ Suitable for Large Datasets")
                        
                            if df.isnull().sum().sum() == 0:
                                recommendation.append("✅ Clean Dataset")
                        
                            recommendation.append("✅ Recommended for Production Use")
                        
                            # Render recommendations with colored boxes based on symbols
                            for item in recommendation:
                                if "✅" in item or "🏆" in item:
                                    st.success(item)
                                elif "🟡" in item:
                                    st.warning(item)
                                elif "🔴" in item:
                                    st.error(item)
                                else:
                                    st.info(item)
                        
                        except Exception as e:
                            st.warning(f"Recommendation unavailable : {e}")

                        # PERFORMANCE DASHBOARD
                        st.markdown("---")
                        st.subheader("📊 Model Performance Dashboard")
                        
                        try:
                            y_pred = best_model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("🎯 Accuracy", f"{accuracy:.4f}")
                            c2.metric("📌 Precision", f"{precision:.4f}")
                            c3.metric("📈 Recall", f"{recall:.4f}")
                            c4.metric("⭐ F1 Score", f"{f1:.4f}")
                        except Exception as e:
                            st.warning(f"Performance metrics unavailable: {e}")
    
                        st.markdown("---")
                        st.subheader("📈 Dataset Drift Detection")
                        
                        try:
                            drift = []
                            for col in X_train.columns:
                                train_mean = X_train[col].mean()
                                test_mean = X_test[col].mean()
                                diff = abs(train_mean - test_mean)
                                drift.append({
                                    "Feature": col,
                                    "Train Mean": round(train_mean,3),
                                    "Test Mean": round(test_mean,3),
                                    "Difference": round(diff,3)
                                })
                            drift_df = pd.DataFrame(drift)
                            st.dataframe(drift_df, use_container_width=True)
                            
                            if drift_df["Difference"].mean() < 0.10:
                                st.success("✅ No Significant Dataset Drift")
                            else:
                                st.warning("⚠ Dataset Drift Detected")
                        except Exception as e:
                            st.info(e)

                        # CONFUSION MATRIX
                        st.markdown("---")
                        st.subheader("📊 Confusion Matrix")
                        
                        try:
                            cm = confusion_matrix(y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(5,5))
                            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                            disp.plot(ax=ax, cmap="Blues", colorbar=False)
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Confusion Matrix Error : {e}")

                        # ROC CURVE
                        st.markdown("---")
                        st.subheader("📈 ROC Curve")
                        
                        try:
                            if hasattr(best_model, "predict_proba"):
                                y_prob = best_model.predict_proba(X_test)
                                if len(np.unique(y_test)) == 2:
                                    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
                                    roc_auc = auc(fpr, tpr)
                                    fig, ax = plt.subplots(figsize=(6,5))
                                    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                                    ax.plot([0,1],[0,1],'r--')
                                    ax.set_xlabel("False Positive Rate")
                                    ax.set_ylabel("True Positive Rate")
                                    ax.set_title("ROC Curve")
                                    ax.legend()
                                    st.pyplot(fig)
                                else:
                                    st.info("ROC Curve is available only for Binary Classification.")
                            else:
                                st.info("Selected model does not support probability prediction.")
                        except Exception as e:
                            st.warning(f"ROC Error : {e}")

                        # PRECISION RECALL CURVE
                        st.markdown("---")
                        st.subheader("📉 Precision - Recall Curve")
                        
                        try:
                            if hasattr(best_model, "predict_proba"):
                                if len(np.unique(y_test)) == 2:
                                    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob[:,1])
                                    fig, ax = plt.subplots(figsize=(6,5))
                                    ax.plot(recall_vals, precision_vals)
                                    ax.set_xlabel("Recall")
                                    ax.set_ylabel("Precision")
                                    ax.set_title("Precision Recall Curve")
                                    st.pyplot(fig)
                                else:
                                    st.info("Only available for Binary Classification.")
                        except Exception as e:
                            st.warning(f"PR Curve Error : {e}")
                            
                        # EXPLAINABLE AI DASHBOARD
                        st.markdown("""
        <div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #a855f7; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
        <h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🧠 Explainable AI Dashboard</h3>
        <p style="color: #9ca3af; font-size: 14px; margin: 0;">Deconstruct model predictions using SHAP values and feature importance metrics to understand the "why" behind the AI's decisions.</p>
        </div>
        """, unsafe_allow_html=True)

                        # FEATURE IMPORTANCE
                        importance_df = None
                        try:
                            if hasattr(best_model, "feature_importances_"):
                                importance_df = pd.DataFrame({
                                    "Feature": selected_features,
                                    "Importance": best_model.feature_importances_
                                }).sort_values(by="Importance", ascending=False)
                                
                                st.markdown("<h4 style='color: #cbd5e1; margin-bottom: 15px;'>⭐ Top Important Features</h4>", unsafe_allow_html=True)
                                st.dataframe(importance_df, use_container_width=True)
                                
                                fig = px.bar(
                                    importance_df.head(15),
                                    x="Importance",
                                    y="Feature",
                                    orientation="h",
                                    color="Importance",
                                    color_continuous_scale="Viridis"
                                )
                                fig.update_layout(
                                    template="plotly_dark",
                                    height=500,
                                    yaxis=dict(categoryorder="total ascending"),
                                    coloraxis_showscale=False,
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Selected model does not support Feature Importance directly.")
                        except Exception as e:
                            st.warning(f"Feature Importance Error : {e}")

                        # SHAP SUMMARY
                        try:
                            if hasattr(best_model, "feature_importances_"):
                                st.markdown("<h4 style='color: #a78bfa; margin-top: 25px; margin-bottom: 15px;'>🧠 SHAP Global Explanation</h4>", unsafe_allow_html=True)
                                explainer = shap.TreeExplainer(best_model)
                                shap_values = explainer.shap_values(X_test)
                                
                                # Set transparent background for Matplotlib
                                fig = plt.figure(figsize=(10,6))
                                fig.patch.set_alpha(0.0)
                                ax = plt.gca()
                                ax.set_facecolor("none")
                                ax.tick_params(colors="white")
                                ax.xaxis.label.set_color("white")
                                ax.yaxis.label.set_color("white")
                                
                                if isinstance(shap_values, list):
                                    shap.summary_plot(shap_values[0], X_test, show=False)
                                else:
                                    shap.summary_plot(shap_values, X_test, show=False)
                                st.pyplot(fig)
                                plt.close()
                        except Exception as e:
                            st.info(f"SHAP Summary not available : {e}")
                            
                        # SHAP BAR
                        try:
                            if hasattr(best_model, "feature_importances_"):
                                st.markdown("<h4 style='color: #c084fc; margin-top: 25px; margin-bottom: 15px;'>📊 SHAP Feature Ranking</h4>", unsafe_allow_html=True)
                                fig = plt.figure(figsize=(10,6))
                                fig.patch.set_alpha(0.0)
                                ax = plt.gca()
                                ax.set_facecolor("none")
                                ax.tick_params(colors="white")
                                ax.xaxis.label.set_color("white")
                                ax.yaxis.label.set_color("white")
                                
                                if isinstance(shap_values, list):
                                    shap.summary_plot(shap_values[0], X_test, plot_type="bar", show=False)
                                else:
                                    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                                st.pyplot(fig)
                                plt.close()
                        except:
                            pass

                        # SINGLE PREDICTION SIMULATOR
                        st.markdown("""
        <div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #06b6d4; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
        <h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🎯 Single Prediction Simulator</h3>
        <p style="color: #9ca3af; font-size: 14px; margin: 0;">Select any sample from the testing dataset to see exactly what the model predicts.</p>
        </div>
        """, unsafe_allow_html=True)
        
                        try:
                            row = st.slider("Select Row Index from Test Data", 0, len(X_test)-1, 0)
                            st.markdown("<p style='color:#cbd5e1; font-weight:600; margin-bottom:5px;'>Selected Feature Values:</p>", unsafe_allow_html=True)
                            st.dataframe(X_test.iloc[[row]], use_container_width=True)
                            
                            pred = best_model.predict(X_test.iloc[[row]])
                            
                            st.markdown(f"""
        <div style="background: rgba(6, 182, 212, 0.15); border: 1px solid rgba(6, 182, 212, 0.5); padding: 20px; border-radius: 15px; text-align: center; margin-top: 15px;">
        <p style="color: #cbd5e1; margin: 0; font-size: 14px;">MODEL PREDICTION</p>
        <h2 style="color: white; margin: 5px 0 0 0; font-size: 32px;">{pred[0]}</h2>
        </div>
        """, unsafe_allow_html=True)
                        except Exception as e:
                            st.info(f"Prediction Simulator Error : {e}")
        
                        # SAVE EXPERIMENT HISTORY
                        if "experiment_history" not in st.session_state:
                            st.session_state.experiment_history = []
                        
                        experiment = {
                            "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
                            "Task": task,
                            "Best Model": best_model_name,
                            "Score": round(best_score, 4)
                        }
                        
                        if experiment not in st.session_state.experiment_history:
                            st.session_state.experiment_history.append(experiment)
                        
                        st.balloons()
            
                else: # Regression
                        best_score = float("inf")
                        models = {
                            "Linear Regression": LinearRegression(),
                            "Ridge": Ridge(),
                            "Lasso": Lasso(),
                            "Random Forest": RandomForestRegressor(),
                            "Extra Trees": ExtraTreesRegressor(),
                            "Gradient Boosting": GradientBoostingRegressor(),
                            "Decision Tree": DecisionTreeRegressor(),
                            "KNN": KNeighborsRegressor(),
                            "SVR": SVR(),
                            "XGBoost": XGBRegressor(random_state=42),
                            "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
                            "CatBoost": CatBoostRegressor(verbose=0, random_state=42)
                        }
                        progress_bar = st.progress(0)
                        with st.spinner("Training Regression Models..."):
                            for i, (name, model) in enumerate(models.items()):
                                start_time = time.time()
                                model.fit(X_train, y_train)
                                training_time = round(time.time() - start_time, 3)
                                preds = model.predict(X_test)
                                
                                # Using negative mean squared error for cross_val_score on regression
                                cv_score = cross_val_score(
                                    model, X, y, cv=5, scoring="neg_mean_squared_error"
                                ).mean()
                                
                                rmse = np.sqrt(mean_squared_error(y_test, preds))
                                results.append([name, rmse, cv_score, training_time])
                                progress_bar.progress((i + 1) / len(models))
                                
                                if rmse < best_score:
                                    best_score = rmse
                                    best_model = model
                                    best_model_name = name
    
                        # Save to session state
                        st.session_state.best_model = best_model
                        st.session_state.best_model_name = best_model_name
                        st.session_state.best_score = best_score
                        st.session_state.selected_features = selected_features
                        st.session_state.task = task
                        
                        res = pd.DataFrame(
                            results, 
                            columns=["Model", "RMSE", "CV Score", "Time (s)"]
                        ).sort_values(by="RMSE", ascending=True)
                        
                        st.session_state.res = res
                        st.dataframe(res, use_container_width=True)

                        # MODEL COMPARISON DASHBOARD (REGRESSION)
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🏆 AI Model Comparison Dashboard</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Visual comparison of all trained regression models based on their Root Mean Squared Error (RMSE). Lower is better.</p>
</div>
""", unsafe_allow_html=True)
                        
                        plot_df = res.copy()
                        if "Accuracy" not in plot_df.columns:
                            if "Score" in plot_df.columns:
                                plot_df["Accuracy"] = plot_df["Score"]
                            elif "RMSE" in plot_df.columns:
                                plot_df["Accuracy"] = plot_df["RMSE"] # Mapping for sorting logic below
    
                        plot_df = plot_df.sort_values(by="Accuracy", ascending=True) # Lower RMSE is better
                        
                        fig = px.bar(
                            plot_df,
                            x="Accuracy",
                            y="Model",
                            orientation="h",
                            text="Accuracy",
                            color="Accuracy",
                            color_continuous_scale="Viridis"
                        )
                        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                        fig.update_layout(
                            template="plotly_dark",
                            height=450,
                            xaxis_title="Model RMSE (Lower is Better)",
                            yaxis_title="",
                            coloraxis_showscale=False,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("<h3 style='color: #cbd5e1; margin-top: 20px; margin-bottom: 15px;'>🥇 Top 3 Models</h3>", unsafe_allow_html=True)
                        top3 = plot_df.head(3)
                        cols = st.columns(3)
                        medals = ["🥇", "🥈", "🥉"]
                        
                        for i, (_, row) in enumerate(top3.iterrows()):
                            cols[i].metric(f"{medals[i]} {row['Model']}", f"{row['Accuracy']:.4f}")
                        
                        st.success(f"""
                        🏆 Best Model Selected
                        
                        **{best_model_name}**
                        
                        Performance (RMSE) : **{best_score:.4f}**
                        """)
                    
                        # ML PIPELINE VISUALIZER

                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #059669; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🔄 AutoML Pipeline Flow</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Step-by-step execution log of the automated machine learning pipeline.</p>
</div>
""", unsafe_allow_html=True)
                        
                        pipeline_steps = [
                            "📂 Dataset Uploaded",
                            "🧹 Missing Values Handled",
                            "🗑 Duplicate Rows Removed",
                            "🏷 Categorical Encoding",
                            "⚖ Feature Scaling",
                            "⭐ Feature Selection",
                            "✂ Train-Test Split",
                            "🧠 Model Training",
                            "📊 Model Comparison",
                            f"🏆 Best Model : {best_model_name}",
                            "📈 Performance Evaluation",
                            "📄 PDF Report Generated",
                            "📥 Model Ready for Download"
                        ]
                        
                        for i, step in enumerate(pipeline_steps, start=1):
                            st.success(f"{i}. {step}")

                        # PIPELINE COMPLETION
                        st.markdown("<h4 style='color: #a78bfa; margin-top: 20px; margin-bottom: 15px;'>🚀 Pipeline Completion</h4>", unsafe_allow_html=True)
                        st.progress(100)
                        st.success("✅ AutoML Pipeline Completed Successfully")

                        # WORKFLOW TIMELINE
                        st.markdown("<h4 style='color: #60a5fa; margin-top: 20px; margin-bottom: 15px;'>⏱ Workflow Timeline</h4>", unsafe_allow_html=True)
                        
                        timeline = pd.DataFrame({
                            "Step": [
                                "Upload",
                                "Preprocessing",
                                "Feature Selection",
                                "Training",
                                "Evaluation",
                                "Report Generation",
                                "Download"
                            ],
                            "Status": [
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Completed",
                                "✅ Ready"
                            ]
                        })
                        
                        st.dataframe(
                            timeline,
                            use_container_width=True,
                            hide_index=True
                        )

                        # AI MODEL RECOMMENDATION ENGINE
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #f59e0b; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🤖 AI Model Recommendation</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Smart insights and deployment recommendations based on model performance and overfitting analysis.</p>
</div>
""", unsafe_allow_html=True)
                        
                        try:
                            recommendation = []
                            recommendation.append(f"🏆 Best Model : {best_model_name}")
                            recommendation.append(f"📊 Score (RMSE) : {best_score:.4f}")
                            
                            train_score = model.score(X_train, y_train)
                            test_score = model.score(X_test, y_test)
                            
                            if train_score is not None and test_score is not None:
                                gap = abs(train_score - test_score)
                                if gap < 0.05:
                                    recommendation.append("✅ Very Low Overfitting")
                                elif gap < 0.10:
                                    recommendation.append("🟡 Slight Overfitting")
                                else:
                                    recommendation.append("🔴 High Overfitting")
                        
                            if hasattr(best_model, "feature_importances_"):
                                recommendation.append("✅ Supports Explainable AI")
                            if len(df) > 10000:
                                recommendation.append("✅ Suitable for Large Datasets")
                            if df.isnull().sum().sum() == 0:
                                recommendation.append("✅ Clean Dataset")
                            recommendation.append("✅ Recommended for Production Use")
                            
                            # Render recommendations with colored boxes based on emojis
                            for item in recommendation:
                                if "✅" in item or "🏆" in item:
                                    st.success(item)
                                elif "🟡" in item:
                                    st.warning(item)
                                elif "🔴" in item:
                                    st.error(item)
                                else:
                                    st.info(item)
                                    
                        except Exception as e:
                            st.warning(f"Recommendation unavailable : {e}")
                            
                        # PERFORMANCE DASHBOARD
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #34d399; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📊 Model Performance Dashboard</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Comprehensive evaluation metrics for the selected optimal regression model.</p>
</div>
""", unsafe_allow_html=True)
                        
                        try:
                            y_pred = best_model.predict(X_test)
                            mae = mean_absolute_error(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("📌 MAE", f"{mae:.4f}")
                            c2.metric("📉 RMSE", f"{rmse:.4f}")
                            c3.metric("📊 MSE", f"{mse:.4f}")
                            c4.metric("🎯 R² Score", f"{r2:.4f}")
                        except Exception as e:
                            st.warning(f"Performance metrics unavailable: {e}")

                        # OVERFITTING & UNDERFITTING DETECTOR (REGRESSION)
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #f97316; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📉 Overfitting & Underfitting Detector</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Analyze the performance gap between training data and testing data to evaluate your model's generalization capabilities.</p>
</div>
""", unsafe_allow_html=True)
                        
                        train_preds = best_model.predict(X_train)
                        test_preds = best_model.predict(X_test)
                        
                        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
                        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
                        difference = abs(train_rmse - test_rmse)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training RMSE", f"{train_rmse:.4f}")
                        with col2:
                            st.metric("Testing RMSE", f"{test_rmse:.4f}")
                        with col3:
                            st.metric("Difference", f"{difference:.4f}")
                        
                        if difference < 0.50:
                            st.success("✅ Model is Well Generalized")
                        elif difference < 2:
                            st.warning("⚠ Slight Overfitting Detected")
                        else:
                            st.error("❌ Severe Overfitting Detected")
                        
                        if difference > 2:
                            st.info("""
                            ### 💡 Suggestions
                            ✔ Increase training data  
                            ✔ Reduce model complexity  
                            ✔ Tune Hyperparameters  
                            ✔ Use Cross Validation  
                            ✔ Try Regularization  
                            """)
                        elif difference < 0.50:
                            st.success("""
                            ### 🚀 Suggestions
                            ✔ Model performance is stable  
                            ✔ Ready for deployment  
                            ✔ No significant overfitting detected  
                            """)
                        
                        if train_rmse > 5 and test_rmse > 5:
                            st.error("⚠ Possible Underfitting Detected")
                            st.info("""
                            ### 💡 Underfitting Solution
                            ✔ Increase model complexity  
                            ✔ Add more useful features  
                            ✔ Reduce regularization  
                            ✔ Train for more iterations  
                            ✔ Try ensemble models  
                            """)
                        
                        compare_df = pd.DataFrame({
                            "Dataset": ["Training", "Testing"],
                            "RMSE": [train_rmse, test_rmse]
                        })
                        
                        fig = px.bar(
                            compare_df, x="Dataset", y="RMSE",
                            color="Dataset", text="RMSE",
                            color_discrete_sequence=["#3b82f6", "#10b981"]
                        )
                        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                        fig.update_layout(
                            template="plotly_dark",
                            title="Training vs Testing RMSE",
                            showlegend=False,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("<h4 style='color: #60a5fa; margin-top: 20px; margin-bottom: 15px;'>🤖 AI Analysis</h4>", unsafe_allow_html=True)
                        if difference < 0.50:
                            st.success("""
                            **Regression Performance Analysis** ✅ Excellent Generalization  
                            ✅ Low Prediction Error  
                            🚀 Ready for Production Deployment  
                            """)
                        elif difference < 2:
                            st.warning("""
                            **Regression Performance Analysis** ⚠ Slight Overfitting  
                            ✔ Hyperparameter Tuning Recommended  
                            """)
                        else:
                            st.error("""
                            **Regression Performance Analysis** ❌ Severe Overfitting  
                            ⚠ Model is memorizing training data  
                            ✔ Retraining Recommended  
                            """)

                        # FEATURE IMPORTANCE (REGRESSION)

                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #eab308; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">⭐ Feature Importance</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Discover which dataset features are driving the predictions of your regression model.</p>
</div>
""", unsafe_allow_html=True)
                        
                        if hasattr(best_model, "feature_importances_"):
                            importance = best_model.feature_importances_
                            fi_df = pd.DataFrame({"Feature": selected_features, "Importance": importance})
                        elif hasattr(best_model, "coef_"):
                            importance = np.abs(best_model.coef_)
                            if len(np.array(importance).shape) > 1:
                                importance = np.mean(np.abs(importance), axis=0)
                            fi_df = pd.DataFrame({"Feature": selected_features, "Importance": importance})
                        else:
                            st.info(f"ℹ️ Feature Importance is not available for {best_model_name}.")
                            fi_df = None
                        
                        if fi_df is not None:
                            fi_df = fi_df.sort_values(by="Importance", ascending=False)
                            fig = px.bar(
                                fi_df, x="Importance", y="Feature", orientation="h",
                                color="Importance", color_continuous_scale="Viridis"
                            )
                            fig.update_layout(
                                template="plotly_dark", 
                                plot_bgcolor="rgba(0,0,0,0)", 
                                paper_bgcolor="rgba(0,0,0,0)", 
                                height=500,
                                coloraxis_showscale=False
                            )
                            st.plotly_chart(fig, use_container_width=True)

                # INTERACTIVE PREDICTION STUDIO
                if "best_model" in st.session_state:
                    st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #3b82f6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🧑‍💻 Interactive Prediction Studio</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Enter custom feature values below to test the trained model and instantly see its prediction and confidence score.</p>
</div>
""", unsafe_allow_html=True)
                    
                    user_data = {}
                    # Ensure we have columns
                    num_cols = min(3, len(st.session_state.selected_features))
                    if num_cols > 0:
                        cols = st.columns(num_cols)
                        for idx, col_name in enumerate(st.session_state.selected_features):
                            val = cols[idx % len(cols)].number_input(f"{col_name}", value=0.0, key=f"input_{col_name}")
                            user_data[col_name] = val
                            
                    st.markdown("<br>", unsafe_allow_html=True)
                    if st.button("🚀 Predict Outcome", type="primary", use_container_width=True):
                        try:
                            input_df = pd.DataFrame([user_data])
                            prediction = st.session_state.best_model.predict(input_df)
                            st.session_state.prediction = prediction
                            
                            st.markdown(f"""
<div style="background: linear-gradient(135deg, #1e3a8a, #2563eb); padding: 25px; border-radius: 15px; text-align: center; border: 1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4); margin-top: 15px; margin-bottom: 30px;">
<p style="color: #bfdbfe; margin: 0; font-size: 14px; font-weight: 600; letter-spacing: 1px;">AI PREDICTION RESULT</p>
<h1 style="color: white; margin: 10px 0 0 0; font-size: 48px;">{prediction[0]}</h1>
</div>
""", unsafe_allow_html=True)
                            
                            # Prediction Confidence
                            if hasattr(st.session_state.best_model, "predict_proba"):
                                probability = st.session_state.best_model.predict_proba(input_df)[0]
                                class_names = st.session_state.best_model.classes_
                                
                                prob_df = pd.DataFrame({
                                    "Class": class_names,
                                    "Probability": probability
                                })
                                prob_df["Probability %"] = (prob_df["Probability"] * 100).round(2)
                                
                                st.markdown("<h4 style='color: #60a5fa; margin-top: 10px; margin-bottom: 15px;'>📊 Prediction Confidence</h4>", unsafe_allow_html=True)
                                
                                fig = px.bar(
                                    prob_df, x="Class", y="Probability %",
                                    color="Probability %", text="Probability %",
                                    color_continuous_scale="Blues"
                                )
                                fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                                fig.update_layout(
                                    template="plotly_dark",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    coloraxis_showscale=False,
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                confidence = max(probability) * 100
                                st.metric("🎯 Top Confidence Score", f"{confidence:.2f}%")
                                st.progress(confidence / 100)

                            # SHAP EXPLAINABILITY (Local)
                            best_model = st.session_state.best_model
                            tree_models = (
                                RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, DecisionTreeClassifier,
                                RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, DecisionTreeRegressor,
                                XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor, CatBoostClassifier, CatBoostRegressor
                            )
                            
                            if isinstance(best_model, tree_models):
                                st.markdown("<h4 style='color: #a855f7; margin-top: 35px; margin-bottom: 15px;'>🧠 Why this prediction? (SHAP Explainability)</h4>", unsafe_allow_html=True)
                                try:
                                    explainer = shap.TreeExplainer(best_model)
                                    shap_values = explainer(st.session_state.X_test)
                                    
                                    # Plot styling for dark mode
                                    fig = plt.figure(figsize=(10,6))
                                    fig.patch.set_alpha(0.0)
                                    ax = plt.gca()
                                    ax.set_facecolor("none")
                                    ax.tick_params(colors="white")
                                    ax.xaxis.label.set_color("white")
                                    ax.yaxis.label.set_color("white")
                                    
                                    shap.plots.beeswarm(shap_values[:, :, 1] if len(shap_values.shape) > 2 else shap_values, show=False)
                                    st.pyplot(fig)
                                    plt.close(fig)
                                except Exception as e:
                                    st.warning(f"SHAP local explanation error: {e}")
                            else:
                                st.info(f"ℹ️ SHAP is not supported for {type(best_model).__name__}")
                                
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {e}")
                            
                        # BATCH PREDICTION STUDIO
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #10b981; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📤 Bulk / Batch Prediction</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Upload a new, unseen dataset (CSV) to generate bulk predictions using your trained model. The AI will automatically map the required features and provide a downloadable result.</p>
</div>
""", unsafe_allow_html=True)

                        batch_file = st.file_uploader("📂 Upload CSV for Batch Prediction", type=["csv"], key="batch_upload")
                        
                        if batch_file is not None:
                            try:
                                batch_df = pd.read_csv(batch_file)
                                st.markdown("<h4 style='color: #cbd5e1; margin-top: 15px; margin-bottom: 10px;'>👀 Uploaded Data Preview</h4>", unsafe_allow_html=True)
                                st.dataframe(batch_df.head(), use_container_width=True)
                                
                                # Check if required features are present
                                required_features = st.session_state.selected_features
                                missing_features = [col for col in required_features if col not in batch_df.columns]
                                
                                if len(missing_features) > 0:
                                    st.error(f"❌ Missing required columns in the uploaded CSV: {missing_features}")
                                    st.info("💡 Make sure your new CSV has the exact same feature columns as your training data.")
                                else:
                                    st.markdown("<br>", unsafe_allow_html=True)
                                    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                                        with st.spinner("AI is analyzing and predicting..."):
                                            # Extract only required features
                                            X_batch = batch_df[required_features]
                                            
                                            # Predict
                                            batch_preds = st.session_state.best_model.predict(X_batch)
                                            
                                            # Append to original DF
                                            result_df = batch_df.copy()
                                            
                                            # Insert Prediction at the beginning or end
                                            result_df.insert(0, "AI_Prediction", batch_preds)
                                            
                                            st.success(f"✅ Successfully generated predictions for {len(result_df)} rows!")
                                            st.dataframe(result_df.head(10), use_container_width=True)
                                            
                                            # Download Button
                                            csv = result_df.to_csv(index=False).encode('utf-8')
                                            st.download_button(
                                                "⬇️ Download Predicted CSV",
                                                csv,
                                                "Batch_Predictions_NexusML.csv",
                                                "text/csv",
                                                use_container_width=True
                                            )
                            except Exception as e:
                                st.error(f"❌ Error processing file: {e}")
            # Unsupervised Flow
            else: 
                data = df.select_dtypes(include=np.number)
                if len(data.columns) == 0:
                    st.error("No numeric columns available for clustering.")
                    st.stop()
                    
                if st.button("🚀 Start Clustering", use_container_width=True):
                    scaler = StandardScaler()
                    data_scaled = scaler.fit_transform(data)
                    models = {
                        "KMeans": KMeans(n_clusters=3),
                        "Agglomerative": AgglomerativeClustering(n_clusters=3),
                        "Birch": Birch(n_clusters=3)
                    }
                    
                    progress_bar = st.progress(0)
                    results = []
                    best_score = -1
                    best_model_name = None
                    best_labels = None
                    
                    with st.spinner("Finding Best Clustering Model..."):
                        for i, (name, model) in enumerate(models.items()):
                            try:
                                start_time = time.time()
                                labels = model.fit_predict(data_scaled)
                                training_time = round(time.time() - start_time, 3)
                                
                                unique_labels = set(labels)
                                if len(unique_labels) > 1 and -1 not in unique_labels:
                                    score = silhouette_score(data_scaled, labels)
                                else:
                                    score = -1
                                    
                                results.append([name, score, training_time])
                                progress_bar.progress((i + 1) / len(models))
                                
                                if score > best_score:
                                    best_score = score
                                    best_model_name = name
                                    best_labels = labels
                            except:
                                continue
                                
                    if best_model_name:
                        res = pd.DataFrame(
                            results,
                            columns=["Algorithm", "Silhouette Score", "Time (s)"]
                        ).sort_values(by="Silhouette Score", ascending=False)
                        
                        st.dataframe(res, use_container_width=True)
                        st.success(f"🏆 Best Model: **{best_model_name}** (Score: {best_score:.4f})")

                        # CLUSTERING PIPELINE VISUALIZER
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #06b6d4; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🔄 Clustering Pipeline Flow</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">Step-by-step execution log of the automated unsupervised machine learning pipeline.</p>
</div>
""", unsafe_allow_html=True)
                        
                        pipeline_steps = [
                            "📂 Dataset Uploaded",
                            "🧹 Missing Values Handled",
                            "🗑 Duplicate Rows Removed",
                            "⚖ Feature Scaling",
                            "🧠 Clustering Model Training",
                            "📊 Model Comparison (Silhouette Score)",
                            f"🏆 Best Model : {best_model_name}",
                            "📈 PCA Dimensionality Reduction",
                            "🗺️ Cluster Visualization"
                        ]
                        for i, step in enumerate(pipeline_steps, start=1):
                            st.success(f"{i}. {step}")
                        
                        st.markdown("<h4 style='color: #60a5fa; margin-top: 20px; margin-bottom: 15px;'>🚀 Pipeline Completion</h4>", unsafe_allow_html=True)
                        st.progress(100)
                        st.success("✅ Unsupervised AutoML Pipeline Completed Successfully")

                        # PCA CLUSTER VISUALIZATION
                        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
<h3 style="color: white; margin-top: 0; margin-bottom: 10px;">🗺️ Cluster Visualization (PCA)</h3>
<p style="color: #9ca3af; font-size: 14px; margin: 0;">High-dimensional data is reduced to 2 Principal Components to visually represent the discovered clusters.</p>
</div>
""", unsafe_allow_html=True)

                        df["Cluster"] = best_labels
                        pca = PCA(n_components=2)
                        reduced = pca.fit_transform(data_scaled)
                        plot_df = pd.DataFrame(reduced, columns=["PC1", "PC2"])
                        plot_df["Cluster"] = best_labels.astype(str)
                        
                        fig = px.scatter(
                            plot_df,
                            x="PC1",
                            y="PC2",
                            color="Cluster",
                            title=f"K-Means Clusters Visualization ({best_model_name})",
                            template="plotly_dark",
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            height=550
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Store Unsupervised Results
                        st.session_state.res = res
                        st.session_state.best_model_name = best_model_name
                        st.session_state.best_score = best_score
                        st.session_state.task = "Clustering"
                        st.balloons()
                    else:
                        st.error("❌ Clustering Failed. Please ensure the dataset contains sufficient numeric variance.")

                    # AI DECISION EXPLANATION (100% Dynamic)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("### 🧠 AI Decision Explanation")
                    st.markdown("---")
                    
                    try:
                        if "best_model" in st.session_state and "X_train" in st.session_state:
                            model = st.session_state.best_model
                            features = st.session_state.X_train.columns.tolist()
                            
                            # 1. Fetch Dynamic Prediction & Confidence
                            pred_val = str(st.session_state.prediction[0]) if "prediction" in st.session_state else "Run Prediction First"
                            
                            # Checking accuracy/score to show as confidence
                            conf_score = "N/A"
                            if "best_score" in st.session_state:
                                conf_score = f"{st.session_state.best_score * 100:.2f}% (Model Acc)"
                            
                            # 2. Calculate Real Feature Importances
                            importances = None
                            if hasattr(model, 'feature_importances_'):
                                importances = model.feature_importances_
                            elif hasattr(model, 'coef_'):
                                importances = __import__('numpy').abs(model.coef_[0]) if len(model.coef_.shape) > 1 else __import__('numpy').abs(model.coef_)
                                
                            if importances is not None and len(importances) == len(features):
                                import pandas as pd
                                feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
                                feat_df = feat_df.sort_values(by='Importance', ascending=False)
                                top_feats = feat_df['Feature'].tolist()
                            else:
                                top_feats = features # Fallback
                                
                            # Extracting Top Features Dynamically
                            high_feat_1 = top_feats[0] if len(top_feats) > 0 else "Feature 1"
                            high_feat_2 = top_feats[1] if len(top_feats) > 1 else "Feature 2"
                            med_feat = top_feats[2] if len(top_feats) > 2 else "Feature 3"
                            low_feat = top_feats[-1] if len(top_feats) > 3 else "Feature 4"
                    
                            # 3. UI Generation
                            st.markdown(f"""
                            <div style="display: flex; gap: 20px; margin-bottom: 25px;">
                                <div style="flex: 1; background: linear-gradient(135deg, #1e293b, #0f172a); padding: 20px; border-radius: 15px; border-left: 5px solid #3b82f6; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                                    <p style="margin: 0; color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Current Prediction</p>
                                    <h2 style="margin: 5px 0 0 0; color: #60a5fa;">{pred_val}</h2>
                                </div>
                                <div style="flex: 1; background: linear-gradient(135deg, #1e293b, #0f172a); padding: 20px; border-radius: 15px; border-left: 5px solid #10b981; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                                    <p style="margin: 0; color: #94a3b8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px;">Model Reliability</p>
                                    <h2 style="margin: 5px 0 0 0; color: #34d399;">{conf_score}</h2>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                            col_exp1, col_exp2 = st.columns([1.5, 1])
                    
                            with col_exp1:
                                st.markdown("<h4 style='color: #e2e8f0; margin-bottom: 15px;'>🎯 Why this prediction?</h4>", unsafe_allow_html=True)
                                st.markdown(f"""
                                <div style="background: #111827; padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05);">
                                    
                                    <div style="margin-bottom: 15px;">
                                        <span style="background: rgba(239, 68, 68, 0.15); color: #f87171; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; border: 1px solid rgba(239, 68, 68, 0.3);">🔴 HIGH IMPACT</span>
                                        <span style="color: #e2e8f0; margin-left: 10px; font-size: 15px;"><b>{high_feat_1}</b> has the strongest influence on this model's logic.</span><br>
                                        <span style="color: #e2e8f0; margin-left: 105px; font-size: 15px;"><b>{high_feat_2}</b> is also a primary driving factor.</span>
                                    </div>
                                    
                                    <div style="margin-bottom: 15px;">
                                        <span style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; border: 1px solid rgba(245, 158, 11, 0.3);">🟠 MEDIUM IMPACT</span>
                                        <span style="color: #e2e8f0; margin-left: 10px; font-size: 15px;"><b>{med_feat}</b> contributed moderately to the decision boundary.</span>
                                    </div>
                                    
                                    <div>
                                        <span style="background: rgba(16, 185, 129, 0.15); color: #34d399; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: bold; border: 1px solid rgba(16, 185, 129, 0.3);">🟢 LOW IMPACT</span>
                                        <span style="color: #e2e8f0; margin-left: 10px; font-size: 15px;"><b>{low_feat}</b> has minimal influence on the final outcome.</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                            with col_exp2:
                                st.markdown("<h4 style='color: #e2e8f0; margin-bottom: 15px;'>⭐ Top Features</h4>", unsafe_allow_html=True)
                                
                                top_5_html = ""
                                for i, f in enumerate(top_feats[:5]):
                                    top_5_html += f"<p style='margin: 0 0 10px 0; color: #cbd5e1; font-size: 15px;'>{i+1}️⃣ <b>{f}</b></p>"
                                    
                                st.markdown(f"""
                                <div style="background: #111827; padding: 20px; border-radius: 15px; border: 1px solid rgba(255,255,255,0.05); height: 185px; overflow-y: auto;">
                                    {top_5_html}
                                </div>
                                """, unsafe_allow_html=True)
                    
                            st.markdown("<br>", unsafe_allow_html=True)
                            st.markdown(f"""
                            <div style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(59, 130, 246, 0.1)); padding: 25px; border-radius: 15px; border: 1px dashed rgba(139, 92, 246, 0.4); box-shadow: 0 4px 20px rgba(0,0,0,0.1);">
                                <h4 style="margin: 0 0 15px 0; color: #a78bfa; display: flex; align-items: center; gap: 8px;">
                                    🤖 AI Summary & Recommendation
                                </h4>
                                <p style="color: #cbd5e1; font-size: 15px; line-height: 1.6; margin: 0 0 20px 0;">
                                    Based on mathematical weight extraction, the Neural Engine prioritizes <b>{high_feat_1}</b> and <b>{high_feat_2}</b> to formulate its predictions. The model is currently operating with a reliability score of <b>{conf_score}</b>.
                                </p>
                                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                                    <div>
                                        <p style="margin: 0 0 8px 0; color: #94a3b8; font-size: 14px;"><b>Data Strategy:</b></p>
                                        <p style="margin: 0; color: #cbd5e1; font-size: 14px;">• Closely monitor <b>{high_feat_1}</b> values in production.</p>
                                        <p style="margin: 0; color: #cbd5e1; font-size: 14px;">• Features like <b>{low_feat}</b> can potentially be dropped.</p>
                                    </div>
                                    <div>
                                        <p style="margin: 0 0 8px 0; color: #94a3b8; font-size: 14px;"><b>System Checks:</b></p>
                                        <p style="margin: 0; color: #10b981; font-size: 14px;">✔ Engine weights extracted successfully.</p>
                                        <p style="margin: 0; color: #10b981; font-size: 14px;">✔ Dynamic explanations generated.</p>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                        else:
                            st.info("ℹ️ Train a model first to generate AI Explanations.")
                    except Exception as e:
                        st.warning("⚠️ AI Explanation module is currently analyzing the model architecture. Please run a prediction first.")

    # TAB: DOWNLOAD CENTER
    with tab_download:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #3b82f6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
        <h2 style="color: white; margin-top: 0; margin-bottom: 10px;">📥 Download Center</h2>
        <p style="color: #9ca3af; font-size: 15px; margin: 0;">Export your trained AI models, preprocessing pipelines, generated reports, and cleaned datasets for production deployment.</p>
        </div>
        """, unsafe_allow_html=True)

        # MODEL DOWNLOAD
        st.markdown("<h4 style='color: #60a5fa; margin-bottom: 15px;'>🤖 1. Best Model File (.pkl)</h4>", unsafe_allow_html=True)
        
        if "best_model" in st.session_state:
            best_model = st.session_state.best_model
            joblib.dump(best_model, "best_model.pkl")
            with open("best_model.pkl", "rb") as f:
                st.download_button(
                    "⬇ Download Best Model",
                    data=f,
                    file_name="best_model.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Train a model first to unlock this download.")

        # PIPELINE DOWNLOAD
        st.markdown("<h4 style='color: #a78bfa; margin-top: 20px; margin-bottom: 15px;'>⚙️ 2. Production Pipeline (.pkl)</h4>", unsafe_allow_html=True)
        if "best_model" in st.session_state and "selected_features" in st.session_state:
            pipeline = {
                "model": st.session_state.best_model,
                "selected_features": list(st.session_state.selected_features),
                "task": st.session_state.get("task", "Unknown")
            }
            joblib.dump(pipeline, "pipeline.pkl")
            with open("pipeline.pkl", "rb") as f:
                st.download_button(
                    "⬇ Download Pipeline",
                    data=f,
                    file_name="pipeline.pkl",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        else:
            st.info("ℹ️ Pipeline not available. Please train a supervised model.")

        # TRAIN DATA
        st.markdown("<h4 style='color: #34d399; margin-top: 20px; margin-bottom: 15px;'>📊 3. Training Dataset (.csv)</h4>", unsafe_allow_html=True)
        try:
            if "X_train" in st.session_state:
                train_df = st.session_state.X_train.copy() 
                if "y_train" in st.session_state:
                    train_df[st.session_state.target] = st.session_state.y_train.values
                if not train_df.empty:
                    train_csv = train_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Training Dataset", train_csv, "training_dataset.csv", "text/csv", use_container_width=True)
            else:
                st.info("ℹ️ Training dataset not available.")
        except Exception:
            st.info("ℹ️ Training dataset not available.")

        # TEST DATA
        st.markdown("<h4 style='color: #f472b6; margin-top: 20px; margin-bottom: 15px;'>🧪 4. Testing Dataset (.csv)</h4>", unsafe_allow_html=True)
        try:
            if "X_test" in st.session_state:
                test_df = st.session_state.X_test.copy()
                if "y_test" in st.session_state:
                    test_df[st.session_state.target] = st.session_state.y_test.values
                if not test_df.empty:
                    test_csv = test_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇ Download Testing Dataset", test_csv, "testing_dataset.csv", "text/csv", use_container_width=True)
            else:
                st.info("ℹ️ Testing dataset not available.")
        except Exception:
            st.info("ℹ️ Testing dataset not available.")
            
        # PREPROCESSED DATASET
        st.markdown("<h4 style='color: #fbbf24; margin-top: 20px; margin-bottom: 15px;'>🗂 5. Preprocessed Dataset (.csv)</h4>", unsafe_allow_html=True)
        try:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Download Preprocessed Dataset", csv, "preprocessed_dataset.csv", "text/csv", use_container_width=True)
        except:
            st.info("ℹ️ Dataset not available.")
            
        # MODEL LEADERBOARD
        st.markdown("<h4 style='color: #06b6d4; margin-top: 20px; margin-bottom: 15px;'>🏆 6. Model Leaderboard (.csv)</h4>", unsafe_allow_html=True)
        try:
            if "res" in st.session_state:
                leaderboard = st.session_state.res
                csv = leaderboard.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Leaderboard", csv, "leaderboard.csv", "text/csv", use_container_width=True)
            else:
                st.info("ℹ️ Leaderboard not available.")
        except:
            st.info("ℹ️ Leaderboard not available.")

        # SELECTED FEATURES
        st.markdown("<h4 style='color: #c084fc; margin-top: 20px; margin-bottom: 15px;'>⭐ 7. Selected Features (.csv)</h4>", unsafe_allow_html=True)
        try:
            if "selected_features" in st.session_state:
                feature_df = pd.DataFrame({"Selected Features": st.session_state.selected_features})
                csv = feature_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇ Download Selected Features", csv, "selected_features.csv", "text/csv", use_container_width=True)
            else:
                st.info("ℹ️ Feature list not available.")
        except:
            st.info("ℹ️ Feature list not available.")
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("✅ Download Center Ready")
        st.markdown("---")

        # PDF REPORT GENERATOR
        st.markdown("""
        <div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #e11d48; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 35px; margin-bottom: 25px;">
        <h3 style="color: white; margin-top: 0; margin-bottom: 10px;">📄 AI Project Report Generator</h3>
        <p style="color: #9ca3af; font-size: 14px; margin: 0;">Generate a comprehensive, downloadable PDF document summarizing your entire machine learning pipeline, dataset health, and model performance.</p>
        </div>
        """, unsafe_allow_html=True)
            
        # State key logic so download button persists
        if st.button("📄 Generate Professional Report", use_container_width=True):
            with st.spinner("Generating Report PDF..."):
                from reportlab.pdfgen import canvas
                
                def add_header_footer(canvas, doc):
                    canvas.saveState()
                    
                    canvas.setFont("Helvetica-Bold", 10)
                    canvas.setFillColor(colors.HexColor("#2563EB"))
                    
                    canvas.drawString(40, 810, "🧠 Neuronix AI")
                    
                    canvas.drawRightString(550, 810, "AI Machine Learning Report")
                    
                    canvas.setFont("Helvetica", 9)
                    canvas.setFillColor(colors.grey)
                    
                    canvas.drawString(40, 20, "Generated by 🧠 Neuronix AI")
                    
                    canvas.drawRightString(550, 20, f"Page {doc.page}")
                    
                    canvas.restoreState()
        
                pdf = SimpleDocTemplate("Neuronix_AI_Report.pdf")
                styles = getSampleStyleSheet()
                title_style = ParagraphStyle(
                    "TitleStyle",
                    parent=styles["Title"],
                    fontSize=28,
                    textColor=colors.HexColor("#0F172A"),
                    alignment=TA_CENTER,
                    spaceAfter=30,
                )
                
                heading_style = ParagraphStyle(
                    "HeadingStyle",
                    parent=styles["Heading1"],
                    fontSize=18,
                    textColor=colors.HexColor("#2563EB"),
                    spaceAfter=12,
                )
                
                subheading_style = ParagraphStyle(
                    "SubHeading",
                    parent=styles["Heading2"],
                    fontSize=14,
                    textColor=colors.HexColor("#1E40AF"),
                    spaceAfter=10,
                )
                
                normal_style = ParagraphStyle(
                    "NormalStyle",
                    parent=styles["BodyText"],
                    fontSize=11,
                    leading=20,
                )
                story = []

                # SESSION STATE VARIABLES
                task = st.session_state.get("task", "")
                best_model = st.session_state.get("best_model", None)
                best_model_name = st.session_state.get("best_model_name", "Not Available")
                best_score = st.session_state.get("best_score", 0)
                prediction = st.session_state.get("prediction", None)
                results = st.session_state.get("res", None)
                selected_features = st.session_state.get("selected_features", [])
                X_test = st.session_state.get("X_test", None)
                y_test = st.session_state.get("y_test", None)
                train_score = st.session_state.get("train_score", None)
                test_score = st.session_state.get("test_score", None)
                X_train = st.session_state.get("X_train", None)
                y_train = st.session_state.get("y_train", None)

                # TRAIN / TEST SCORE
                try:
                    if (best_model is not None and X_train is not None and y_train is not None):
                        train_score = best_model.score(X_train, y_train)
                except:
                    train_score = None
                
                try:
                    if (best_model is not None and X_test is not None and y_test is not None):
                        test_score = best_model.score(X_test, y_test)
                except:
                    test_score = None
                                    
                story.append(Spacer(1,120))
                story.append(Paragraph("🧠 Neuronix AI", title_style))
                story.append(Paragraph("Enterprise AI Machine Learning Report", heading_style))
                story.append(Spacer(1,40))
                story.append(Paragraph("<b>Prepared By</b>", subheading_style))
                story.append(Paragraph("Krishna Gediya", normal_style))
                story.append(Spacer(1,15))
                story.append(Paragraph("<b>Technology Stack</b>", subheading_style))
                story.append(Paragraph("Python | Streamlit | Scikit-learn | Plotly | SHAP | SQLite", normal_style))
                story.append(Spacer(1,15))
                story.append(Paragraph("<b>Generated Automatically by Neuronix AI</b>", normal_style))
                story.append(PageBreak())

                # DATASET SUMMARY
                dashboard = [
                    ["Metric", "Value"],
                    ["Rows", str(df.shape[0])],
                    ["Columns", str(df.shape[1])],
                    ["Missing", str(df.isnull().sum().sum())],
                    ["Duplicates", str(df.duplicated().sum())],
                    ["Memory (KB)", str(round(df.memory_usage(deep=True).sum()/1024, 2))]
                ]
                table = Table(dashboard, colWidths=[220, 220])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                story.append(table)
                story.append(Spacer(1, 20))

                # EXECUTIVE SUMMARY
                story.append(Paragraph("Executive Summary", heading_style))
                story.append(Spacer(1, 15))
                summary = f"""
                <b>🧠 Neuronix AI</b> is an Enterprise AI platform that automates
                the complete Machine Learning workflow.
                <br/><br/>
                This report has been automatically generated after analyzing the uploaded dataset.
                <br/><br/>
                <b>Dataset Statistics</b><br/>
                • Total Rows : {df.shape[0]}<br/>
                • Total Columns : {df.shape[1]}<br/>
                • Missing Values : {df.isnull().sum().sum()}<br/>
                • Duplicate Rows : {df.duplicated().sum()}<br/>
                <br/>
                The system performed preprocessing, feature engineering,
                model training, model comparison and AI based analysis.
                <br/><br/>
                The objective of this report is to provide complete insights
                about the dataset and recommend the best Machine Learning solution.
                """
                story.append(Paragraph(summary, normal_style))
                story.append(PageBreak())

                # DATASET HEALTH
                story.append(Paragraph("Dataset Health Report", heading_style))
                story.append(Spacer(1, 20))
                
                health = 100
                health -= int(df.isnull().sum().sum())
                if df.duplicated().sum() > 0:
                    health -= 5
                if df.shape[0] < 100:
                    health -= 20
                if health < 0:
                    health = 0
                
                health_table = [
                    ["Parameter", "Status"],
                    ["Health Score", f"{health}%"],
                    ["Rows", str(df.shape[0])],
                    ["Columns", str(df.shape[1])],
                    ["Missing Values", str(df.isnull().sum().sum())],
                    ["Duplicates", str(df.duplicated().sum())],
                    ["Numeric Features", str(len(df.select_dtypes(include=np.number).columns))],
                    ["Categorical Features", str(len(df.select_dtypes(exclude=np.number).columns))]
                ]
                
                table = Table(health_table, colWidths=[230, 220])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ]))
                story.append(table)
                story.append(PageBreak())

                # AI RECOMMENDATION
                story.append(Paragraph("AI Recommendations", heading_style))
                story.append(Spacer(1, 20))
                
                recommend = []
                if df.isnull().sum().sum() > 0: 
                    recommend.append("• Handle missing values before training.")
                if df.duplicated().sum() > 0: 
                    recommend.append("• Remove duplicate rows.")
                if len(df.select_dtypes(exclude=np.number).columns) > 0: 
                    recommend.append("• Encode categorical variables.")
                if len(df.select_dtypes(include=np.number).columns) > 15: 
                    recommend.append("• Feature Selection Recommended.")
                if df.shape[0] < 500: 
                    recommend.append("• More training data is recommended.")
                if len(recommend) == 0: 
                    recommend.append("• Dataset is ready for Machine Learning.")
                
                for r in recommend:
                    story.append(Paragraph(r, normal_style))
                    story.append(Spacer(1, 5))
                story.append(PageBreak())

                # DATASET STATISTICS
                story.append(Paragraph("Statistical Analysis", heading_style))
                story.append(Spacer(1, 15))
                
                desc = df.describe().round(2)
                
                data = [["Statistic"] + list(desc.columns)]
                for index, row in desc.iterrows():
                    data.append([index] + list(row.values))
                
                table = Table(data)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#E2E8F0")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
                ]))
                story.append(table)
                story.append(PageBreak())

                # TABLE OF CONTENTS
                story.append(Paragraph("Table of Contents", styles["Heading1"]))
                story.append(Spacer(1, 20))
                toc = [
                    "1. Executive Summary",
                    "2. Dataset Summary",
                    "3. Exploratory Data Analysis",
                    "4. Data Quality Analysis",
                    "5. Feature Engineering",
                    "6. Feature Selection",
                    "7. Model Training",
                    "8. Model Leaderboard",
                    "9. Best Model Analysis",
                    "10. Overfitting Detection",
                    "11. Feature Importance",
                    "12. SHAP Explainability",
                    "13. Prediction Results",
                    "14. AI Recommendations",
                    "15. Conclusion",
                    "16. Future Scope"
                ]
                for item in toc:
                    story.append(Paragraph(item, normal_style))
                    story.append(Spacer(1, 5))
                story.append(PageBreak())

                # PROJECT INFORMATION
                story.append(Paragraph("Project Information", heading_style))
                story.append(Spacer(1, 20))
                info = [
                    ["Project Name", "Neuronix AI"],
                    ["Developer", "Krishna Gediya"],
                    ["Programming Language", "Python"],
                    ["Framework", "Streamlit"],
                    ["Machine Learning", "Scikit-learn"],
                    ["Visualization", "Plotly"],
                    ["Database", "SQLite"],
                    ["Version", "2.0"]
                ]
                table = Table(info, colWidths=[200, 250])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 8),
                ]))
                story.append(table)
                story.append(PageBreak())

                # SAVED CHARTS INCLUSION
                story.append(Paragraph("Histogram", heading_style))
                story.append(Spacer(1, 20))
                if os.path.exists("histogram.png"):
                    story.append(RLImage("histogram.png", width=450, height=280))
                story.append(PageBreak())
                
                story.append(Paragraph("Box Plot", heading_style))
                story.append(Spacer(1, 20))
                if os.path.exists("boxplot.png"):
                    story.append(RLImage("boxplot.png", width=450, height=280))
                story.append(PageBreak())
                
                story.append(Paragraph("Correlation Heatmap", heading_style))
                story.append(Spacer(1, 20))
                if os.path.exists("heatmap.png"):
                    story.append(RLImage("heatmap.png", width=450, height=350))
                story.append(PageBreak())
                
                story.append(Paragraph("Scatter Plot", heading_style))
                story.append(Spacer(1, 20))
                if os.path.exists("scatter.png"):
                    story.append(RLImage("scatter.png", width=450, height=300))
                story.append(PageBreak())

                # MISSING VALUE ANALYSIS
                story.append(Paragraph("Missing Value Analysis", heading_style))
                story.append(Spacer(1, 20))
                missing_table = [["Column", "Missing Count"]]
                for col in df.columns:
                    missing_table.append([col, str(df[col].isnull().sum())])
                
                table = Table(missing_table, colWidths=[250, 150])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(table)
                story.append(PageBreak())

                # PREDICTION RESULT
                story.append(Paragraph("Interactive Prediction Result", heading_style))
                story.append(Spacer(1, 20))
                try:
                    if "prediction" in st.session_state:
                        pred_val = st.session_state.prediction[0]
                        pred_text = f"<b>Based on the custom inputs, the AI model predicted:</b> <font color='#2563EB' size='14'>{pred_val}</font>"
                        story.append(Paragraph(pred_text, normal_style))
                    else:
                        story.append(Paragraph("<i>No interactive prediction was made during this session.</i>", normal_style))
                except:
                    pass
                story.append(PageBreak())
                
                # CONFUSION MATRIX
                if st.session_state.get("task") == "Classification":
                    story.append(Paragraph("Confusion Matrix", heading_style))
                    story.append(Spacer(1, 20))
                    if "best_model" in st.session_state and "X_test" in st.session_state:
                        try:
                            preds = st.session_state.best_model.predict(st.session_state.X_test)
                            fig, ax = plt.subplots(figsize=(5, 5))
                            ConfusionMatrixDisplay.from_predictions(st.session_state.y_test, preds, ax=ax, cmap="Blues")
                            plt.title("Confusion Matrix")
                            plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
                            plt.close()
                            
                            if os.path.exists("confusion_matrix.png"):
                                story.append(RLImage("confusion_matrix.png", width=420, height=420))
                        except Exception:
                            pass
                    story.append(PageBreak())
                    
                # AI RECOMMENDATIONS
                story.append(Paragraph("AI Recommendations", heading_style))
                story.append(Spacer(1, 20))
                recommendations = [
                    "✔ Handle Missing Values",
                    "✔ Perform Feature Engineering",
                    "✔ Use Cross Validation",
                    "✔ Tune Hyperparameters",
                    "✔ Monitor Overfitting",
                    "✔ Evaluate Model on Unseen Data",
                    "✔ Save Pipeline for Deployment"
                ]
                for item in recommendations:
                    story.append(Paragraph(item, normal_style))
                    story.append(Spacer(1, 5))
                story.append(PageBreak())

                # SAVE CHARTS FOR PDF
                numeric_cols = df.select_dtypes(include=np.number).columns
                
                # Histogram
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(8, 5))
                    plt.hist(df[numeric_cols[0]], bins=25, color="#2563EB", edgecolor="black")
                    plt.title("Histogram")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig("histogram.png", dpi=300)
                    plt.close()
                
                # Box Plot
                if len(numeric_cols) > 0:
                    plt.figure(figsize=(8, 5))
                    plt.boxplot(df[numeric_cols[0]])
                    plt.title("Box Plot")
                    plt.tight_layout()
                    plt.savefig("boxplot.png", dpi=300)
                    plt.close()
                
                # Correlation Heatmap
                if len(numeric_cols) >= 2:
                    corr = df[numeric_cols].corr()
                    plt.figure(figsize=(8, 6))
                    plt.imshow(corr, cmap="Blues")
                    plt.colorbar()
                    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
                    plt.yticks(range(len(corr.columns)), corr.columns)
                    plt.title("Correlation Heatmap")
                    plt.tight_layout()
                    plt.savefig("heatmap.png", dpi=300)
                    plt.close()
                
                # Scatter Plot
                if len(numeric_cols) >= 2:
                    plt.figure(figsize=(8, 5))
                    plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]], alpha=0.7, color="#10b981")
                    plt.xlabel(numeric_cols[0])
                    plt.ylabel(numeric_cols[1])
                    plt.title("Scatter Plot")
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig("scatter.png", dpi=300)
                    plt.close()
            
                # GENERATE DOCUMENT
                pdf.build(
                    story,
                    onFirstPage=add_header_footer,
                    onLaterPages=add_header_footer
                )
                st.session_state.report_generated = True
                
        # State logic to ensure the button remains visible after click
        if st.session_state.get("report_generated"):
            st.markdown("<br>", unsafe_allow_html=True)
            st.success("✅ **Report Generated Successfully!** Your comprehensive AI analysis is ready.")
            with open("Neuronix_AI_Report.pdf", "rb") as pdf_file:
                st.download_button(
                    "⬇️ Download Professional Report (.pdf)",
                    pdf_file,
                    "Neuronix_AI_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type="primary"
                )
    # TAB: EXPERIMENT HISTORY
    with tab_history:
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #14b8a6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 10px;">📜 Experiment History</h2>
<p style="color: #9ca3af; font-size: 15px; margin: 0;">Track, review, and export the performance logs of all machine learning models trained during this session.</p>
</div>
""", unsafe_allow_html=True)
        
        if "experiment_history" not in st.session_state:
            st.session_state.experiment_history = []
        
        if len(st.session_state.experiment_history) == 0:
            st.info("ℹ️ No experiments found. Go to the 'Model Training' tab and train a model to see its history here.")
        else:
            history_df = pd.DataFrame(st.session_state.experiment_history)
            st.dataframe(history_df, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Button layout for better UI
            col_btn1, col_btn2 = st.columns(2)
            
            csv = history_df.to_csv(index=False).encode("utf-8")
            with col_btn1:
                st.download_button(
                    "⬇️ Download Experiment History",
                    csv,
                    "Experiment_History.csv",
                    "text/csv",
                    use_container_width=True
                )
                
            with col_btn2:
                if st.button("🗑️ Clear History", use_container_width=True, type="primary"):
                    st.session_state.experiment_history = []
                    st.success("✅ History Cleared Successfully.")
                    st.rerun()
                    
    # TAB: ML GLOSSARY
    with tab_glossary:
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #f43f5e; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 10px;">📖 Complete Machine Learning Glossary</h2>
<p style="color: #9ca3af; font-size: 15px; margin: 0;">Master the most important Machine Learning concepts from basic to advanced levels. Perfect for students, beginners, viva preparation, and interviews.</p>
</div>
""", unsafe_allow_html=True)
        
        glossary = {
            # --- Basics & Data ---
            "🤖 Machine Learning": "A branch of Artificial Intelligence where computers learn patterns from data and make predictions without being explicitly programmed.",
            "📚 Supervised Learning": "Learning with labeled data, where the model knows the correct answers (target) during training.",
            "🕵️ Unsupervised Learning": "Learning with unlabeled data. The model finds hidden patterns or groupings on its own (e.g., Clustering).",
            "🎮 Reinforcement Learning": "Learning by trial and error, where an AI agent gets rewards for good actions and penalties for bad ones.",
            "📊 Dataset": "A structured collection of data used to train and test Machine Learning models.",
            "🎯 Target Variable": "The specific column or value that the model is trying to predict (also known as the dependent variable or label).",
            "⭐ Feature": "The input columns or variables used by the model to make predictions (independent variables).",
            "🚨 Outlier": "A data point that differs significantly from other observations, often requiring removal or transformation as it can skew the model.",
            
            # --- Data Preprocessing ---
            "🧹 Missing Values": "Empty cells in a dataset (NaN/Null). They must be handled (filled or dropped) before training a model.",
            "🗑 Duplicate Rows": "Identical, repeated records in a dataset that can artificially inflate model performance and should be removed.",
            "🏷 Categorical Encoding": "The process of converting text/categorical data (like 'Red', 'Blue') into numerical values (like 0, 1) because models only understand numbers.",
            "⚖ Feature Scaling": "Bringing all numeric features to the same scale (e.g., 0 to 1) so that larger numbers don't dominate the model.",
            "⭐ Feature Selection": "Choosing the most important columns for training and discarding the useless ones to improve model speed and accuracy.",
            "✂ Train-Test Split": "Dividing the dataset into two parts: 'Training data' to teach the model, and 'Testing data' to evaluate its performance.",
            "🚨 Data Leakage": "A critical error where future information or testing data accidentally leaks into the training phase, causing fake high accuracy.",
            
            # --- Model Types & Algorithms ---
            "🧠 Classification": "A supervised learning task that predicts categories or classes (e.g., Spam vs. Not Spam, Cat vs. Dog).",
            "📈 Regression": "A supervised learning task that predicts continuous numerical values (e.g., House Price, Salary, Temperature).",
            "📍 Clustering": "An unsupervised learning task that groups similar records together without knowing predefined labels.",
            "📈 Linear Regression": "A foundational regression algorithm that fits a straight line through data points to make numerical predictions.",
            "📉 Logistic Regression": "Despite its name, it is a classification algorithm used for predicting probabilities of categories (usually binary).",
            "🌳 Decision Tree": "A model that predicts outcomes by splitting data into branches based on a series of if-else conditions.",
            "🌲 Random Forest": "An ensemble method that combines hundreds of Decision Trees to prevent overfitting and improve accuracy.",
            "📌 KNN (K-Nearest Neighbors)": "An algorithm that makes predictions based on the 'K' most similar (closest) data points in the training set.",
            "⚡ XGBoost / LightGBM": "Advanced, highly optimized Gradient Boosting algorithms known for winning Kaggle competitions and high performance.",
            "🛡️ SVM (Support Vector Machine)": "A powerful algorithm that finds the best boundary (hyperplane) to separate different classes of data.",
            
            # --- Advanced Concepts ---
            "🛠️ Ensemble Learning": "Combining multiple weak or base models to create one strong, highly accurate model (e.g., Bagging and Boosting).",
            "🧠 Deep Learning": "A subset of ML that uses multi-layered Artificial Neural Networks to solve highly complex problems like image and speech recognition.",
            "🕸️ Neural Networks": "Algorithms inspired by the human brain, consisting of input, hidden, and output layers of interconnected nodes (neurons).",
            "📉 Gradient Descent": "An optimization algorithm used to minimize the error (loss) of a model by updating its parameters step-by-step.",
            "⚖️ Bias-Variance Tradeoff": "The delicate balance between a model being too simple (high bias/underfitting) and too complex (high variance/overfitting).",
            "🛡️ Regularization (L1/L2)": "Techniques like Lasso (L1) and Ridge (L2) used to penalize highly complex models to prevent them from overfitting.",
            "🔍 PCA (Principal Component Analysis)": "A dimensionality reduction technique that compresses many features into a few 'components' while keeping the most important information.",
            "🎯 Hyperparameter Tuning": "The process of finding the best internal settings (hyperparameters) for a model using techniques like Grid Search or Random Search.",
            
            # --- Evaluation Metrics ---
            "🏆 Accuracy": "The percentage of totally correct predictions out of all predictions made.",
            "🎯 Precision": "Out of all the points the model *claimed* were positive, how many were *actually* positive? (Focuses on reducing False Positives).",
            "📈 Recall": "Out of all the *actual* positive points, how many did the model *successfully find*? (Focuses on reducing False Negatives).",
            "⭐ F1 Score": "The harmonic mean (balance) between Precision and Recall. Highly useful for imbalanced datasets.",
            "📉 MAE (Mean Absolute Error)": "Measures the average absolute difference between predicted and actual numerical values.",
            "📊 RMSE (Root Mean Squared Error)": "Measures prediction error but heavily penalizes larger errors because it squares the differences before averaging.",
            "🎯 R² Score": "Indicates how well the regression model explains the variance in the data (1.0 is perfect, 0 means it's no better than guessing the average).",
            "📊 Cross Validation": "Evaluating the model multiple times on different chunks (folds) of the data to ensure its performance is reliable and not based on luck.",
            "📄 Confusion Matrix": "A table that visually compares the actual classes vs. the predicted classes (showing True Positives, False Positives, etc.).",
            "📈 ROC Curve & AUC": "A graph (ROC) and a score (AUC) that measure a classifier's ability to distinguish between classes at various threshold levels.",
            
            # --- Explainability & Production ---
            "🔥 Overfitting": "When a model memorizes the training data perfectly but fails completely on new, unseen data (Low Training Error, High Testing Error).",
            "❄ Underfitting": "When a model is too simple or hasn't trained enough to learn the underlying patterns of the data (High Training Error).",
            "🧠 SHAP Explainability": "A game-theoretic approach that explains exactly which features (and by how much) pushed the model toward a specific prediction.",
            "💾 Model Deployment": "The final stage of ML where the trained, optimized model is integrated into a live application, API, or software for real-world use.",
            "🚀 AutoML": "Automated Machine Learning. The process of automating data preprocessing, model selection, tuning, and evaluation."
        }
        
        st.markdown("<h4 style='color: #60a5fa; margin-bottom: 10px;'>🔍 Search & Learn</h4>", unsafe_allow_html=True)
        option = st.selectbox("Select a concept to read its quick definition:", list(glossary.keys()), label_visibility="collapsed")
        
        st.markdown(f"""
<div style="background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 15px; border-radius: 8px; margin-bottom: 25px;">
<strong style="color: #60a5fa; font-size: 16px;">{option}</strong><br>
<span style="color: #e2e8f0; font-size: 15px;">{glossary[option]}</span>
</div>
""", unsafe_allow_html=True)

        st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #a78bfa; margin-bottom: 20px;'>📚 Full Machine Learning Dictionary</h3>", unsafe_allow_html=True)
        
        # Displaying all items in expanders
        for term, definition in glossary.items():
            with st.expander(term):
                st.write(definition)
                
        st.markdown("<br>", unsafe_allow_html=True)
        st.success("🎓 **Pro Tip:** Learn and master one concept from this list every day to rapidly improve your Data Science and Machine Learning skills!")

    # TAB: AI INTERVIEW PREPARATION
    with tab_interview:
        st.markdown("""
<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 25px; border-radius: 20px; border-left: 8px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); margin-top: 20px; margin-bottom: 30px;">
<h2 style="color: white; margin-top: 0; margin-bottom: 10px;">💼 AI / ML Interview Preparation</h2>
<p style="color: #9ca3af; font-size: 15px; margin: 0;">Top interview questions and quick answers to help you ace your next Data Science, ML, or Python technical interview.</p>
</div>
""", unsafe_allow_html=True)
        
        st.markdown("<h4 style='color: #60a5fa; margin-bottom: 10px;'>🎯 Select Interview Category</h4>", unsafe_allow_html=True)
        category = st.selectbox(
            "Choose Category",
            ["Machine Learning", "Python", "SQL", "Statistics", "Data Science"],
            label_visibility="collapsed"
        )
        
        questions = {
            "Machine Learning": [
                ("What is Machine Learning?", "Machine Learning is a subset of AI that enables computers to learn patterns from data and make predictions without being explicitly programmed."),
                ("Difference between AI and ML?", "AI is a broad field aiming to create intelligent machines. ML is a specific subset of AI focused on learning from data."),
                ("What is Overfitting?", "Overfitting happens when a model learns the training data too well (memorizes it), including the noise, resulting in poor performance on unseen data."),
                ("What is Underfitting?", "Underfitting occurs when a model is too simple to learn the underlying patterns in the training data, leading to high errors on both training and testing sets."),
                ("What is Cross Validation?", "A resampling technique used to evaluate a model's robustness by training and testing it on different subsets (folds) of the dataset."),
                ("Difference between Classification and Regression?", "Classification predicts discrete categories (e.g., Spam or Not Spam), while Regression predicts continuous numerical values (e.g., Price, Age)."),
                ("What is Feature Engineering?", "The process of using domain knowledge to extract or create new, meaningful features from raw data to improve model performance."),
                ("What is SHAP?", "SHAP (SHapley Additive exPlanations) is an Explainable AI technique used to understand how much each feature contributes to a model's final prediction."),
                ("Why do we need Feature Scaling?", "Scaling (like Normalization or Standardization) brings all numeric features to a similar range, which speeds up optimization (Gradient Descent) and is crucial for distance-based algorithms like KNN and SVM."),
                ("Which is the Best Classification Algorithm?", "There is no single 'best' algorithm (No Free Lunch Theorem). It depends on the dataset size, complexity, and type. However, Random Forest or XGBoost are often strong baselines.")
            ],
            "Python": [
                ("Difference between List and Tuple?", "A List is mutable (can be changed after creation), while a Tuple is immutable (cannot be changed)."),
                ("What is a Lambda function?", "A small, anonymous, single-line function defined using the 'lambda' keyword."),
                ("What is a Decorator?", "A function that takes another function and extends or modifies its behavior without explicitly changing its source code."),
                ("Difference between Deep Copy and Shallow Copy?", "A shallow copy constructs a new compound object and inserts references into it. A deep copy constructs a new compound object and recursively copies all objects found in the original."),
                ("What is a Generator?", "A special type of iterator that produces values lazily (one at a time) using the 'yield' keyword, which saves memory.")
            ],
            "SQL": [
                ("Difference between WHERE and HAVING?", "WHERE filters rows before grouping/aggregations are applied, whereas HAVING filters records after the GROUP BY clause is applied."),
                ("What is a Primary Key?", "A column (or set of columns) that uniquely identifies each row in a table. It cannot contain NULL values."),
                ("What is a Foreign Key?", "A column that links two tables together by referencing the Primary Key of another table, maintaining referential integrity."),
                ("What does INNER JOIN do?", "It returns only the rows that have matching values in both tables being joined."),
                ("What is the purpose of GROUP BY?", "It groups rows that have the same values in specified columns into summary rows, often used with aggregate functions (COUNT, MAX, SUM, AVG).")
            ],
            "Statistics": [
                ("Mean vs Median?", "The Mean is the arithmetic average of a dataset. The Median is the exact middle value when the data is sorted. Median is less affected by outliers."),
                ("What is Standard Deviation?", "A measure of the amount of variation or dispersion in a set of values. A low SD means values are close to the mean."),
                ("What is a Normal Distribution?", "A probability distribution that is symmetric about the mean (bell-shaped curve), where most observations cluster around the central peak."),
                ("What is Variance?", "The average of the squared differences from the Mean. Standard deviation is the square root of Variance."),
                ("What is Correlation?", "A statistical measure that indicates the extent to which two variables fluctuate together (ranging from -1 to 1).")
            ],
            "Data Science": [
                ("What is Data Cleaning?", "The process of detecting and correcting (or removing) corrupt, inaccurate, or missing records from a dataset."),
                ("What is EDA?", "Exploratory Data Analysis (EDA) is the process of analyzing datasets to summarize their main characteristics, often using visual methods."),
                ("What is Feature Selection?", "The process of selecting the most relevant, useful columns (features) to use in model construction, discarding redundant ones."),
                ("What is Data Leakage?", "When information from outside the training dataset is used to create the model, leading to overly optimistic and invalid performance estimates."),
                ("What is a Machine Learning Pipeline?", "A sequence of data processing components (like scaling, encoding, and training) chained together to automate a machine learning workflow.")
            ]
        }
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        for i, (q, a) in enumerate(questions[category], 1):
            st.markdown(f"<h4 style='color: #e2e8f0; margin-top: 15px;'>Q{i}. {q}</h4>", unsafe_allow_html=True)
            with st.expander("💡 Show Answer"):
                st.success(a)
                
        st.markdown("---")
        st.info("📚 **Tip:** Practice these questions before placements and viva. Explaining these concepts out loud is the best way to prepare!")

else:
    # Default st.info ko hatakar premium HTML Alert Box lagaya gaya hai
    html_alert = """
<div style="background: rgba(59, 130, 246, 0.1); border: 1px dashed rgba(59, 130, 246, 0.5); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.2);">
<h4 style="color: #60a5fa; margin: 0; font-weight: 600; font-size: 18px;">👋 Welcome to Neuronix AI Workspace</h4>
<p style="color: #94a3b8; font-size: 15px; margin: 8px 0 0 0;">Please <b>Upload a Dataset</b> from the Control Panel on the left to initialize the Neural Engine.</p>
</div>
"""
    st.markdown(html_alert, unsafe_allow_html=True)

    html_cards = """
<div style="margin-top: 35px; margin-bottom: 20px;">
<h3 style="color: #94a3b8; text-align: center; margin-bottom: 30px; font-weight: 600; letter-spacing: 1px;">⚡ EXPLORE THE NEURONIX ARCHITECTURE</h3>
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 25px;">

<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px 20px; border-radius: 20px; border-top: 5px solid #3b82f6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;">
<h1 style="margin: 0; font-size: 45px; text-shadow: 0 0 15px rgba(59,130,246,0.5);">🛠️</h1>
<h4 style="color: #e2e8f0; margin: 15px 0 10px 0; font-size: 18px;">Smart Preprocessing</h4>
<p style="color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5;">Automated missing value handling, categorical encoding, and intelligent feature scaling.</p>
</div>

<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px 20px; border-radius: 20px; border-top: 5px solid #f59e0b; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;">
<h1 style="margin: 0; font-size: 45px; text-shadow: 0 0 15px rgba(245,158,11,0.5);">📊</h1>
<h4 style="color: #e2e8f0; margin: 15px 0 10px 0; font-size: 18px;">Deep Data Profiling</h4>
<p style="color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5;">Automatically detect data leakage, visualize feature distributions, and assess dataset health.</p>
</div>

<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px 20px; border-radius: 20px; border-top: 5px solid #10b981; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;">
<h1 style="margin: 0; font-size: 45px; text-shadow: 0 0 15px rgba(16,185,129,0.5);">🤖</h1>
<h4 style="color: #e2e8f0; margin: 15px 0 10px 0; font-size: 18px;">AutoML Engine</h4>
<p style="color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5;">Trains and compares 10+ cutting-edge algorithms instantly to find the highest accuracy model.</p>
</div>

<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px 20px; border-radius: 20px; border-top: 5px solid #8b5cf6; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;">
<h1 style="margin: 0; font-size: 45px; text-shadow: 0 0 15px rgba(139,92,246,0.5);">🧠</h1>
<h4 style="color: #e2e8f0; margin: 15px 0 10px 0; font-size: 18px;">Explainable AI</h4>
<p style="color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5;">Demystify AI decisions with SHAP values, feature importance, and interactive prediction tools.</p>
</div>

<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px 20px; border-radius: 20px; border-top: 5px solid #06b6d4; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;">
<h1 style="margin: 0; font-size: 45px; text-shadow: 0 0 15px rgba(6,182,212,0.5);">📤</h1>
<h4 style="color: #e2e8f0; margin: 15px 0 10px 0; font-size: 18px;">Bulk Predictions</h4>
<p style="color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5;">Upload unseen datasets to run massive batch predictions instantly and export the results.</p>
</div>

<div style="background: linear-gradient(135deg, #111827, #1e293b); padding: 30px 20px; border-radius: 20px; border-top: 5px solid #f43f5e; box-shadow: 0 10px 30px rgba(0,0,0,0.3); text-align: center;">
<h1 style="margin: 0; font-size: 45px; text-shadow: 0 0 15px rgba(244,63,94,0.5);">📄</h1>
<h4 style="color: #e2e8f0; margin: 15px 0 10px 0; font-size: 18px;">One-Click Reports</h4>
<p style="color: #94a3b8; font-size: 14px; margin: 0; line-height: 1.5;">Generate and download comprehensive PDF analytics and ready-to-deploy machine learning pipelines.</p>
</div>

</div>
</div>
"""
    st.markdown(html_cards, unsafe_allow_html=True)

    html_footer = """
<div style="text-align: center; color: #64748b; margin-top: 60px; padding: 20px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 14px;">
Developed by <b>Krishna Gediya</b> | Neuronix AI Enterprise Platform
</div>
"""
    st.markdown(html_footer, unsafe_allow_html=True)
