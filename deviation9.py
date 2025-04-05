import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO
import datetime

# Initialiser l'état de session pour le suivi de l'entraînement des modèles
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model_azimuth' not in st.session_state:
    st.session_state.model_azimuth = None
if 'model_inclinaison' not in st.session_state:
    st.session_state.model_inclinaison = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None
if 'columns_mapped' not in st.session_state:
    st.session_state.columns_mapped = False
if 'augmented_df' not in st.session_state:
    st.session_state.augmented_df = None
if 'use_augmented_data' not in st.session_state:
    st.session_state.use_augmented_data = False

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de Déviation des Forages Miniers",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un look moderne
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
    
    /* Couleurs personnalisées */
    :root {
        --primary: #3563E9;
        --primary-light: #4F8BF9;
        --secondary: #1A2C55;
        --accent: #FF4B4B;
        --background: #F5F7FB;
        --text: #202939;
        --light-text: #64748B;
        --card: white;
        --success: #0CCE6B;
        --warning: #FFC107;
        --danger: #FF4B4B;
        --info: #3ABFF8;
    }
    
    /* Corps du document */
    .main {
        background-color: var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
        padding: 1.5rem;
    }
    
    /* En-têtes */
    h1, h2, h3, h4, h5, h6 {
        color: var(--secondary);
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    h1 {
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #e0e4ec;
        text-align: center;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e0e4ec;
        text-align: center;
    }
    
    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        color: #1e3a8a;
    }
    
    /* Cards */
    .stDataFrame, .css-1r6slb0, div[data-testid="stBlock"] {
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        margin-bottom: 1.5rem;
        border: 1px solid #f0f0f5;
    }
    
    /* Widgets */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(53, 99, 233, 0.25);
    }
    
    .stButton>button:hover {
        background-color: #2952D6;
        box-shadow: 0 4px 12px rgba(53, 99, 233, 0.35);
        transform: translateY(-1px);
    }

    .stSidebar .stButton>button {
        width: 100%;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #EEF2F9;
        padding: 10px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 8px;
        padding: 0 24px;
        font-weight: 500;
        background-color: #F5F7FB !important;
        color: #4A5568 !important;
        font-family: 'Poppins', sans-serif;
        font-size: 0.95rem;
        letter-spacing: 0.01em;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3563E9 !important;
        color: white !important;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1A2C55;
        border-radius: 0 24px 24px 0;
        padding-top: 1.5rem;
    }
    
    [data-testid="stSidebar"] header {
        background-color: #1A2C55;
    }
    
    .css-1d391kg, .css-1wrcr25 {
        background-color: #1A2C55;
    }
    
    .css-1d391kg .css-163ttbj, .css-1wrcr25 .css-163ttbj {
        color: white;
    }
    
    .css-1d391kg label, .css-1wrcr25 label {
        color: #E2E8F0;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Radio buttons in sidebar */
    .stRadio > div {
        background-color: #253662;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) inset;
    }
    
    .stRadio label {
        color: white !important;
        font-weight: 500;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Selectbox in sidebar */
    .stSelectbox > div > div {
        background-color: #253662;
        border-radius: 8px;
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: #253662;
        border-radius: 8px;
        color: white;
    }
    
    /* Slider in sidebar */
    .stSlider > div > div {
        color: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--primary) !important;
        font-family: 'Poppins', sans-serif;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        color: #4A5568;
    }
    
    /* Author Banner */
    .author-banner {
        background: linear-gradient(135deg, #1A2C55 0%, #2D4584 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        position: relative;
        overflow: hidden;
    }
    
    .author-banner::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAwIiBoZWlnaHQ9IjYwMCIgdmlld0JveD0iMCAwIDYwMCA2MDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxnIG9wYWNpdHk9IjAuMTUiPgo8cGF0aCBkPSJNNTk2LjA4NSA0MTUuNTg5QzU5Mi40NTMgNDA2LjIzNCA1NzguODM4IDM5My43ODkgNTU5LjU1NSAzOTAuMTU2QzU0MC4yNzMgMzg2LjUyMyA1MTkuODIgMzkyLjYxOCA1MTMuNzI0IDQwNi41MTJDNTEwLjYzNiA0MjMuNjI3IDUxNS45MTEgNDM1LjE3OCA1MzUuMTk0IDQzOC44MTFDNTYwLjEwMSA0NDMuNTM5IDU5MS4zNTYgNDMzLjI0OSA1OTYuMDg1IDQxNS41ODlaIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIvPgo8cGF0aCBkPSJNMTQyLjg5MiA0NjkuOTE1QzEzOC4xNjMgNDkwLjU3NSAxNDMuNjE0IDUxNi45NDcgMTUyLjA4NiA1MjguMjIxQzE2MC41NTggNTM5LjQ5NSAxNzIuODMxIDUzNS4xMzUgMTc4LjY1NCA1MjcuNDg2QzE4NC40NzggNTE5LjgzNiAxODAuODQ1IDUwNS45NDIgMTczLjM3NCA0OTQuNjY5QzE2MC42OSA0NzUuNjA5IDE0Ny43OTcgNDQ5LjI1NSAxNDIuODkyIDQ2OS45MTVaIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIvPgo8cGF0aCBkPSJNMTI2LjUzOSAzNzguODc5QzEyMS45OTggMzk1LjgxMSAxMzQuOTg3IDQwNC45MDggMTUxLjQwNyA0MDguMDg4QzE2Ny44MjcgNDExLjI2NyAxODcuNjczIDQwNC44MTkgMTg2LjE3MSAzODguNTI2QzE4NC42NjkgMzcyLjIzMiAxNzAuMDU0IDM2MS4xNDcgMTUzLjYzNCAzNTcuOTY4QzEzNy4yMTMgMzU0Ljc4OCAxMzEuNDQzIDM2MC4zMzggMTI2LjUzOSAzNzguODc5WiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjUiLz4KPHBhdGggZD0iTTE5NS45MjcgMTQzLjU2NUMyMDAuNDcgMTM1LjAwNiAyMDMuMTEgMTMwLjYwNSAxOTMuODE1IDEyNC4xODdDMTc4LjUyMSAxMTMuODM2IDE1Ny44ODMgMTE3LjY0MSAxNDMuMzAxIDEyNy4yMTlDMTI4LjcxOSAxMzYuNzk3IDEyNC43MyAxNTEuMDYyIDEzMC41NTMgMTYxLjYyNUMxMzYuMzc3IDE3Mi4xODcgMTUwLjc2NCAxNzMuNjkgMTY1LjM0NSAxNjQuMTEyQzE3OS45MjcgMTU0LjUzNCAxOTEuMzg0IDE1Mi4xMjQgMTk1LjkyNyAxNDMuNTY1WiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjUiLz4KPHBhdGggZD0iTTIzOC40MzkgNDM0LjkxM0MyNTMuMTkgNDM4LjM2NCAyNzEuNzAzIDQzNC4wODggMjgxLjk4MSA0MjQuMTc1QzI5Mi4yNTkgNDE0LjI2MSAyOTIuOTk0IDQwMC4xODYgMjg0LjM0NSAzOTQuMzYzQzI3NS42OTcgMzg4LjUzOSAyNjAuODY5IDM5Mi4wODggMjUwLjU5MSA0MDIuMDAyQzI0MC4zMTMgNDExLjkxNiAyMjMuNjg3IDQzMS40NjMgMjM4LjQzOSA0MzQuOTEzWiIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyLjUiLz4KPHBhdGggZD0iTTQzMS4xNjcgMzY0LjIyMUM0MjcuMTY3IDM1MC4xNDYgNDA2LjQwOCAzNDQuMzIyIDM4OC45OTMgMzU0Ljk3MkMzNzEuNTc5IDM2NS42MiAzNjMuNDg2IDM4Ny4wNjYgMzcyLjg1NiAzOTUuNjUzQzM4Mi4yMjYgNDA0LjI0IDQwMS4xNDcgMzk1LjgzOSA0MTguNTYxIDM4NS4xOEM0MzUuOTc1IDM3NC41MzIgNDM1LjE2OCAzNzguMjk1IDQzMS4xNjcgMzY0LjIyMVoiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMi41Ii8+CjxwYXRoIGQ9Ik00NDkuNzQ2IDIyMC41NUM0NTMuNDQgMjA3Ljc3MyA0NTIuNjI0IDIwMy40ODIgNDQwLjM1NSAxOTcuMjlDNDIyLjQ5IDE4OC4wODMgMzk5LjI2MSAxODQuMDg3IDM4My41MTggMTkzLjYzNEMzNjcuNzc1IDIwMy4xODIgMzYyLjE0NiAyMTcuNzY2IDM2OS43MzUgMjI2Ljk3NEMzNzcuMzIzIDIzNi4xODIgMzk0LjgyNSAyMzIuOTkgNDEwLjU2NyAyMjMuNDQyQzQyNi4zMSAyMTMuODk0IDQ0Ni4wNTIgMjMzLjMyNyA0NDkuNzQ2IDIyMC41NVoiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMi41Ii8+CjxwYXRoIGQ9Ik0yNTUuNDc5IDE1Ni42MDJDMjU5LjIyNCAxNjcuNDQgMjcxLjQ5OSAxNzQuNTc3IDI4NC4xOTQgMTcyLjMyOUMyOTYuODg5IDE3MC4wOCAzMDYuOTE3IDE1OS4yNCAzMDUuMzUyIDE0OC4yMTJDMzAzLjc4NyAxMzcuMTgzIDI5Mi4wNzYgMTI5LjQ0MiAyNzkuMzgxIDEzMS42OTFDMjY2LjY4NiAxMzMuOTM5IDI1MS43MzMgMTQ1Ljc2MyAyNTUuNDc5IDE1Ni42MDJaIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIuNSIvPgo8L2c+Cjwvc3ZnPgo=') no-repeat center center;
        background-size: cover;
        opacity: 0.3;
        z-index: 0;
    }
    
    .author-info {
        display: flex;
        flex-direction: column;
        position: relative;
        z-index: 1;
    }
    
    .author-name {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .author-title {
        font-size: 1rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    .app-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
        font-family: 'Poppins', sans-serif;
    }
    
    .app-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
        text-align: center;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-trained {
        background-color: var(--success);
        box-shadow: 0 0 8px var(--success);
    }
    
    .status-untrained {
        background-color: var(--accent);
        box-shadow: 0 0 8px var(--accent);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #EBF4FF;
        border-left: 4px solid var(--primary);
        padding: 1.25rem;
        border-radius: 8px;
        margin-bottom: 1.25rem;
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: var(--primary);
        height: 8px;
        border-radius: 4px;
    }
    
    .stProgress > div > div {
        background-color: #E2E8F0;
        height: 8px;
        border-radius: 4px;
    }
    
    /* Plot styling */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        background-color: white;
        padding: 1.25rem;
        border: 1px solid #f0f0f5;
    }
    
    /* Performance Card Styling */
    .performance-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 14px;
        padding: 2.5rem;
        box-shadow: 0 12px 28px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.3);
    }
    
    .performance-card.excellent {
        background: linear-gradient(135deg, #0BA360 0%, #3CBA92 100%);
    }
    
    .performance-card.good {
        background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%);
    }
    
    .performance-card.moderate {
        background: linear-gradient(135deg, #FF8008 0%, #FFC837 100%);
    }
    
    .performance-card.limited {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .performance-card h4 {
        color: white;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
        font-family: 'Poppins', sans-serif;
    }
    
    .performance-card p {
        color: white;
        font-size: 1.15rem;
        line-height: 1.7;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
        font-family: 'DM Sans', sans-serif;
    }
    
    .performance-card .score-badge {
        position: absolute;
        top: 25px;
        right: 25px;
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 30px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        font-size: 1.3rem;
        color: #333;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        font-family: 'Poppins', sans-serif;
    }
    
    .performance-card .augment-note {
        margin-top: 1.75rem;
        padding-top: 1.25rem;
        border-top: 1px solid rgba(255, 255, 255, 0.4);
        font-style: italic;
        color: rgba(255, 255, 255, 0.95);
    }
    
    /* Main navigation tabs */
    [data-testid="stHorizontalBlock"] [data-baseweb="tab-list"] {
        display: flex;
        justify-content: center;
        gap: 12px;
        border-bottom: none;
        padding-bottom: 12px;
        background-color: transparent;
    }
    
    [data-testid="stHorizontalBlock"] [data-baseweb="tab"] {
        background-color: #EEF2F9 !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        min-width: 140px;
        text-align: center;
        border: none;
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        color: #4A5568 !important;
        transition: all 0.2s ease;
    }
    
    [data-testid="stHorizontalBlock"] [data-baseweb="tab"][aria-selected="true"] {
        background-color: #3563E9 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(53, 99, 233, 0.25);
    }
    
    [data-testid="stHorizontalBlock"] [data-baseweb="tab-highlight"] {
        display: none;
    }
    
    /* Center main app title */
    h1.main-title {
        text-align: center;
        font-size: 2.5rem;
        margin: 2rem auto;
        padding-bottom: 1.25rem;
        max-width: 800px;
        border-bottom: 2px solid #e0e4ec;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
    }
    
    /* Data card styling */
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        border: 1px solid #f0f0f5;
        height: 100%;
    }
    
    .metric-card h4 {
        color: #3E4C59;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #64748B;
        font-size: 0.9rem;
    }
    
    /* Add footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: #64748B;
        font-size: 0.9rem;
        margin-top: 3rem;
        border-top: 1px solid #E2E8F0;
    }
    
    /* Improve form inputs */
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 1px solid #E2E8F0;
        padding: 0.75rem 1rem;
        font-family: 'DM Sans', sans-serif;
    }
    
    .stSelectbox [data-baseweb="select"] {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
    }
    
    .stSelectbox [data-baseweb="select"]:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(53, 99, 233, 0.2);
    }
    
    /* Download button */
    .stDownloadButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        font-family: 'Poppins', sans-serif;
        letter-spacing: 0.02em;
        box-shadow: 0 2px 8px rgba(76, 175, 80, 0.25);
        width: 100%;
    }
    
    .stDownloadButton button:hover {
        background-color: #3d8b40;
        box-shadow: 0 4px 12px rgba(76, 175, 80, 0.35);
        transform: translateY(-1px);
    }
    
    /* Author signature */
    .author-signature {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .author-avatar {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background-color: #3563E9;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 12px;
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .author-details {
        display: flex;
        flex-direction: column;
    }
    
    .author-name-small {
        font-weight: 600;
        font-size: 1rem;
        color: #1A2C55;
    }
    
    .author-title-small {
        font-size: 0.85rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)

# Titre et auteur
st.markdown("""
<div class="author-banner">
    <div style="flex: 2; text-align: center;">
        <div class="app-title">Prédiction de Déviation des Forages Miniers</div>
        <div class="app-subtitle">Application de machine learning pour anticiper les déviations</div>
    </div>
    <div style="flex: 1; text-align: right;" class="author-info">
        <span class="author-name">Didier Ouedraogo, P.Geo.</span>
        <span class="author-title">Géologue & Data Scientist</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Fonction pour créer un icône
def get_icon_html(icon_name, color="white", size=24):
    return f'<i class="material-icons" style="color: {color}; font-size: {size}px;">{icon_name}</i>'

# Fonction pour charger les données
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

# Fonction pour l'augmentation des données
def augment_data(df, num_augmented_samples=500, noise_level=0.1, categorical_variation=True):
    """
    Génère des données synthétiques en ajoutant du bruit aléatoire aux données existantes
    et en variant les catégories pour les variables catégorielles.
    
    Args:
        df: DataFrame contenant les données d'origine
        num_augmented_samples: Nombre d'échantillons à générer
        noise_level: Niveau de bruit à ajouter aux variables numériques (proportion de l'écart-type)
        categorical_variation: Si True, varie les catégories pour les variables catégorielles
        
    Returns:
        DataFrame avec les données d'origine et les données augmentées
    """
    # Créer une copie des données d'origine
    augmented_df = df.copy()
    
    # Séparer les colonnes numériques et catégorielles
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Calculer les écarts-types pour les colonnes numériques
    std_devs = df[numeric_cols].std()
    
    # Générer des données augmentées
    for _ in range(num_augmented_samples):
        # Sélectionner un échantillon aléatoire comme base
        sample_idx = np.random.randint(0, len(df))
        new_sample = df.iloc[sample_idx].copy()
        
        # Ajouter du bruit aux variables numériques
        for col in numeric_cols:
            noise = np.random.normal(0, std_devs[col] * noise_level)
            new_sample[col] += noise
        
        # Varier les catégories pour les variables catégorielles
        if categorical_variation and categorical_cols:
            # Pour chaque colonne catégorielle, il y a une chance de changer la valeur
            for col in categorical_cols:
                if np.random.random() < 0.3:  # 30% de chance de changer la catégorie
                    unique_values = df[col].unique()
                    new_value = np.random.choice(unique_values)
                    new_sample[col] = new_value
        
        # Ajouter l'échantillon augmenté au DataFrame
        augmented_df = pd.concat([augmented_df, pd.DataFrame([new_sample])], ignore_index=True)
    
    return augmented_df

# Sidebar pour les options
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem; justify-content: center;">
        <h3 style="margin: 0; color: white; text-align: center; width: 100%; font-size: 1.5rem; font-family: 'Poppins', sans-serif;">Configuration</h3>
    </div>
    
    <div class="author-signature" style="margin-bottom: 1.5rem;">
        <div class="author-avatar">DO</div>
        <div class="author-details">
            <div class="author-name-small" style="color: white;">Didier Ouedraogo</div>
            <div class="author-title-small" style="color: #B2BDCC;">Géologue & Data Scientist</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Statut du modèle
    model_status = "trained" if st.session_state.model_trained else "untrained"
    columns_status = "mapped" if st.session_state.columns_mapped else "unmapped"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 1.5rem; background-color: #253662; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);">
        <span class="status-indicator status-{model_status}"></span>
        <span style="color: white; font-size: 0.95rem; font-family: 'DM Sans', sans-serif;">Statut du modèle: {'Entraîné' if st.session_state.model_trained else 'Non entraîné'}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Option pour uploader les données ou utiliser des données de démonstration
    st.markdown('<p style="color: #E2E8F0; font-weight: 600; margin-bottom: 0.75rem; font-family: \'Poppins\', sans-serif;">Source des données</p>', unsafe_allow_html=True)
    data_option = st.radio(
        "",
        ["Charger mes données", "Utiliser données démo"],
        label_visibility="collapsed"
    )
    
    if data_option == "Charger mes données":
        uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
        
        if uploaded_file is not None and st.session_state.raw_df is None:
            # Charger les données brutes
            st.session_state.raw_df = load_data(uploaded_file)
            st.session_state.columns_mapped = False
    
    # Séparateur visuel
    st.markdown('<hr style="margin: 1.75rem 0; border-color: #3D4A6A; opacity: 0.6;">', unsafe_allow_html=True)
    
    # Sélection du modèle
    st.markdown('<p style="color: #E2E8F0; font-weight: 600; margin-bottom: 0.75rem; font-family: \'Poppins\', sans-serif;">Modèle de machine learning</p>', unsafe_allow_html=True)
    model_option = st.selectbox(
        "",
        ["Random Forest", "SVM", "Régression Linéaire", "Réseau de Neurones"],
        label_visibility="collapsed"
    )
    
    # Option d'augmentation des données
    st.markdown('<p style="color: #E2E8F0; font-weight: 600; margin-bottom: 0.75rem; margin-top: 1.75rem; font-family: \'Poppins\', sans-serif;">Augmentation des données</p>', unsafe_allow_html=True)
    
    use_augmentation = st.checkbox("Utiliser l'augmentation des données", help="Génère des échantillons synthétiques pour améliorer l'apprentissage")
    
    if use_augmentation:
        st.session_state.use_augmented_data = True
        aug_samples = st.slider("Échantillons supplémentaires", min_value=100, max_value=1000, value=500, step=100)
        noise_level = st.slider("Niveau de variation", min_value=0.05, max_value=0.3, value=0.1, step=0.05)
    else:
        st.session_state.use_augmented_data = False
    
    # Bouton d'entraînement
    st.markdown('<div style="margin-top: 1.75rem;"></div>', unsafe_allow_html=True)
    train_button = st.button("Entraîner le modèle")

# Initialisation des données
df = None

if data_option == "Charger mes données" and st.session_state.raw_df is not None and not st.session_state.columns_mapped:
    st.markdown("<h2>Mappage des colonnes</h2>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        <b>Mappage requis</b>: Veuillez associer les colonnes de votre fichier CSV aux colonnes attendues par l'application.
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher un aperçu des données brutes
    st.markdown("<h3>Aperçu de vos données</h3>", unsafe_allow_html=True)
    st.dataframe(st.session_state.raw_df.head(), use_container_width=True)
    
    # Colonnes requises par l'application - Ajout de "company"
    required_columns = {
        'profondeur_finale': 'Profondeur finale du forage (mètres)',
        'azimuth_initial': 'Azimuth initial du forage (degrés)',
        'inclinaison_initiale': 'Inclinaison initiale du forage (degrés)',
        'lithologie': 'Type de roche traversée',
        'vitesse_rotation': 'Vitesse de rotation de la tige (tr/min)',
        'company': 'Entreprise de forage',
        'deviation_azimuth': 'Déviation mesurée en azimuth (degrés)',
        'deviation_inclinaison': 'Déviation mesurée en inclinaison (degrés)'
    }
    
    st.markdown("<h3>Associer les colonnes</h3>", unsafe_allow_html=True)
    st.markdown("""
    <p style="margin-bottom: 1.5rem;">Pour chaque paramètre requis, sélectionnez la colonne correspondante dans votre fichier CSV.</p>
    """, unsafe_allow_html=True)
    
    # Créer un dictionnaire pour stocker les mappages
    column_mapping = {}
    
    # Créer une liste des colonnes disponibles dans le CSV
    available_columns = st.session_state.raw_df.columns.tolist()
    
    # Ajouter une option "Non disponible" pour les colonnes facultatives
    available_columns_with_na = ['Non disponible'] + available_columns
    
    # Créer des sélecteurs pour chaque colonne requise
    col1, col2 = st.columns(2)
    
    with col1:
        for i, (required_col, description) in enumerate(list(required_columns.items())[:4]):
            # Suggérer une correspondance basée sur des mots-clés
            suggested_index = 0  # Par défaut "Non disponible"
            for j, col in enumerate(available_columns):
                if required_col.lower() in col.lower() or any(word in col.lower() for word in required_col.split('_')):
                    suggested_index = j + 1  # +1 car nous avons ajouté "Non disponible" en première position
                    break
            
            column_mapping[required_col] = st.selectbox(
                f"{description}",
                available_columns_with_na,
                index=suggested_index,
                help=f"Sélectionnez la colonne de votre CSV qui correspond à '{required_col}'"
            )
    
    with col2:
        for i, (required_col, description) in enumerate(list(required_columns.items())[4:]):
            # Suggérer une correspondance basée sur des mots-clés
            suggested_index = 0  # Par défaut "Non disponible"
            for j, col in enumerate(available_columns):
                if required_col.lower() in col.lower() or any(word in col.lower() for word in required_col.split('_')):
                    suggested_index = j + 1  # +1 car nous avons ajouté "Non disponible" en première position
                    break
            
            column_mapping[required_col] = st.selectbox(
                f"{description}",
                available_columns_with_na,
                index=suggested_index,
                help=f"Sélectionnez la colonne de votre CSV qui correspond à '{required_col}'"
            )
    
    # Vérifier si toutes les colonnes obligatoires sont mappées
    missing_required = [col for col, mapped in column_mapping.items() 
                       if mapped == 'Non disponible' and col not in ['lithologie', 'company']]
    
    if len(missing_required) > 0:
        st.warning(f"⚠️ Certaines colonnes obligatoires n'ont pas été mappées: {', '.join(missing_required)}")
        can_proceed = False
    else:
        can_proceed = True
    
    # Bouton pour valider le mappage
    mapping_col1, mapping_col2, mapping_col3 = st.columns([1, 2, 1])
    with mapping_col2:
        if st.button("Valider le mappage", disabled=not can_proceed, use_container_width=True):
            # Créer un nouveau DataFrame avec les colonnes mappées
            mapped_df = pd.DataFrame()
            
            for required_col, source_col in column_mapping.items():
                if source_col != 'Non disponible':
                    mapped_df[required_col] = st.session_state.raw_df[source_col]
                else:
                    # Si la colonne est facultative, on peut générer des valeurs par défaut
                    if required_col == 'lithologie':
                        mapped_df[required_col] = 'Inconnu'  # Valeur par défaut pour la lithologie
                    elif required_col == 'company':
                        mapped_df[required_col] = 'Non spécifiée'  # Valeur par défaut pour l'entreprise
            
            # Stocker le DataFrame mappé dans la session
            st.session_state.df = mapped_df
            st.session_state.columns_mapped = True
            st.success("✅ Mappage validé! Vous pouvez maintenant explorer et modéliser vos données.")
            st.rerun()

elif data_option == "Charger mes données" and st.session_state.columns_mapped:
    # Utiliser le DataFrame déjà mappé
    df = st.session_state.df
    
elif data_option == "Utiliser données démo":
    # Données de démonstration
    st.markdown("""
    <div class="info-box">
        <b>Mode démo</b>: Utilisation de données synthétiques pour illustrer le fonctionnement de l'application.
    </div>
    """, unsafe_allow_html=True)
    
    # Créer des données synthétiques pour la démonstration
    np.random.seed(42)
    n_samples = 1000
    
    prof_finale = np.random.uniform(100, 1000, n_samples)
    azimuth_initial = np.random.uniform(0, 360, n_samples)
    inclinaison_initiale = np.random.uniform(-90, 0, n_samples)
    vitesse_rotation = np.random.uniform(50, 200, n_samples)
    
    # Lithologies possibles
    lithologies = ['Granite', 'Schiste', 'Gneiss', 'Calcaire', 'Basalte']
    lithologie = np.random.choice(lithologies, n_samples)
    
    # Entreprises de forage possibles
    companies = ['ForageTech', 'MineXpert', 'DrillPro', 'GeoForage', 'TerraDrill']
    company = np.random.choice(companies, n_samples)
    
    # Créer une relation entre les entrées et les déviations (simplifiée)
    azimuth_deviation = (
        0.05 * prof_finale 
        + 0.02 * azimuth_initial 
        + 0.1 * inclinaison_initiale 
        + 0.03 * vitesse_rotation 
        + np.random.normal(0, 10, n_samples)
    )
    
    inclinaison_deviation = (
        0.03 * prof_finale 
        - 0.01 * azimuth_initial 
        + 0.05 * inclinaison_initiale 
        + 0.02 * vitesse_rotation 
        + np.random.normal(0, 5, n_samples)
    )
    
    # Ajouter un effet de la lithologie (différent pour chaque type)
    lithology_effect = {
        'Granite': (2.0, 1.0),
        'Schiste': (-1.5, 3.0),
        'Gneiss': (0.5, -2.0),
        'Calcaire': (-1.0, -1.5),
        'Basalte': (3.0, 2.5)
    }
    
    # Ajouter un effet de l'entreprise de forage (différent pour chaque entreprise)
    company_effect = {
        'ForageTech': (1.5, 0.8),
        'MineXpert': (-1.0, -0.5),
        'DrillPro': (0.0, 2.0),
        'GeoForage': (-2.0, -1.0),
        'TerraDrill': (2.5, 1.5)
    }
    
    for i in range(n_samples):
        # Effet de la lithologie
        effect_az_lith, effect_inc_lith = lithology_effect[lithologie[i]]
        azimuth_deviation[i] += effect_az_lith
        inclinaison_deviation[i] += effect_inc_lith
        
        # Effet de l'entreprise de forage
        effect_az_comp, effect_inc_comp = company_effect[company[i]]
        azimuth_deviation[i] += effect_az_comp
        inclinaison_deviation[i] += effect_inc_comp
    
    # Créer le DataFrame
    df = pd.DataFrame({
        'profondeur_finale': prof_finale,
        'azimuth_initial': azimuth_initial,
        'inclinaison_initiale': inclinaison_initiale,
        'lithologie': lithologie,
        'vitesse_rotation': vitesse_rotation,
        'company': company,
        'deviation_azimuth': azimuth_deviation,
        'deviation_inclinaison': inclinaison_deviation
    })
    
    # Stocker dans la session state
    st.session_state.df = df
    st.session_state.columns_mapped = True

# Si des données sont disponibles, afficher l'application principale
if df is not None:
    # Onglets pour les différentes sections
    tabs = st.tabs(["📊 Exploration", "🧠 Modélisation", "🔮 Prédiction"])
    
    with tabs[0]:  # Exploration des données
        st.markdown("<h2>Exploration des données</h2>", unsafe_allow_html=True)
        
        # Affichage des données en deux colonnes
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("<h3>Aperçu des données</h3>", unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
        
        with col2:
            st.markdown("<h3>Statistiques descriptives</h3>", unsafe_allow_html=True)
            st.dataframe(df.describe().style.highlight_max(axis=0), use_container_width=True)
        
        # Distribution des lithologies et métriques globales
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Onglets pour la distribution des catégories
            category_tabs = st.tabs(["Lithologie", "Entreprise"])
            
            with category_tabs[0]:
                st.markdown("<h3>Distribution des lithologies</h3>", unsafe_allow_html=True)
                fig_litho = px.histogram(df, x='lithologie', color='lithologie', 
                                         color_discrete_sequence=px.colors.qualitative.Bold,
                                         template="plotly_white")
                fig_litho.update_layout(
                    xaxis_title="Lithologie",
                    yaxis_title="Nombre de forages",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_litho, use_container_width=True)
            
            with category_tabs[1]:
                st.markdown("<h3>Distribution des entreprises de forage</h3>", unsafe_allow_html=True)
                fig_company = px.histogram(df, x='company', color='company', 
                                         color_discrete_sequence=px.colors.qualitative.Set2,
                                         template="plotly_white")
                fig_company.update_layout(
                    xaxis_title="Entreprise",
                    yaxis_title="Nombre de forages",
                    showlegend=False,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_company, use_container_width=True)
            
        with col2:
            st.markdown("<h3>Métriques globales</h3>", unsafe_allow_html=True)
            
            # Calculer des métriques intéressantes
            mean_az_dev = df['deviation_azimuth'].abs().mean()
            max_az_dev = df['deviation_azimuth'].abs().max()
            mean_inc_dev = df['deviation_inclinaison'].abs().mean()
            max_inc_dev = df['deviation_inclinaison'].abs().max()
            
            # Afficher les métriques
            c1, c2 = st.columns(2)
            c1.metric("Déviation moyenne d'azimuth", f"{mean_az_dev:.2f}°")
            c2.metric("Déviation max. d'azimuth", f"{max_az_dev:.2f}°")
            
            c1, c2 = st.columns(2)
            c1.metric("Déviation moyenne d'inclinaison", f"{mean_inc_dev:.2f}°")
            c2.metric("Déviation max. d'inclinaison", f"{max_inc_dev:.2f}°")
            
            # Calculer la lithologie avec la déviation la plus importante
            lithology_deviation = df.groupby('lithologie')[['deviation_azimuth', 'deviation_inclinaison']].apply(
                lambda x: (x['deviation_azimuth']**2 + x['deviation_inclinaison']**2).mean()**0.5
            ).sort_values(ascending=False)
            
            most_deviated = lithology_deviation.index[0]
            deviation_value = lithology_deviation.iloc[0]
            
            # Calculer l'entreprise avec la déviation la plus importante
            company_deviation = df.groupby('company')[['deviation_azimuth', 'deviation_inclinaison']].apply(
                lambda x: (x['deviation_azimuth']**2 + x['deviation_inclinaison']**2).mean()**0.5
            ).sort_values(ascending=False)
            
            most_deviated_company = company_deviation.index[0]
            company_deviation_value = company_deviation.iloc[0]
            
            # Définir la couleur pour la lithologie la plus déviée
            lithology_colors = {
                'Granite': '#FF6B6B', 
                'Schiste': '#4ECDC4', 
                'Gneiss': '#45B7D1', 
                'Calcaire': '#FFBE0B', 
                'Basalte': '#9F84BD'
            }
            color = lithology_colors.get(most_deviated, '#4F8BF9')
            
            st.markdown(f"""
            <div style="margin-top: 1rem; background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; border-left: 4px solid {color};">
                <p style="margin: 0; font-size: 0.95rem; font-weight: 500;">Lithologie avec le plus de déviation :</p>
                <p style="margin: 0; font-weight: 700; font-size: 1.2rem; color: #1e3a8a;">{most_deviated} ({deviation_value:.2f}°)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Afficher l'entreprise avec le plus de déviation
            st.markdown(f"""
            <div style="margin-top: 1rem; background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; border-left: 4px solid #3563E9;">
                <p style="margin: 0; font-size: 0.95rem; font-weight: 500;">Entreprise avec le plus de déviation :</p>
                <p style="margin: 0; font-weight: 700; font-size: 1.2rem; color: #1e3a8a;">{most_deviated_company} ({company_deviation_value:.2f}°)</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Analyse exploratoire détaillée
        st.markdown("<h3>Analyse exploratoire approfondie</h3>", unsafe_allow_html=True)
        
        explore_tabs = st.tabs(["Corrélations", "Déviations par catégorie", "Relations"])
        
        with explore_tabs[0]:
            # Matrice de corrélation
            numeric_cols = df.select_dtypes(include=np.number).columns
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(corr_matrix, 
                                text_auto=True, 
                                color_continuous_scale='RdBu_r',
                                title="Matrice de corrélation",
                                template="plotly_white")
            fig_corr.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Interprétation automatique des corrélations
            strong_correlations = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if abs(corr_matrix.iloc[i, j]) > 0.3:  # Seuil de corrélation
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'corr': corr_matrix.iloc[i, j]
                        })
            
            if strong_correlations:
                st.markdown("<h4>Corrélations significatives</h4>", unsafe_allow_html=True)
                for corr in sorted(strong_correlations, key=lambda x: abs(x['corr']), reverse=True):
                    relation = "positive" if corr['corr'] > 0 else "négative"
                    strength = "forte" if abs(corr['corr']) > 0.7 else "modérée"
                    st.markdown(f"- Corrélation {strength} {relation} ({corr['corr']:.2f}) entre **{corr['var1']}** et **{corr['var2']}**")
        
        with explore_tabs[1]:
            # Déviations par catégorie
            deviation_category_tabs = st.tabs(["Par lithologie", "Par entreprise"])
            
            with deviation_category_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_box1 = px.box(df, x='lithologie', y='deviation_azimuth', 
                                    title="Déviation d'azimuth par lithologie", 
                                    color='lithologie',
                                    color_discrete_sequence=px.colors.qualitative.Bold,
                                    template="plotly_white")
                    fig_box1.update_layout(
                        xaxis_title="Lithologie",
                        yaxis_title="Déviation d'azimuth (°)",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_box1, use_container_width=True)
                
                with col2:
                    fig_box2 = px.box(df, x='lithologie', y='deviation_inclinaison', 
                                    title="Déviation d'inclinaison par lithologie", 
                                    color='lithologie',
                                    color_discrete_sequence=px.colors.qualitative.Bold,
                                    template="plotly_white")
                    fig_box2.update_layout(
                        xaxis_title="Lithologie",
                        yaxis_title="Déviation d'inclinaison (°)",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_box2, use_container_width=True)
                
                # Résumé statistique par lithologie
                st.markdown("<h4>Résumé statistique par lithologie</h4>", unsafe_allow_html=True)
                
                litho_stats = df.groupby('lithologie')[['deviation_azimuth', 'deviation_inclinaison']].agg(
                    ['mean', 'std', 'min', 'max']
                ).round(2)
                
                litho_stats.columns = ['Azimuth Moy', 'Azimuth Std', 'Azimuth Min', 'Azimuth Max', 
                                    'Inclinaison Moy', 'Inclinaison Std', 'Inclinaison Min', 'Inclinaison Max']
                
                st.dataframe(litho_stats, use_container_width=True)
            
            with deviation_category_tabs[1]:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_box1 = px.box(df, x='company', y='deviation_azimuth', 
                                    title="Déviation d'azimuth par entreprise", 
                                    color='company',
                                    color_discrete_sequence=px.colors.qualitative.Set2,
                                    template="plotly_white")
                    fig_box1.update_layout(
                        xaxis_title="Entreprise",
                        yaxis_title="Déviation d'azimuth (°)",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_box1, use_container_width=True)
                
                with col2:
                    fig_box2 = px.box(df, x='company', y='deviation_inclinaison', 
                                    title="Déviation d'inclinaison par entreprise", 
                                    color='company',
                                    color_discrete_sequence=px.colors.qualitative.Set2,
                                    template="plotly_white")
                    fig_box2.update_layout(
                        xaxis_title="Entreprise",
                        yaxis_title="Déviation d'inclinaison (°)",
                        showlegend=False,
                        margin=dict(l=20, r=20, t=50, b=20),
                    )
                    st.plotly_chart(fig_box2, use_container_width=True)
                
                # Résumé statistique par entreprise
                st.markdown("<h4>Résumé statistique par entreprise</h4>", unsafe_allow_html=True)
                
                company_stats = df.groupby('company')[['deviation_azimuth', 'deviation_inclinaison']].agg(
                    ['mean', 'std', 'min', 'max']
                ).round(2)
                
                company_stats.columns = ['Azimuth Moy', 'Azimuth Std', 'Azimuth Min', 'Azimuth Max', 
                                      'Inclinaison Moy', 'Inclinaison Std', 'Inclinaison Min', 'Inclinaison Max']
                
                st.dataframe(company_stats, use_container_width=True)
        
        with explore_tabs[2]:
            # Relations entre paramètres et déviations
            features = ['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'vitesse_rotation']
            
            selected_feature = st.selectbox(
                "Sélectionner un paramètre pour explorer sa relation avec les déviations",
                features
            )
            
            # Sélectionner la variable catégorielle pour la couleur
            color_var = st.radio(
                "Colorer par:",
                ["lithologie", "company"],
                horizontal=True
            )
            
            color_discrete_map = px.colors.qualitative.Bold if color_var == "lithologie" else px.colors.qualitative.Set2
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_scatter1 = px.scatter(df, x=selected_feature, y='deviation_azimuth', 
                                        color=color_var, opacity=0.7,
                                        title=f"Déviation d'azimuth vs {selected_feature}",
                                        color_discrete_sequence=color_discrete_map,
                                        trendline="ols",
                                        template="plotly_white")
                fig_scatter1.update_layout(
                    xaxis_title=selected_feature,
                    yaxis_title="Déviation d'azimuth (°)",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scatter1, use_container_width=True)
            
            with col2:
                fig_scatter2 = px.scatter(df, x=selected_feature, y='deviation_inclinaison', 
                                        color=color_var, opacity=0.7,
                                        title=f"Déviation d'inclinaison vs {selected_feature}",
                                        color_discrete_sequence=color_discrete_map,
                                        trendline="ols",
                                        template="plotly_white")
                fig_scatter2.update_layout(
                    xaxis_title=selected_feature,
                    yaxis_title="Déviation d'inclinaison (°)",
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_scatter2, use_container_width=True)
            
            # Distribution du paramètre sélectionné
            fig_hist = px.histogram(df, x=selected_feature, color=color_var,
                                   title=f"Distribution de {selected_feature}",
                                   color_discrete_sequence=color_discrete_map,
                                   marginal="box",
                                   template="plotly_white")
            fig_hist.update_layout(
                xaxis_title=selected_feature,
                yaxis_title="Nombre de forages",
                margin=dict(l=20, r=20, t=50, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tabs[1]:  # Modélisation
        st.markdown("<h2>Modélisation des déviations</h2>", unsafe_allow_html=True)
        
        # Vérifier si l'augmentation des données est demandée
        if st.session_state.use_augmented_data and st.session_state.augmented_df is None:
            # Augmenter les données
            with st.spinner("Augmentation des données en cours..."):
                augmented_df = augment_data(df, num_augmented_samples=aug_samples, noise_level=noise_level)
                st.session_state.augmented_df = augmented_df
                
                # Afficher une info sur l'augmentation des données
                st.info(f"✅ Données augmentées : {len(df)} échantillons originaux + {len(augmented_df) - len(df)} échantillons synthétiques = {len(augmented_df)} échantillons au total")
                
                # Calculer le pourcentage d'augmentation
                aug_percentage = ((len(augmented_df) - len(df)) / len(df)) * 100
                
                # Afficher un graphique comparant avant/après
                data_size_comparison = pd.DataFrame({
                    'Type de données': ['Données originales', 'Données augmentées'],
                    'Nombre d\'échantillons': [len(df), len(augmented_df)]
                })
                
                fig_augmentation = px.bar(data_size_comparison, 
                                         x='Type de données', 
                                         y='Nombre d\'échantillons',
                                         color='Type de données',
                                         color_discrete_sequence=['#3563E9', '#0CCE6B'],
                                         title=f"Effet de l'augmentation des données (+{aug_percentage:.1f}%)",
                                         template="plotly_white")
                fig_augmentation.update_layout(
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="Nombre d'échantillons"
                )
                st.plotly_chart(fig_augmentation, use_container_width=True)
                
                # Utiliser les données augmentées pour l'entraînement
                modeling_df = augmented_df
        elif st.session_state.use_augmented_data and st.session_state.augmented_df is not None:
            # Utiliser les données augmentées déjà générées
            modeling_df = st.session_state.augmented_df
            st.info(f"✅ Utilisation des données augmentées : {len(modeling_df)} échantillons au total")
        else:
            # Utiliser les données originales
            modeling_df = df
        
        # Définir les caractéristiques et cibles
        X = modeling_df[['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'lithologie', 'company', 'vitesse_rotation']]
        y_azimuth = modeling_df['deviation_azimuth']
        y_inclinaison = modeling_df['deviation_inclinaison']
        
        # Préparation pour la modélisation
        numeric_features = ['profondeur_finale', 'azimuth_initial', 'inclinaison_initiale', 'vitesse_rotation']
        categorical_features = ['lithologie', 'company']
        
        # Préprocesseurs
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        # Choix du modèle
        if model_option == "Random Forest":
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
        elif model_option == "SVM":
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', SVR())
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', SVR())
            ])
            
        elif model_option == "Régression Linéaire":
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            
        else:  # Réseau de Neurones
            model_azimuth = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))
            ])
            
            model_inclinaison = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', MLPRegressor(hidden_layer_sizes=(100,50), max_iter=1000, random_state=42))
            ])
        
        # Diviser les données
        X_train, X_test, y_azimuth_train, y_azimuth_test, y_inclinaison_train, y_inclinaison_test = train_test_split(
            X, y_azimuth, y_inclinaison, test_size=0.2, random_state=42
        )
        
        # Description du modèle sélectionné
        model_descriptions = {
            "Random Forest": """
                **Random Forest** est un algorithme d'ensemble qui utilise plusieurs arbres de décision pour améliorer 
                la précision et réduire le surapprentissage. Il est efficace pour capturer les relations non linéaires 
                dans les données.
                
                **Avantages**:
                - Bonne performance sur les données complexes
                - Gère bien les valeurs manquantes
                - Fournit des mesures d'importance des variables
                
                **Complexité du modèle**: Moyenne à élevée
            """,
            "SVM": """
                **Support Vector Machine (SVM)** est un algorithme qui trouve un hyperplan optimal pour séparer les données.
                Dans sa version régression (SVR), il cherche à trouver une fonction qui s'écarte le moins possible des points.
                
                **Avantages**:
                - Efficace dans les espaces de grande dimension
                - Polyvalent grâce aux différents noyaux
                - Bonne capacité de généralisation
                
                **Complexité du modèle**: Moyenne
            """,
            "Régression Linéaire": """
                **Régression Linéaire** est un modèle simple qui établit une relation linéaire entre les variables 
                d'entrée et la sortie. Il est facile à interpréter mais limité pour capturer des relations complexes.
                
                **Avantages**:
                - Simple et interprétable
                - Rapide à entraîner
                - Faible variance
                
                **Complexité du modèle**: Faible
            """,
            "Réseau de Neurones": """
                **Réseau de Neurones** est un modèle inspiré du cerveau humain, composé de couches de neurones artificiels.
                Il peut modéliser des relations très complexes et non linéaires.
                
                **Avantages**:
                - Capacité à modéliser des relations très complexes
                - Peut apprendre des représentations hiérarchiques
                - Très flexible
                
                **Complexité du modèle**: Élevée
            """
        }
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"<h3>Modèle sélectionné: {model_option}</h3>", unsafe_allow_html=True)
            st.markdown(model_descriptions[model_option])
        
        with col2:
            st.markdown("<h3>Configuration</h3>", unsafe_allow_html=True)
            st.markdown(f"""
            <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.25rem; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                <p style="margin: 0; font-weight: 600; font-size: 1.05rem; color: #1e3a8a; margin-bottom: 0.5rem;">Répartition des données</p>
                <p style="margin: 0; font-size: 1rem;">80% Entraînement / 20% Test</p>
            </div>
            
            <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                <p style="margin: 0; font-weight: 600; font-size: 1.05rem; color: #1e3a8a; margin-bottom: 0.5rem;">Variables d'entrée</p>
                <ul style="margin: 0; padding-left: 1.25rem;">
                    <li style="margin-bottom: 0.25rem;">Profondeur finale</li>
                    <li style="margin-bottom: 0.25rem;">Azimuth initial</li>
                    <li style="margin-bottom: 0.25rem;">Inclinaison initiale</li>
                    <li style="margin-bottom: 0.25rem;">Lithologie</li>
                    <li style="margin-bottom: 0.25rem;">Entreprise de forage</li>
                    <li>Vitesse de rotation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Entraîner les modèles si l'utilisateur clique sur le bouton
        if train_button:
            st.markdown("<h3>Entraînement en cours...</h3>", unsafe_allow_html=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Étape 1: Préparation des données
            status_text.text("Préparation des données...")
            progress_bar.progress(20)
            
            # Étape 2: Entraînement du modèle d'azimuth
            status_text.text("Entraînement du modèle pour la déviation d'azimuth...")
            model_azimuth.fit(X_train, y_azimuth_train)
            y_azimuth_pred = model_azimuth.predict(X_test)
            progress_bar.progress(50)
            
            # Étape 3: Entraînement du modèle d'inclinaison
            status_text.text("Entraînement du modèle pour la déviation d'inclinaison...")
            model_inclinaison.fit(X_train, y_inclinaison_train)
            y_inclinaison_pred = model_inclinaison.predict(X_test)
            progress_bar.progress(80)
            
            # Étape 4: Évaluation des performances
            status_text.text("Évaluation des performances...")
            azimuth_rmse = np.sqrt(mean_squared_error(y_azimuth_test, y_azimuth_pred))
            azimuth_r2 = r2_score(y_azimuth_test, y_azimuth_pred)
            
            inclinaison_rmse = np.sqrt(mean_squared_error(y_inclinaison_test, y_inclinaison_pred))
            inclinaison_r2 = r2_score(y_inclinaison_test, y_inclinaison_pred)
            progress_bar.progress(100)
            
            # Stocker les modèles dans la session state
            st.session_state.model_azimuth = model_azimuth
            st.session_state.model_inclinaison = model_inclinaison
            st.session_state.model_trained = True
            
            status_text.text("Entraînement terminé!")
            
            # Affichage des résultats
            st.markdown("<h3>Résultats de l'entraînement</h3>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<h4>Déviation d'azimuth</h4>", unsafe_allow_html=True)
                
                # Métrique avec évaluation de la performance
                r2_color = '#0CCE6B' if azimuth_r2 > 0.7 else ('#FFC107' if azimuth_r2 > 0.5 else '#FF4B4B')
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 1.25rem;">
                    <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; width: 48%; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                        <p style="margin: 0; font-size: 0.95rem; color: #4A5568; font-weight: 500;">RMSE</p>
                        <p style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #3563E9; font-family: 'Poppins', sans-serif;">{azimuth_rmse:.4f}°</p>
                    </div>
                    <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; width: 48%; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                        <p style="margin: 0; font-size: 0.95rem; color: #4A5568; font-weight: 500;">R²</p>
                        <p style="margin: 0; font-weight: 700; font-size: 1.6rem; color: {r2_color}; font-family: 'Poppins', sans-serif;">{azimuth_r2:.4f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique des prédictions vs réelles
                fig_pred_az = px.scatter(x=y_azimuth_test, y=y_azimuth_pred, 
                                        labels={'x': 'Valeurs réelles (°)', 'y': 'Prédictions (°)'},
                                        title="Prédictions vs Réelles - Déviation d'azimuth",
                                        template="plotly_white")
                fig_pred_az.add_shape(type='line', line=dict(dash='dash', color='rgba(0,0,0,0.3)'),
                                    x0=y_azimuth_test.min(), y0=y_azimuth_test.min(),
                                    x1=y_azimuth_test.max(), y1=y_azimuth_test.max())
                fig_pred_az.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pred_az, use_container_width=True)
            
            with col2:
                st.markdown("<h4>Déviation d'inclinaison</h4>", unsafe_allow_html=True)
                
                # Métrique avec évaluation de la performance
                r2_color = '#0CCE6B' if inclinaison_r2 > 0.7 else ('#FFC107' if inclinaison_r2 > 0.5 else '#FF4B4B')
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; margin-bottom: 1.25rem;">
                    <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; width: 48%; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                        <p style="margin: 0; font-size: 0.95rem; color: #4A5568; font-weight: 500;">RMSE</p>
                        <p style="margin: 0; font-weight: 700; font-size: 1.6rem; color: #3563E9; font-family: 'Poppins', sans-serif;">{inclinaison_rmse:.4f}°</p>
                    </div>
                    <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; width: 48%; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                        <p style="margin: 0; font-size: 0.95rem; color: #4A5568; font-weight: 500;">R²</p>
                        <p style="margin: 0; font-weight: 700; font-size: 1.6rem; color: {r2_color}; font-family: 'Poppins', sans-serif;">{inclinaison_r2:.4f}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Graphique des prédictions vs réelles
                fig_pred_inc = px.scatter(x=y_inclinaison_test, y=y_inclinaison_pred, 
                                        labels={'x': 'Valeurs réelles (°)', 'y': 'Prédictions (°)'},
                                        title="Prédictions vs Réelles - Déviation d'inclinaison",
                                        template="plotly_white")
                fig_pred_inc.add_shape(type='line', line=dict(dash='dash', color='rgba(0,0,0,0.3)'),
                                    x0=y_inclinaison_test.min(), y0=y_inclinaison_test.min(),
                                    x1=y_inclinaison_test.max(), y1=y_inclinaison_test.max())
                fig_pred_inc.update_layout(
                    margin=dict(l=20, r=20, t=50, b=20),
                )
                st.plotly_chart(fig_pred_inc, use_container_width=True)
            
            # Interprétation des résultats
            st.markdown("<h3>Interprétation des résultats</h3>", unsafe_allow_html=True)
            
            # Calculer la performance moyenne
            avg_r2 = (azimuth_r2 + inclinaison_r2) / 2
            
            # Déterminer la classe de performance et le texte correspondant
            if avg_r2 > 0.8:
                performance_class = "excellent"
                performance_text = "excellente"
                performance_detail = """
                    Le modèle capture très bien les facteurs influençant les déviations. Vous pouvez utiliser ces prédictions 
                    avec un haut niveau de confiance pour la planification des forages.
                """
            elif avg_r2 > 0.7:
                performance_class = "good"
                performance_text = "bonne"
                performance_detail = """
                    Le modèle capture bien les tendances principales des déviations. Les prédictions sont fiables pour
                    la plupart des conditions de forage.
                """
            elif avg_r2 > 0.5:
                performance_class = "moderate"
                performance_text = "modérée"
                performance_detail = """
                    Le modèle capture les tendances générales mais manque de précision dans certains cas. Utilisez les 
                    prédictions comme indicateurs mais prévoyez des marges de sécurité.
                """
            else:
                performance_class = "limited"
                performance_text = "limitée"
                performance_detail = """
                    Le modèle a du mal à capturer la complexité des facteurs influençant les déviations. Les prédictions
                    doivent être utilisées avec prudence et des facteurs supplémentaires pourraient être nécessaires.
                """
            
            # Ajouter une information si l'augmentation des données a été utilisée
            augmentation_note = ""
            if st.session_state.use_augmented_data:
                augmentation_note = f"""
                <div class="augment-note">
                    Note: L'augmentation des données a été utilisée pour l'entraînement ({len(modeling_df)} échantillons au total).
                </div>
                """
            
            # Afficher la carte de performance améliorée
            st.markdown(f"""
            <div class="performance-card {performance_class}">
                <div class="score-badge">R² = {avg_r2:.2f}</div>
                <h4>Performance globale: {performance_text.capitalize()}</h4>
                <p>{performance_detail}</p>
                {augmentation_note}
            </div>
            """, unsafe_allow_html=True)
            
            # Si Random Forest, afficher l'importance des caractéristiques
            if model_option == "Random Forest":
                st.markdown("<h3>Importance des caractéristiques</h3>", unsafe_allow_html=True)
                
                # Extraire l'importance des caractéristiques pour l'azimuth
                rf_azimuth = model_azimuth.named_steps['regressor']
                preprocessor_azimuth = model_azimuth.named_steps['preprocessor']
                
                # Obtenir les noms des caractéristiques après transformation
                cat_features = preprocessor_azimuth.transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)
                feature_names = np.concatenate([numeric_features, cat_features])
                
                # Obtenir l'importance des caractéristiques
                feature_importance_azimuth = rf_azimuth.feature_importances_
                
                # Pour l'inclinaison
                rf_inclinaison = model_inclinaison.named_steps['regressor']
                feature_importance_inclinaison = rf_inclinaison.feature_importances_
                
                # Créer un DataFrame pour l'affichage
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance_Azimuth': feature_importance_azimuth,
                    'Importance_Inclinaison': feature_importance_inclinaison
                })
                
                # Afficher sous forme de graphique
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_imp_az = px.bar(
                        importance_df.sort_values('Importance_Azimuth', ascending=False),
                        y='Feature', x='Importance_Azimuth',
                        title="Importance des facteurs - Déviation d'azimuth",
                        template="plotly_white",
                        color='Importance_Azimuth',
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    fig_imp_az.update_layout(
                        yaxis_title="",
                        xaxis_title="Importance relative",
                        margin=dict(l=20, r=20, t=50, b=20),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_imp_az, use_container_width=True)
                
                with col2:
                    fig_imp_inc = px.bar(
                        importance_df.sort_values('Importance_Inclinaison', ascending=False),
                        y='Feature', x='Importance_Inclinaison',
                        title="Importance des facteurs - Déviation d'inclinaison",
                        template="plotly_white",
                        color='Importance_Inclinaison',
                        color_continuous_scale=px.colors.sequential.Blues
                    )
                    fig_imp_inc.update_layout(
                        yaxis_title="",
                        xaxis_title="Importance relative",
                        margin=dict(l=20, r=20, t=50, b=20),
                        coloraxis_showscale=False
                    )
                    st.plotly_chart(fig_imp_inc, use_container_width=True)
    
    with tabs[2]:  # Prédiction
        st.markdown("<h2>Prédiction pour un nouveau forage</h2>", unsafe_allow_html=True)
        
        # Vérification si un modèle est entraîné
        if not st.session_state.model_trained:
            st.markdown("""
            <div style="background-color: #FFF5F5; border-left: 4px solid #FF4B4B; padding: 1.25rem; border-radius: 10px; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(255, 75, 75, 0.1);">
                <p style="margin: 0; color: #C53030; font-weight: 700; margin-bottom: 0.5rem; font-size: 1.1rem;">Modèle non entraîné</p>
                <p style="margin: 0; color: #C53030;">Veuillez d'abord entraîner un modèle depuis l'onglet "Modélisation" ou la barre latérale.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Formulaire de prédiction
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3>Paramètres du forage</h3>", unsafe_allow_html=True)
            
            prof_finale_input = st.number_input("Profondeur finale (m)", min_value=50.0, max_value=2000.0, value=500.0, step=50.0)
            azimuth_initial_input = st.number_input("Azimuth initial (degrés)", min_value=0.0, max_value=360.0, value=90.0, step=5.0)
            inclinaison_initiale_input = st.number_input("Inclinaison initiale (degrés)", min_value=-90.0, max_value=0.0, value=-45.0, step=5.0)
            
            lithologies_uniques = df['lithologie'].unique().tolist()
            lithologie_input = st.selectbox("Lithologie", lithologies_uniques)
            
            companies_uniques = df['company'].unique().tolist()
            company_input = st.selectbox("Entreprise de forage", companies_uniques)
            
            vitesse_rotation_input = st.number_input("Vitesse de rotation (tr/min)", min_value=20.0, max_value=300.0, value=120.0, step=10.0)
        
        with col2:
            st.markdown("<h3>Illustration schématique</h3>", unsafe_allow_html=True)
            
            # Visualisation schématique du forage initial
            def generate_drill_illustration(azimuth, inclination):
                fig = go.Figure()
                
                # Calculer les coordonnées pour une représentation simple
                depth = 100
                x = depth * np.cos(np.radians(inclination)) * np.sin(np.radians(azimuth))
                y = depth * np.cos(np.radians(inclination)) * np.cos(np.radians(azimuth))
                z = depth * np.sin(np.radians(inclination))
                
                # Surface (grille)
                x_surface = np.linspace(-100, 100, 5)
                y_surface = np.linspace(-100, 100, 5)
                z_surface = np.zeros((5, 5))
                
                fig.add_trace(go.Surface(x=x_surface, y=y_surface, z=z_surface, 
                                        colorscale=[[0, '#E2F3EC'], [1, '#C6E8DE']],
                                        showscale=False, opacity=0.5))
                
                # Point de départ du forage
                fig.add_trace(go.Scatter3d(
                    x=[0], y=[0], z=[0],
                    mode='markers',
                    marker=dict(size=12, color='#4CAF50', symbol='diamond'),
                    name='Départ'
                ))
                
                # Direction initiale
                fig.add_trace(go.Scatter3d(
                    x=[0, x], y=[0, y], z=[0, z],
                    mode='lines',
                    line=dict(color='#3563E9', width=6),
                    name='Direction initiale'
                ))
                
                # Paramètres de visualisation
                fig.update_layout(
                    title=f"Orientation initiale: Azimuth {azimuth:.1f}°, Inclinaison {inclination:.1f}°",
                    scene = dict(
                        xaxis_title='Est (m)',
                        yaxis_title='Nord (m)',
                        zaxis_title='Profondeur (m)',
                        aspectmode='manual',
                        aspectratio=dict(x=1, y=1, z=1),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2)
                        ),
                    ),
                    margin=dict(l=0, r=0, t=40, b=0),
                    template="plotly_white",
                    height=300
                )
                
                return fig
            
            drill_fig = generate_drill_illustration(azimuth_initial_input, inclinaison_initiale_input)
            st.plotly_chart(drill_fig, use_container_width=True)
            
            # Informations supplémentaires sur la lithologie et l'entreprise
            lithology_info = {
                'Granite': "Roche ignée à grains grossiers, abrasive et résistante.",
                'Schiste': "Roche métamorphique feuilletée de dureté moyenne.",
                'Gneiss': "Roche métamorphique à bandes alternées, dure et résistante.",
                'Calcaire': "Roche sédimentaire tendre à moyennement dure.",
                'Basalte': "Roche volcanique dense, dure et abrasive."
            }
            
            company_info = {
                'ForageTech': "Spécialiste des forages de haute précision, technologie avancée.",
                'MineXpert': "Expertise en terrains difficiles, approche méthodique.",
                'DrillPro': "Service rapide, bon rapport qualité-prix.",
                'GeoForage': "Spécialiste en forages profonds, équipement robuste.",
                'TerraDrill': "Technologie innovante, personnel hautement qualifié."
            }
            
            st.markdown(f"""
            <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; flex: 1; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                    <p style="margin: 0; font-weight: 600; color: #1e3a8a; margin-bottom: 0.5rem;">Lithologie: {lithologie_input}</p>
                    <p style="margin: 0; font-size: 0.95rem; color: #4A5568;">{lithology_info.get(lithologie_input, "")}</p>
                </div>
                <div style="background-color: #F0F4FA; padding: 1.25rem; border-radius: 10px; flex: 1; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0;">
                    <p style="margin: 0; font-weight: 600; color: #1e3a8a; margin-bottom: 0.5rem;">Entreprise: {company_input}</p>
                    <p style="margin: 0; font-size: 0.95rem; color: #4A5568;">{company_info.get(company_input, "")}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Bouton de prédiction
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        
        with predict_col2:
            predict_button = st.button("⚡ Prédire les déviations", use_container_width=True, type="primary", disabled=not st.session_state.model_trained)
        
        # Faire la prédiction
        if predict_button and st.session_state.model_trained:
            # Créer un dataframe avec les données d'entrée
            input_data = pd.DataFrame({
                'profondeur_finale': [prof_finale_input],
                'azimuth_initial': [azimuth_initial_input],
                'inclinaison_initiale': [inclinaison_initiale_input],
                'lithologie': [lithologie_input],
                'company': [company_input],
                'vitesse_rotation': [vitesse_rotation_input]
            })
            
            # Faire les prédictions avec les modèles stockés dans session_state
            predicted_azimuth = st.session_state.model_azimuth.predict(input_data)[0]
            predicted_inclinaison = st.session_state.model_inclinaison.predict(input_data)[0]
            
            # Calculer les valeurs finales
            azimuth_final = azimuth_initial_input + predicted_azimuth
            inclinaison_final = inclinaison_initiale_input + predicted_inclinaison
            
            # Normalization pour azimuth (0-360°)
            azimuth_final = azimuth_final % 360
            
            # Contraindre l'inclinaison entre -90 et 0
            inclinaison_final = max(-90, min(0, inclinaison_final))
            
            # Afficher les résultats
            st.markdown("<h3>Résultats de la prédiction</h3>", unsafe_allow_html=True)
            
            # Créer un cadre moderne pour les résultats
            st.markdown("""
            <div style="background-color: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); padding: 1.75rem; margin: 1.75rem 0; border: 1px solid #f0f0f5;">
                <h4 style="margin-top: 0; text-align: center; margin-bottom: 1.75rem; color: #1e3a8a; font-family: 'Poppins', sans-serif;">Déviations prédites</h4>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Déviation d'azimuth", f"{predicted_azimuth:.2f}°")
                st.metric("Azimuth final", f"{azimuth_final:.2f}°")
            
            with col2:
                st.metric("Déviation d'inclinaison", f"{predicted_inclinaison:.2f}°")
                st.metric("Inclinaison finale", f"{inclinaison_final:.2f}°")
            
            # Calcul de l'intensité de la déviation pour le texte d'interprétation
            deviation_magnitude = (predicted_azimuth**2 + predicted_inclinaison**2)**0.5
            
            if deviation_magnitude < 5:
                deviation_text = "faible"
                deviation_impact = "minime"
                deviation_color = "#0CCE6B"
            elif deviation_magnitude < 15:
                deviation_text = "modérée"
                deviation_impact = "à considérer"
                deviation_color = "#FFC107"
            else:
                deviation_text = "importante"
                deviation_impact = "significatif"
                deviation_color = "#FF4B4B"
            
            st.markdown(f"""
                <p style="text-align: center; margin: 1.25rem 0; padding-top: 1.25rem; border-top: 1px solid #f0f0f5; font-size: 1.1rem; color: #4A5568;">
                    La déviation prédite est <strong style="color: {deviation_color};">{deviation_text}</strong>, avec un impact <strong>{deviation_impact}</strong> sur la position finale du forage.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Visualisation 3D de la trajectoire du forage
            st.markdown("<h3>Visualisation de la trajectoire</h3>", unsafe_allow_html=True)
            
            # Conversion des coordonnées polaires en coordonnées cartésiennes
            def sph_to_cart(depth, azimuth, inclination):
                # Conversion des degrés en radians
                azimuth_rad = np.radians(azimuth)
                inclination_rad = np.radians(inclination)
                
                # x pointe vers l'est, y vers le nord, z vers le haut
                x = depth * np.cos(inclination_rad) * np.sin(azimuth_rad)
                y = depth * np.cos(inclination_rad) * np.cos(azimuth_rad)
                z = depth * np.sin(inclination_rad)  # z est négatif car inclination est négative
                
                return x, y, z
            
            # Calculer plusieurs points le long de la trajectoire
            num_points = 100
            depths = np.linspace(0, prof_finale_input, num_points)
            
            # Interpolation linéaire entre les angles initiaux et finaux
            azimuth_interp = np.linspace(azimuth_initial_input, azimuth_final, num_points)
            inclination_interp = np.linspace(inclinaison_initiale_input, inclinaison_final, num_points)
            
            # Calculer les coordonnées cartésiennes pour chaque point
            x_coords, y_coords, z_coords = [], [], []
            for i in range(num_points):
                x, y, z = sph_to_cart(depths[i], azimuth_interp[i], inclination_interp[i])
                x_coords.append(x)
                y_coords.append(y)
                z_coords.append(z)
            
            # Surface (grille)
            x_surface = np.linspace(min(x_coords)-50, max(x_coords)+50, 10)
            y_surface = np.linspace(min(y_coords)-50, max(y_coords)+50, 10)
            x_surface_grid, y_surface_grid = np.meshgrid(x_surface, y_surface)
            z_surface_grid = np.zeros_like(x_surface_grid)
            
            # Créer la visualisation 3D
            fig = go.Figure()
            
            # Ajouter la surface
            fig.add_trace(go.Surface(
                x=x_surface_grid,
                y=y_surface_grid,
                z=z_surface_grid,
                colorscale=[[0, '#E2F3EC'], [1, '#C6E8DE']],
                showscale=False,
                opacity=0.5
            ))
            
            # Ajouter la trajectoire
            fig.add_trace(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(
                    color='#3563E9',
                    width=7
                ),
                name='Trajectoire prédite'
            ))
            
            # Ajouter la position de départ
            fig.add_trace(go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#4CAF50',
                    symbol='diamond'
                ),
                name='Départ'
            ))
            
            # Ajouter la position finale
            fig.add_trace(go.Scatter3d(
                x=[x_coords[-1]],
                y=[y_coords[-1]],
                z=[z_coords[-1]],
                mode='markers',
                marker=dict(
                    size=10,
                    color='#FF4B4B',
                    symbol='diamond'
                ),
                name='Arrivée'
            ))
            
            # Ajouter la trajectoire idéale (ligne droite) avec une meilleure visibilité
            x_ideal, y_ideal, z_ideal = sph_to_cart(prof_finale_input, azimuth_initial_input, inclinaison_initiale_input)
            
            fig.add_trace(go.Scatter3d(
                x=[0, x_ideal],
                y=[0, y_ideal],
                z=[0, z_ideal],
                mode='lines',
                line=dict(
                    color='rgba(255, 99, 71, 0.7)',  # Rouge plus visible
                    width=5,                        # Ligne plus épaisse
                    dash='dash'                     # Conserver le style tiret
                ),
                name='Trajectoire idéale'
            ))
            
            # Calculer l'écart final en mètres
            final_deviation = ((x_coords[-1] - x_ideal)**2 + (y_coords[-1] - y_ideal)**2 + (z_coords[-1] - z_ideal)**2)**0.5
            
            fig.update_layout(
                title=f"Trajectoire du forage (Écart final: {final_deviation:.2f} m)",
                scene=dict(
                    xaxis_title='Est (m)',
                    yaxis_title='Nord (m)',
                    zaxis_title='Profondeur (m)',
                    aspectmode='data'
                ),
                template="plotly_white",
                height=700,
                margin=dict(l=0, r=0, t=50, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Ajouter une section d'interprétation et de recommandation
            st.markdown("<h3>Interprétation et recommandations</h3>", unsafe_allow_html=True)
            
            # Déterminer les recommandations basées sur la déviation
            if deviation_magnitude < 5:
                recommendations = """
                - La déviation prédite est faible et ne devrait pas nécessiter d'ajustements particuliers.
                - Procéder au forage selon les paramètres planifiés.
                - Surveiller régulièrement l'orientation pendant l'opération.
                """
                gauge_color = "#0CCE6B"
            elif deviation_magnitude < 15:
                recommendations = """
                - Une déviation modérée est anticipée, des ajustements préventifs peuvent être envisagés.
                - Considérer une légère compensation de l'orientation initiale.
                - Prévoir des mesures de contrôle plus fréquentes pendant le forage.
                - Réduire la vitesse de rotation dans les zones critiques.
                """
                gauge_color = "#FFC107"
            else:
                recommendations = """
                - Une déviation importante est prévue, des mesures correctives sont nécessaires.
                - Ajuster significativement l'orientation initiale pour compenser la déviation.
                - Utiliser des stabilisateurs supplémentaires pour maintenir la trajectoire.
                - Envisager des techniques de forage dirigé si disponibles.
                - Effectuer des mesures de contrôle très fréquentes.
                """
                gauge_color = "#FF4B4B"
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="background-color: white; border-radius: 12px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); padding: 1.75rem; height: 100%; border: 1px solid #f0f0f5;">
                    <h4 style="margin-top: 0; color: #1e3a8a; font-family: 'Poppins', sans-serif;">Recommandations</h4>
                    <ul style="margin-top: 1rem; padding-left: 1.5rem; color: #4A5568;">
                        {recommendations.replace('- ', '<li style="margin-bottom: 0.75rem;">').replace('\n', '</li>')}
                    </ul>
                    <p style="margin-top: 1.5rem; font-style: italic; color: #64748B; font-size: 0.95rem;">Note: Ces recommandations sont basées sur les prédictions du modèle et doivent être adaptées aux conditions spécifiques du site.</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Créer une jauge pour visualiser l'intensité de la déviation
                gauge_fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=deviation_magnitude,
                    title={'text': "Intensité de la déviation (°)", 'font': {'family': 'Poppins', 'size': 18}},
                    gauge={
                        'axis': {'range': [None, 30], 'tickwidth': 1, 'tickcolor': "#4A5568"},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'steps': [
                            {'range': [0, 5], 'color': "#E2F3EC"},
                            {'range': [5, 15], 'color': "#FFEFC0"},
                            {'range': [15, 30], 'color': "#FFEAE5"}
                        ],
                        'threshold': {
                            'line': {'color': gauge_color, 'width': 4},
                            'thickness': 0.75,
                            'value': deviation_magnitude
                        }
                    },
                    number={'font': {'family': 'Poppins', 'size': 30, 'color': gauge_color}}
                ))
                
                gauge_fig.update_layout(
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                    template="plotly_white",
                    paper_bgcolor="white",
                    font={'family': 'DM Sans'}
                )
                
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Ajouter une option pour télécharger le rapport
            st.markdown("<h3>Télécharger le rapport</h3>", unsafe_allow_html=True)
            
            # Date du rapport
            report_date = datetime.datetime.now().strftime('%d/%m/%Y')
            
            # Créer un rapport PDF (simulé avec un texte formaté)
            report_text = f"""
RAPPORT DE PRÉDICTION DE DÉVIATION DE FORAGE
===========================================
Date: {report_date}
Auteur: Didier Ouedraogo, P.Geo. - Géologue & Data Scientist

PARAMÈTRES DU FORAGE
-------------------
- Profondeur finale: {prof_finale_input} m
- Azimuth initial: {azimuth_initial_input}°
- Inclinaison initiale: {inclinaison_initiale_input}°
- Lithologie: {lithologie_input}
- Entreprise de forage: {company_input}
- Vitesse de rotation: {vitesse_rotation_input} tr/min

RÉSULTATS DE LA PRÉDICTION
------------------------
- Déviation d'azimuth: {predicted_azimuth:.2f}°
- Déviation d'inclinaison: {predicted_inclinaison:.2f}°
- Azimuth final prévu: {azimuth_final:.2f}°
- Inclinaison finale prévue: {inclinaison_final:.2f}°
- Écart final estimé: {final_deviation:.2f} m

RECOMMANDATIONS
-------------
{recommendations}

CONCLUSION
---------
La déviation prédite est {deviation_text}, avec un impact {deviation_impact} sur la position finale du forage.
Ces prédictions sont basées sur le modèle {model_option} et sont fournies à titre indicatif.
Des ajustements doivent être apportés en fonction des conditions spécifiques du site et de l'expérience de l'équipe de forage.

===========================================
Rapport généré par l'application "Prédiction de Déviation des Forages Miniers"
Didier Ouedraogo, P.Geo. - Géologue & Data Scientist
            """
            
            # Créer un bouton de téléchargement
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.download_button(
                    label="📄 Télécharger le rapport",
                    data=report_text,
                    file_name=f"rapport_deviation_forage_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    
    # Ajouter un pied de page avec les informations d'auteur
    st.markdown("""
    <div class="footer">
        <p>© 2023-2025 Didier Ouedraogo, P.Geo. Tous droits réservés.</p>
        <p style="font-size: 0.85rem; margin-top: 0.5rem;">Application développée par Didier Ouedraogo, Géologue & Data Scientist</p>
    </div>
    """, unsafe_allow_html=True)

else:
    # Message pour guider l'utilisateur si aucune donnée n'est encore chargée
    if data_option == "Charger mes données" and not st.session_state.columns_mapped and st.session_state.raw_df is None:
        st.markdown("""
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 70vh; text-align: center;">
            <div style="font-size: 4rem; margin-bottom: 1.5rem; color: #E2E8F0;">
                📊
            </div>
            <h2 style="font-family: 'Poppins', sans-serif; font-weight: 700; color: #1A2C55; margin-bottom: 1.5rem;">Bienvenue dans l'application de prédiction de déviation des forages miniers</h2>
            <p style="max-width: 650px; margin: 1rem auto; font-size: 1.1rem; color: #4A5568; line-height: 1.7;">
                Veuillez charger un fichier CSV depuis la barre latérale pour commencer l'analyse et la modélisation.
            </p>
            <div style="background-color: #EFF6FF; padding: 1.5rem; border-radius: 10px; max-width: 650px; margin-top: 1.5rem; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05); border: 1px solid #DBEAFE;">
                <p style="margin: 0; color: #1E40AF; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.5rem;">
                    <strong>Format attendu</strong>: Un fichier CSV contenant des données sur les paramètres de forage et les déviations mesurées.
                </p>
                <p style="margin: 0; color: #3B82F6; font-size: 0.95rem;">
                    Vous pourrez mapper vos colonnes aux données requises par l'application après le chargement.
                </p>
            </div>
            <div style="margin-top: 2.5rem; display: flex; align-items: center; flex-direction: column;">
                <p style="color: #64748B; font-size: 1.05rem; margin-bottom: 1rem;">
                    ou sélectionnez "Utiliser données démo" pour explorer l'application avec des données synthétiques.
                </p>
                <div class="author-signature" style="margin-top: 3rem;">
                    <div class="author-avatar">DO</div>
                    <div class="author-details">
                        <div class="author-name-small">Didier Ouedraogo, P.Geo.</div>
                        <div class="author-title-small">Géologue & Data Scientist</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)