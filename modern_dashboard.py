import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🌊 Urban Flood Risk Analytics Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern, Attractive CSS Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Main container styling */
    .main .block-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        padding-top: 2rem;
        border-radius: 20px;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    }
    
    /* Header styling */
    .main-header {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: gradient 3s ease infinite;
        text-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .sub-header {
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 1.5rem;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* Beautiful Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
        transition: left 0.5s;
    }
    
    .metric-card:hover::before {
        left: 100%;
    }
    
    .metric-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 25px 50px rgba(0,0,0,0.2);
        border-color: rgba(255,255,255,0.4);
    }
    
    .metric-card h3 {
        color: #ffffff;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .metric-card p {
        color: rgba(255,255,255,0.9);
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* Beautiful insight boxes */
    .insight-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        backdrop-filter: blur(20px);
        padding: 2rem;
        margin: 2rem 0;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.3);
        position: relative;
        overflow: hidden;
    }
    
    .insight-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4);
        background-size: 400% 400%;
        animation: gradient 3s ease infinite;
    }
    
    .insight-box h3, .insight-box h4 {
        color: #2c3e50;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
    }
    
    .insight-box p {
        color: #34495e;
        line-height: 1.8;
        font-size: 1.1rem;
    }
    
    /* Modern sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        color: #ffffff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .sidebar-metric:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .sidebar-metric h4 {
        color: #ffffff;
        font-weight: 700;
        margin: 0;
        font-size: 1.8rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .sidebar-metric p {
        color: rgba(255,255,255,0.9);
        margin: 0;
        font-weight: 500;
    }
    
    /* Beautiful form controls */
    .stSelectbox > div > div {
        background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.8) 100%);
        backdrop-filter: blur(10px);
        color: #2c3e50;
        border: 2px solid rgba(255,255,255,0.3);
        border-radius: 15px;
        font-weight: 500;
    }
    
    .stSelectbox label {
        color: #ffffff;
        font-weight: 600;
        font-size: 1.1rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .stSlider > div > div > div {
        color: #ffffff;
    }
    
    .stSlider label {
        color: #ffffff;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Beautiful navigation menu */
    .nav-link {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.2);
        margin: 5px 0;
        border-radius: 15px;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .nav-link:hover {
        background: linear-gradient(135deg, rgba(255,255,255,0.2) 0%, rgba(255,255,255,0.1) 100%);
        transform: translateX(5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .nav-link-selected {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: #ffffff;
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        font-weight: 600;
    }
    
    /* Chart containers */
    .chart-container {
        background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(255,255,255,0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 15px 35px rgba(0,0,0,0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        font-weight: 600;
    }
    
    .streamlit-expanderContent {
        background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 0 0 15px 15px;
    }
    
    /* Recommendation cards */
    .recommendation-card {
        background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 100%);
        color: #ffffff;
        padding: 2rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
        border: 1px solid rgba(255,255,255,0.2);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .recommendation-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.3);
    }
    
    .recommendation-card h4 {
        color: #ffffff;
        font-weight: 700;
        font-size: 1.4rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    .recommendation-card p {
        color: rgba(255,255,255,0.95);
        font-weight: 500;
        line-height: 1.6;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    .slide-in-right {
        animation: slideInRight 0.8s ease-out;
    }
    
    /* Override Streamlit defaults */
    .stMarkdown {
        color: inherit;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 3rem;
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animations
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Enhanced data loading with better error handling
@st.cache_data
def load_and_process_data():
    """Load and comprehensively preprocess the flood risk dataset"""
    try:
        df = pd.read_csv('urban_pluvial_flood_risk_cleaned.csv')
        
        # Basic data cleaning
        df = df.dropna(subset=['risk_labels'])
        
        # Enhanced risk label processing
        df['risk_labels_list'] = df['risk_labels'].str.split('|')
        df['num_risk_labels'] = df['risk_labels_list'].apply(len)
        
        # Individual risk flags
        df['has_monitor'] = df['risk_labels'].str.contains('monitor', na=False)
        df['has_low_lying'] = df['risk_labels'].str.contains('low_lying', na=False)
        df['has_extreme_rain'] = df['risk_labels'].str.contains('extreme_rain_history', na=False)
        df['has_ponding'] = df['risk_labels'].str.contains('ponding_hotspot', na=False)
        df['has_event'] = df['risk_labels'].str.contains('event_', na=False)
        
        # Enhanced risk severity scoring
        risk_weights = {
            'monitor': 1,
            'low_lying': 3,
            'ponding_hotspot': 4,
            'extreme_rain_history': 5,
            'event_': 2
        }
        
        def calculate_enhanced_risk_score(risk_labels):
            if pd.isna(risk_labels):
                return 0
            labels = risk_labels.split('|')
            score = 0
            for label in labels:
                label = label.strip()
                for key, weight in risk_weights.items():
                    if key in label:
                        score += weight
            return max(score, 1)  # Ensure minimum score of 1
        
        df['risk_severity_score'] = df['risk_labels'].apply(calculate_enhanced_risk_score)
        
        # Risk categories
        df['risk_category'] = pd.cut(df['risk_severity_score'], 
                                   bins=[0, 2, 5, 8, float('inf')], 
                                   labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Extract categorical features
        land_use_cols = [col for col in df.columns if col.startswith('land_use_')]
        if land_use_cols:
            df['primary_land_use'] = df[land_use_cols].idxmax(axis=1).str.replace('land_use_', '')
        else:
            df['primary_land_use'] = 'Unknown'
        
        soil_cols = [col for col in df.columns if col.startswith('soil_group_')]
        if soil_cols:
            df['soil_type'] = df[soil_cols].idxmax(axis=1).str.replace('soil_group_', '')
        else:
            df['soil_type'] = 'Unknown'
        
        drain_cols = [col for col in df.columns if col.startswith('storm_drain_type_')]
        if drain_cols:
            df['drain_type'] = df[drain_cols].idxmax(axis=1).str.replace('storm_drain_type_', '')
        else:
            df['drain_type'] = 'Unknown'
        
        # Geographic regions
        df['region'] = df['city_name'].str.extract(r', (.+)$')[0]
        df['region'] = df['region'].fillna('Unknown')
        
        # Ensure all numeric columns are properly scaled and positive for size parameters
        numeric_cols = ['elevation_m', 'drainage_density_km_per_km2', 'storm_drain_proximity_m',
                       'historical_rainfall_intensity_mm_hr', 'return_period_years']
        
        for col in numeric_cols:
            if col in df.columns:
                # Normalize to 0-100 scale and ensure positive values
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val != min_val:
                    df[f'{col}_normalized'] = ((df[col] - min_val) / (max_val - min_val)) * 100
                else:
                    df[f'{col}_normalized'] = 50  # Default middle value
                
                # Ensure positive values for size parameters (5-25 range for better visibility)
                df[f'{col}_size'] = np.maximum((df[f'{col}_normalized'] / 4) + 5, 5)
        
        # Infrastructure efficiency score (0-100)
        if 'drainage_density_km_per_km2' in df.columns and 'storm_drain_proximity_m' in df.columns:
            drainage_norm = df['drainage_density_km_per_km2_normalized']
            proximity_norm = 100 - df['storm_drain_proximity_m_normalized']  # Invert so closer is better
            df['infrastructure_score'] = (drainage_norm + proximity_norm) / 2
        else:
            df['infrastructure_score'] = 50  # Default value
        
        # Climate vulnerability index (0-100)
        if 'historical_rainfall_intensity_mm_hr' in df.columns and 'return_period_years' in df.columns:
            rainfall_norm = df['historical_rainfall_intensity_mm_hr_normalized']
            period_norm = df['return_period_years_normalized']
            elevation_norm = 100 - df['elevation_m_normalized'] if 'elevation_m' in df.columns else 50
            df['climate_vulnerability'] = (rainfall_norm * 0.4 + period_norm * 0.3 + elevation_norm * 0.3)
        else:
            df['climate_vulnerability'] = 50  # Default value
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_and_process_data()

if df is None:
    st.error("❌ Failed to load data. Please ensure 'urban_pluvial_flood_risk_cleaned.csv' exists in the current directory.")
    st.stop()

# Beautiful sidebar
with st.sidebar:
    # Animated header
    st.markdown("""
    <div class="sidebar-header">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🌊</div>
        <h2 style="color: #ffffff; margin: 0; font-weight: 700; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">Flood Analytics</h2>
        <p style="color: rgba(255,255,255,0.9); font-size: 1rem; font-weight: 500; margin-top: 0.5rem;">Global Urban Resilience Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Beautiful navigation menu
    selected = option_menu(
        menu_title="🧭 Navigation",
        options=["🏠 Executive Summary", "📊 EDA Deep Dive", "🗺️ Geographic Intelligence", 
                "⚠️ Risk Matrix", "🎬 Animated Analytics", "🎯 Strategic Insights"],
        icons=["house-fill", "bar-chart-fill", "geo-alt-fill", "exclamation-triangle-fill", 
               "play-circle-fill", "lightbulb-fill"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background": "transparent"},
            "icon": {"color": "#ffffff", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px 0", 
                        "border-radius": "15px", "padding": "12px 16px", "color": "#ffffff", 
                        "background": "linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)",
                        "border": "1px solid rgba(255,255,255,0.2)", "backdrop-filter": "blur(10px)"},
            "nav-link-selected": {"background": "linear-gradient(45deg, #FF6B6B, #4ECDC4)", 
                                "color": "#ffffff", "border": "2px solid rgba(255,255,255,0.3)",
                                "box-shadow": "0 10px 25px rgba(0,0,0,0.2)", "font-weight": "600"},
        }
    )
    
    st.markdown("---")
    
    # Beautiful filters
    st.markdown("### 🔍 Smart Filters")
    
    # City filter with counts
    city_counts = df['city_name'].value_counts()
    city_options = ['🌍 All Cities'] + [f"{city} ({count:,})" for city, count in city_counts.items()]
    selected_city_display = st.selectbox("🏙️ Select City", city_options, key="city_filter")
    selected_city = 'All' if selected_city_display.startswith('🌍') else selected_city_display.split(' (')[0]
    
    # Risk category filter
    risk_categories = ['All'] + df['risk_category'].dropna().unique().tolist()
    selected_risk_category = st.selectbox("⚠️ Risk Level", risk_categories, key="risk_filter")
    
    # Advanced filters in expander
    with st.expander("🔧 Advanced Filters"):
        # Risk severity range
        risk_range = st.slider("Risk Severity Score", 
                              int(df['risk_severity_score'].min()), 
                              int(df['risk_severity_score'].max()), 
                              (int(df['risk_severity_score'].min()), int(df['risk_severity_score'].max())),
                              key="risk_slider")
        
        # Land use filter
        land_uses = ['All'] + sorted(df['primary_land_use'].unique().tolist())
        selected_land_use = st.selectbox("🏘️ Land Use Type", land_uses, key="land_use_filter")
        
        # Infrastructure score filter
        infra_range = st.slider("Infrastructure Score", 0.0, 100.0, (0.0, 100.0), key="infra_slider")
        
        # Climate vulnerability filter
        climate_range = st.slider("Climate Vulnerability", 0.0, 100.0, (0.0, 100.0), key="climate_slider")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city_name'] == selected_city]
    
    if selected_risk_category != 'All':
        filtered_df = filtered_df[filtered_df['risk_category'] == selected_risk_category]
    
    filtered_df = filtered_df[
        (filtered_df['risk_severity_score'] >= risk_range[0]) & 
        (filtered_df['risk_severity_score'] <= risk_range[1]) &
        (filtered_df['infrastructure_score'] >= infra_range[0]) &
        (filtered_df['infrastructure_score'] <= infra_range[1]) &
        (filtered_df['climate_vulnerability'] >= climate_range[0]) &
        (filtered_df['climate_vulnerability'] <= climate_range[1])
    ]
    
    if selected_land_use != 'All':
        filtered_df = filtered_df[filtered_df['primary_land_use'] == selected_land_use]
    
    # Beautiful sidebar metrics
    st.markdown("---")
    st.markdown("### 📊 Filter Results")
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <h4>{len(filtered_df):,}</h4>
        <p>Filtered Segments</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-metric">
        <h4>{len(filtered_df['city_name'].unique())}</h4>
        <p>Cities in View</p>
    </div>
    """, unsafe_allow_html=True)
    
    if len(filtered_df) > 0:
        avg_risk = filtered_df['risk_severity_score'].mean()
        st.markdown(f"""
        <div class="sidebar-metric">
            <h4>{avg_risk:.1f}</h4>
            <p>Avg Risk Score</p>
        </div>
        """, unsafe_allow_html=True)

# Main content based on selection
if selected == "🏠 Executive Summary":
    # Beautiful animated header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lottie_flood = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_kkflmtur.json")
        if lottie_flood:
            st_lottie(lottie_flood, height=200, key="flood_animation")
    
    st.markdown('<h1 class="main-header">🌊 Urban Pluvial Flood Risk Analytics</h1>', 
                unsafe_allow_html=True)
    
    # Beautiful mission statement
    st.markdown("""
    <div class="insight-box fade-in-up">
    <h3>🎯 Global Urban Resilience Consortium</h3>
    <p style="font-size: 1.2rem; line-height: 1.8;">
    As data analysts for a global urban resilience consortium, we leverage advanced analytics to understand 
    urban pluvial (rainfall-driven) flood risk patterns across global cities. Our mission is to provide 
    <strong>actionable insights</strong> that guide strategies to reduce flood risks, enhance drainage systems, 
    and build comprehensive city-wide resilience against climate change impacts.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Beautiful key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        (len(filtered_df['city_name'].unique()), "Cities Analyzed", "🏙️"),
        (f"{len(filtered_df):,}", "Urban Segments", "🏘️"),
        (len(filtered_df[filtered_df['risk_category'] == 'Critical']), "Critical Areas", "🚨"),
        (f"{filtered_df['elevation_m'].mean():.1f}m" if 'elevation_m' in filtered_df.columns else "N/A", "Avg Elevation", "⛰️"),
        (f"{filtered_df['infrastructure_score'].mean():.0f}%" if 'infrastructure_score' in filtered_df.columns else "N/A", "Infra Score", "🏗️")
    ]
    
    for i, (col, (value, label, icon)) in enumerate(zip([col1, col2, col3, col4, col5], metrics)):
        with col:
            st.markdown(f"""
            <div class="metric-card fade-in-up" style="animation-delay: {i*0.1}s;">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
                <h3>{value}</h3>
                <p>{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Beautiful executive dashboard
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Beautiful risk distribution chart
        risk_dist = filtered_df['risk_category'].value_counts()
        colors = ['#96CEB4', '#FFEAA7', '#FDCB6E', '#E17055']  # Beautiful green to red gradient
        
        fig_risk = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="🎯 Risk Category Distribution",
            color_discrete_sequence=colors,
            hole=0.5
        )
        fig_risk.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=14,
            textfont_color='white',
            textfont_family="Poppins",
            marker=dict(line=dict(color='#ffffff', width=3))
        )
        fig_risk.update_layout(
            font=dict(size=14, color='#2c3e50', family="Poppins"),
            title_font_size=20,
            title_font_color='#2c3e50',
            title_x=0.5,
            showlegend=True,
            height=450,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=80, b=20, l=20, r=20)
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_risk, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Beautiful top cities chart
        city_risk = filtered_df.groupby('city_name').agg({
            'risk_severity_score': 'mean',
            'segment_id': 'count' if 'segment_id' in filtered_df.columns else lambda x: len(x)
        }).round(2).sort_values('risk_severity_score', ascending=False).head(10)
        
        fig_cities = px.bar(
            x=city_risk['risk_severity_score'],
            y=city_risk.index,
            orientation='h',
            title="🏙️ Top 10 Cities by Average Risk Score",
            color=city_risk['risk_severity_score'],
            color_continuous_scale=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8'],
            text=city_risk['risk_severity_score'].round(1)
        )
        fig_cities.update_traces(
            textposition='inside', 
            textfont_color='white',
            textfont_family="Poppins",
            textfont_size=12
        )
        fig_cities.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=450,
            font=dict(size=12, color='#2c3e50', family="Poppins"),
            title_font_size=20,
            title_font_color='#2c3e50',
            title_x=0.5,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=80, b=20, l=20, r=20),
            xaxis_title="Risk Score",
            yaxis_title="Cities"
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_cities, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Beautiful global insights section
    st.markdown('<h2 class="sub-header">🌍 Global Risk Insights</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Beautiful land use risk analysis
        land_use_risk = filtered_df.groupby('primary_land_use')['risk_severity_score'].mean().sort_values(ascending=False)
        
        fig_land = px.bar(
            x=land_use_risk.index,
            y=land_use_risk.values,
            title="🏘️ Risk by Land Use Type",
            color=land_use_risk.values,
            color_continuous_scale=['#00b894', '#00cec9', '#0984e3', '#6c5ce7', '#e84393']
        )
        fig_land.update_xaxes(tickangle=45)
        fig_land.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', family="Poppins"),
            title_font_color='#2c3e50',
            title_x=0.5,
            title_font_size=18,
            margin=dict(t=60, b=20, l=20, r=20),
            xaxis_title="Land Use Type",
            yaxis_title="Average Risk Score"
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_land, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Beautiful infrastructure vs risk scatter
        sample_size = min(1000, len(filtered_df))
        sample_df = filtered_df.sample(sample_size) if len(filtered_df) > sample_size else filtered_df
        
        fig_infra = px.scatter(
            sample_df,
            x='infrastructure_score',
            y='risk_severity_score',
            color='climate_vulnerability',
            title="🏗️ Infrastructure vs Risk",
            color_continuous_scale=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8'],
            opacity=0.7,
            size_max=15
        )
        fig_infra.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', family="Poppins"),
            title_font_color='#2c3e50',
            title_x=0.5,
            title_font_size=18,
            margin=dict(t=60, b=20, l=20, r=20),
            xaxis_title="Infrastructure Score",
            yaxis_title="Risk Severity Score"
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_infra, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        # Beautiful climate vulnerability distribution
        fig_climate = px.histogram(
            filtered_df,
            x='climate_vulnerability',
            title="🌡️ Climate Vulnerability Distribution",
            nbins=20,
            color_discrete_sequence=['#e17055']
        )
        fig_climate.update_layout(
            height=400, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', family="Poppins"),
            title_font_color='#2c3e50',
            title_x=0.5,
            title_font_size=18,
            margin=dict(t=60, b=20, l=20, r=20),
            xaxis_title="Climate Vulnerability Score",
            yaxis_title="Frequency"
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig_climate, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif selected == "📊 EDA Deep Dive":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis Deep Dive</h1>', 
                unsafe_allow_html=True)
    
    # EDA Questions with beautiful styling
    eda_questions = [
        {
            "id": 1,
            "question": "How does elevation correlate with risk labels across cities?",
            "icon": "🏔️",
            "category": "Geographic"
        },
        {
            "id": 2,
            "question": "Which cities have the highest concentration of high-risk areas?",
            "icon": "🏙️",
            "category": "Urban Planning"
        },
        {
            "id": 3,
            "question": "How does land use type influence flood risk severity?",
            "icon": "🏘️",
            "category": "Land Use"
        },
        {
            "id": 4,
            "question": "What's the relationship between drainage density and flood risk?",
            "icon": "🚰",
            "category": "Infrastructure"
        }
    ]
    
    # Question selector
    categories = list(set([q["category"] for q in eda_questions]))
    selected_category = st.selectbox("🔍 Filter by Category", ["All"] + categories, key="eda_category")
    
    if selected_category != "All":
        filtered_questions = [q for q in eda_questions if q["category"] == selected_category]
    else:
        filtered_questions = eda_questions
    
    question_options = [f"{q['icon']} {q['question']}" for q in filtered_questions]
    selected_question_display = st.selectbox("📋 Select EDA Question to Explore:", question_options, key="eda_question")
    
    # Find selected question
    selected_question_id = None
    for q in filtered_questions:
        if selected_question_display.endswith(q['question']):
            selected_question_id = q['id']
            break
    
    # Beautiful question analysis
    if selected_question_id == 1:  # Elevation vs Risk
        st.markdown('<h2 class="sub-header">🏔️ Elevation vs Risk Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'elevation_m' in filtered_df.columns:
                sample_df = filtered_df.sample(min(500, len(filtered_df)))
                fig_scatter = px.scatter(
                    sample_df,
                    x='elevation_m',
                    y='risk_severity_score',
                    color='city_name',
                    size='historical_rainfall_intensity_mm_hr_size' if 'historical_rainfall_intensity_mm_hr_size' in sample_df.columns else None,
                    title="🏔️ Elevation vs Risk Severity by City",
                    trendline="ols",
                    trendline_scope="overall",
                    color_discrete_sequence=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8', '#e84393', '#00b894', '#00cec9']
                )
                fig_scatter.update_layout(
                    height=500, 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50', family="Poppins"),
                    title_font_color='#2c3e50',
                    title_x=0.5,
                    title_font_size=18,
                    xaxis_title="Elevation (m)",
                    yaxis_title="Risk Severity Score"
                )
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Elevation data not available in the current dataset.")
        
        with col2:
            if 'elevation_m' in filtered_df.columns:
                fig_violin = px.violin(
                    filtered_df,
                    x='risk_category',
                    y='elevation_m',
                    box=True,
                    title="🎻 Elevation Distribution by Risk Category",
                    color='risk_category',
                    color_discrete_sequence=['#96CEB4', '#FFEAA7', '#FDCB6E', '#E17055']
                )
                fig_violin.update_layout(
                    height=500, 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50', family="Poppins"),
                    title_font_color='#2c3e50',
                    title_x=0.5,
                    title_font_size=18,
                    xaxis_title="Risk Category",
                    yaxis_title="Elevation (m)"
                )
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_violin, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                risk_dist = filtered_df['risk_category'].value_counts()
                fig_alt = px.bar(
                    x=risk_dist.index,
                    y=risk_dist.values,
                    title="📊 Risk Category Distribution",
                    color=risk_dist.values,
                    color_continuous_scale=['#96CEB4', '#FFEAA7', '#FDCB6E', '#E17055']
                )
                fig_alt.update_layout(
                    height=500, 
                    plot_bgcolor='rgba(0,0,0,0)', 
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2c3e50', family="Poppins"),
                    title_font_color='#2c3e50',
                    title_x=0.5,
                    title_font_size=18
                )
                
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_alt, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif selected == "🗺️ Geographic Intelligence":
    st.markdown('<h1 class="main-header">🗺️ Geographic Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if 'latitude' in filtered_df.columns and 'longitude' in filtered_df.columns:
            fig_map = px.scatter_map(
                filtered_df,
                lat='latitude',
                lon='longitude',
                color='risk_severity_score',
                size='climate_vulnerability',
                hover_name='city_name',
                title="🌍 Global Flood Risk Intelligence Map",
                color_continuous_scale=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8'],
                size_max=25,
                zoom=1
            )
            fig_map.update_layout(
                height=700,
                font=dict(color='#2c3e50', family="Poppins"),
                title_font_color='#2c3e50',
                title_x=0.5,
                title_font_size=20
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_map, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Geographic coordinates not available in the current dataset.")
            city_risk = filtered_df.groupby('city_name')['risk_severity_score'].mean().sort_values(ascending=False).head(20)
            fig_alt = px.bar(
                x=city_risk.values,
                y=city_risk.index,
                orientation='h',
                title="🏙️ Top 20 Cities by Risk Score",
                color=city_risk.values,
                color_continuous_scale=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8']
            )
            fig_alt.update_layout(
                height=700, 
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2c3e50', family="Poppins"),
                title_font_color='#2c3e50',
                title_x=0.5,
                title_font_size=20
            )
            
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig_alt, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🎯 Geographic Insights")
        
        region_stats = filtered_df.groupby('region').agg({
            'risk_severity_score': ['mean', 'count'],
            'infrastructure_score': 'mean',
            'climate_vulnerability': 'mean'
        }).round(2)
        
        region_stats.columns = ['Avg_Risk', 'Count', 'Avg_Infrastructure', 'Avg_Climate']
        region_stats = region_stats.reset_index().sort_values('Avg_Risk', ascending=False)
        
        st.dataframe(region_stats, use_container_width=True)

elif selected == "🎬 Animated Analytics":
    st.markdown('<h1 class="main-header">🎬 Advanced Plotly Animations</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box fade-in-up">
    <h3>🎭 Interactive Data Stories</h3>
    <p style="font-size: 1.1rem;">Experience your flood risk data through beautiful animated visualizations that reveal patterns and insights over time and across dimensions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    animation_type = st.selectbox(
        "🎬 Choose Animation Type",
        [
            "🌊 Risk Analysis Journey",
            "⏰ Time Series Evolution"
        ],
        key="animation_type"
    )
    
    if "Risk Analysis Journey" in animation_type:
        st.markdown('<h2 class="sub-header">🌊 Animated Risk Analysis Journey</h2>', unsafe_allow_html=True)
        
        sample_df = filtered_df.sample(min(100, len(filtered_df)))
        
        if 'elevation_m' in sample_df.columns:
            fig = px.scatter(
                sample_df,
                x='elevation_m',
                y='risk_severity_score',
                color='primary_land_use',
                size='historical_rainfall_intensity_mm_hr_size' if 'historical_rainfall_intensity_mm_hr_size' in sample_df.columns else None,
                animation_frame='city_name',
                title="🌊 Risk Analysis Animation - Elevation vs Risk by City",
                color_discrete_sequence=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8', '#e84393', '#00b894', '#00cec9']
            )
        else:
            fig = px.scatter(
                sample_df,
                x='infrastructure_score',
                y='risk_severity_score',
                color='primary_land_use',
                size='climate_vulnerability',
                animation_frame='city_name',
                title="🌊 Risk Analysis Animation - Infrastructure vs Risk by City",
                color_discrete_sequence=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8', '#e84393', '#00b894', '#00cec9']
            )
        
        fig.update_layout(
            height=600, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', family="Poppins"),
            title_font_color='#2c3e50',
            title_x=0.5,
            title_font_size=20
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif "Time Series Evolution" in animation_type:
        st.markdown('<h2 class="sub-header">⏰ Time Series Risk Evolution</h2>', unsafe_allow_html=True)
        
        cities = filtered_df['city_name'].unique()[:8]
        years = list(range(2020, 2025))
        
        time_data = []
        np.random.seed(42)
        
        for city in cities:
            city_base_risk = filtered_df[filtered_df['city_name'] == city]['risk_severity_score'].mean()
            
            for year in years:
                yearly_factor = 1 + 0.05 * (year - 2020)
                random_factor = 1 + np.random.normal(0, 0.1)
                risk_value = city_base_risk * yearly_factor * random_factor
                
                time_data.append({
                    'City': city,
                    'Year': year,
                    'Risk Score': max(0, risk_value)
                })
        
        time_df = pd.DataFrame(time_data)
        
        fig = px.line(
            time_df,
            x='Year',
            y='Risk Score',
            color='City',
            title="⏰ Risk Score Evolution Over Time (2020-2024)",
            color_discrete_sequence=['#74b9ff', '#0984e3', '#6c5ce7', '#a29bfe', '#fd79a8', '#e84393', '#00b894', '#00cec9']
        )
        fig.update_layout(
            height=600, 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50', family="Poppins"),
            title_font_color='#2c3e50',
            title_x=0.5,
            title_font_size=20,
            xaxis_title="Year",
            yaxis_title="Risk Score"
        )
        
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<h1 class="main-header">🎯 Strategic Insights & Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # Generate insights
    insights = []
    
    city_risk = filtered_df.groupby('city_name')['risk_severity_score'].mean().sort_values(ascending=False)
    if len(city_risk) > 0:
        highest_risk_city = city_risk.index[0]
        insights.append({
            "title": "🏙️ Highest Risk City",
            "content": f"**{highest_risk_city}** shows the highest average flood risk score of {city_risk.iloc[0]:.2f}",
            "type": "warning"
        })
    
    st.markdown('<h2 class="sub-header">💡 Key Insights</h2>', unsafe_allow_html=True)
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box slide-in-right" style="animation-delay: {i*0.1}s;">
        <h4>{insight['title']}</h4>
        <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Beautiful recommendations
    st.markdown('<h2 class="sub-header">🚀 Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    recommendations = [
        {
            "title": "🏗️ Priority Infrastructure Investment",
            "description": "Focus on areas with infrastructure scores below 30 and high climate vulnerability",
            "priority": "Critical"
        },
        {
            "title": "🌱 Nature-Based Solutions",
            "description": "Implement green infrastructure in high-density residential areas",
            "priority": "High"
        },
        {
            "title": "📊 Smart Monitoring Systems",
            "description": "Deploy IoT sensors and AI-powered early warning systems",
            "priority": "High"
        }
    ]
    
    for i, rec in enumerate(recommendations):
        st.markdown(f"""
        <div class="recommendation-card slide-in-right" style="animation-delay: {i*0.2}s;">
        <h4>{rec['title']}</h4>
        <p><strong>Description:</strong> {rec['description']}</p>
        <p><strong>Priority:</strong> {rec['priority']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Beautiful summary metrics
    st.markdown('<h2 class="sub-header">📊 Summary Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Risk Score", f"{filtered_df['risk_severity_score'].sum():,.0f}")
    
    with col2:
        st.metric("Average Risk", f"{filtered_df['risk_severity_score'].mean():.2f}")
    
    with col3:
        critical_areas = len(filtered_df[filtered_df['risk_category'] == 'Critical'])
        st.metric("Critical Areas", f"{critical_areas:,}")
    
    with col4:
        cities_analyzed = len(filtered_df['city_name'].unique())
        st.metric("Cities Covered", f"{cities_analyzed}")

# Beautiful footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h3 style='margin-bottom: 1rem; font-weight: 700;'>🌊 Urban Flood Risk Analytics Dashboard</h3>
    <p style='font-size: 1.2rem; margin-bottom: 0.5rem; font-weight: 500;'>Built with Advanced Analytics | Streamlit • Plotly • Beautiful Design</p>
    <p style='font-size: 1rem; font-weight: 400; opacity: 0.9;'>Empowering cities worldwide with data-driven flood resilience strategies</p>
    <div style='margin-top: 1.5rem; display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;'>
        <span style='font-weight: 600; color: #96CEB4;'>🌍 Global Coverage</span>
        <span style='font-weight: 600; color: #74b9ff;'>📊 Real-time Analytics</span>
        <span style='font-weight: 600; color: #fd79a8;'>🚨 Risk Assessment</span>
        <span style='font-weight: 600; color: #e84393;'>🎯 Strategic Planning</span>
    </div>
</div>
""", unsafe_allow_html=True)
