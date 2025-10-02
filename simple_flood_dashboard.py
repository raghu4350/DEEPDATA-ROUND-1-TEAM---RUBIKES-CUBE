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

# Page configuration
st.set_page_config(
    page_title="🌊 Urban Flood Risk Analytics Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        animation: fadeInUp 1s ease-out;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInUp 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.3);
    }
    
    .insight-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #007bff;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        animation: fadeIn 1s ease-out;
    }
    
    .eda-question-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 2px solid #e9ecef;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .eda-question-card:hover {
        border-color: #007bff;
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.3);
        animation: slideInRight 0.8s ease-out;
    }
    
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
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
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
    
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    .sidebar-metric {
        background: linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
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

# Enhanced data loading with more preprocessing
@st.cache_data
def load_and_process_data():
    """Load and comprehensively preprocess the flood risk dataset"""
    try:
        df = pd.read_csv('urban_pluvial_flood_risk_cleaned.csv')
        
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
            'event_': 2  # Events get moderate weight
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
            return score
        
        df['risk_severity_score'] = df['risk_labels'].apply(calculate_enhanced_risk_score)
        
        # Risk categories
        df['risk_category'] = pd.cut(df['risk_severity_score'], 
                                   bins=[0, 2, 5, 8, float('inf')], 
                                   labels=['Low', 'Medium', 'High', 'Critical'])
        
        # Extract categorical features
        land_use_cols = [col for col in df.columns if col.startswith('land_use_')]
        df['primary_land_use'] = df[land_use_cols].idxmax(axis=1).str.replace('land_use_', '')
        
        soil_cols = [col for col in df.columns if col.startswith('soil_group_')]
        df['soil_type'] = df[soil_cols].idxmax(axis=1).str.replace('soil_group_', '')
        
        drain_cols = [col for col in df.columns if col.startswith('storm_drain_type_')]
        df['drain_type'] = df[drain_cols].idxmax(axis=1).str.replace('storm_drain_type_', '')
        
        # Geographic regions
        df['region'] = df['city_name'].str.extract(r', (.+)$')[0]
        df['region'] = df['region'].fillna('Unknown')
        
        # Infrastructure efficiency score
        df['infrastructure_score'] = (
            (df['drainage_density_km_per_km2'] - df['drainage_density_km_per_km2'].min()) / 
            (df['drainage_density_km_per_km2'].max() - df['drainage_density_km_per_km2'].min()) * 50 +
            (1 - (df['storm_drain_proximity_m'] - df['storm_drain_proximity_m'].min()) / 
             (df['storm_drain_proximity_m'].max() - df['storm_drain_proximity_m'].min())) * 50
        )
        
        # Climate vulnerability index
        df['climate_vulnerability'] = (
            (df['historical_rainfall_intensity_mm_hr'] - df['historical_rainfall_intensity_mm_hr'].min()) / 
            (df['historical_rainfall_intensity_mm_hr'].max() - df['historical_rainfall_intensity_mm_hr'].min()) * 40 +
            (df['return_period_years'] - df['return_period_years'].min()) / 
            (df['return_period_years'].max() - df['return_period_years'].min()) * 30 +
            (1 - (df['elevation_m'] - df['elevation_m'].min()) / 
             (df['elevation_m'].max() - df['elevation_m'].min())) * 30
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_and_process_data()

if df is None:
    st.error("❌ Failed to load data. Please ensure 'urban_pluvial_flood_risk_cleaned.csv' exists in the current directory.")
    st.stop()

# Enhanced sidebar with animations
with st.sidebar:
    # Animated header
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 4rem; animation: bounce 2s infinite;">🌊</div>
        <h2 style="color: #1f77b4; margin: 0;">Flood Analytics</h2>
        <p style="color: #666; font-size: 0.9rem;">Global Urban Resilience Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced navigation menu
    selected = option_menu(
        menu_title="🧭 Navigation",
        options=["🏠 Executive Summary", "📊 EDA Deep Dive", "🗺️ Geographic Intelligence", 
                "⚠️ Risk Matrix", "🎬 Animated Analytics", "🎯 Strategic Insights"],
        icons=["house-fill", "bar-chart-fill", "geo-alt-fill", "exclamation-triangle-fill", 
               "play-circle-fill", "lightbulb-fill"],
        menu_icon="compass",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f8f9fa"},
            "icon": {"color": "#1f77b4", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "2px", 
                        "border-radius": "10px", "padding": "10px"},
            "nav-link-selected": {"background-color": "#007bff", "color": "white"},
        }
    )
    
    st.markdown("---")
    
    # Enhanced filters with metrics
    st.markdown("### 🔍 Smart Filters")
    
    # City filter with counts
    city_counts = df['city_name'].value_counts()
    city_options = ['🌍 All Cities'] + [f"{city} ({count:,})" for city, count in city_counts.items()]
    selected_city_display = st.selectbox("🏙️ Select City", city_options)
    selected_city = 'All' if selected_city_display.startswith('🌍') else selected_city_display.split(' (')[0]
    
    # Risk category filter
    risk_categories = ['All'] + df['risk_category'].dropna().unique().tolist()
    selected_risk_category = st.selectbox("⚠️ Risk Level", risk_categories)
    
    # Advanced filters in expander
    with st.expander("🔧 Advanced Filters"):
        # Risk severity range
        risk_range = st.slider("Risk Severity Score", 
                              int(df['risk_severity_score'].min()), 
                              int(df['risk_severity_score'].max()), 
                              (int(df['risk_severity_score'].min()), int(df['risk_severity_score'].max())))
        
        # Land use filter
        land_uses = ['All'] + sorted(df['primary_land_use'].unique().tolist())
        selected_land_use = st.selectbox("🏘️ Land Use Type", land_uses)
        
        # Infrastructure score filter
        infra_range = st.slider("Infrastructure Score", 0.0, 100.0, (0.0, 100.0))
        
        # Climate vulnerability filter
        climate_range = st.slider("Climate Vulnerability", 0.0, 100.0, (0.0, 100.0))
    
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
    
    # Sidebar metrics
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
    # Animated header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        lottie_flood = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_kkflmtur.json")
        if lottie_flood:
            st_lottie(lottie_flood, height=200, key="flood_animation")
    
    st.markdown('<h1 class="main-header">🌊 Urban Pluvial Flood Risk Analytics</h1>', 
                unsafe_allow_html=True)
    
    # Mission statement with enhanced styling
    st.markdown("""
    <div class="insight-box">
    <h3>🎯 Global Urban Resilience Consortium</h3>
    <p style="font-size: 1.1rem; line-height: 1.6;">
    As data analysts for a global urban resilience consortium, we leverage advanced analytics to understand 
    urban pluvial (rainfall-driven) flood risk patterns across global cities. Our mission is to provide 
    <strong>actionable insights</strong> that guide strategies to reduce flood risks, enhance drainage systems, 
    and build comprehensive city-wide resilience against climate change impacts.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced key metrics with animations
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = [
        (len(filtered_df['city_name'].unique()), "Cities Analyzed", "🏙️"),
        (f"{len(filtered_df):,}", "Urban Segments", "🏘️"),
        (len(filtered_df[filtered_df['risk_category'] == 'Critical']), "Critical Areas", "🚨"),
        (f"{filtered_df['elevation_m'].mean():.1f}m", "Avg Elevation", "⛰️"),
        (f"{filtered_df['infrastructure_score'].mean():.0f}%", "Infra Score", "🏗️")
    ]
    
    for i, (col, (value, label, icon)) in enumerate(zip([col1, col2, col3, col4, col5], metrics)):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="animation-delay: {i*0.1}s;">
                <div style="font-size: 2rem;">{icon}</div>
                <h3 style="margin: 0.5rem 0;">{value}</h3>
                <p style="margin: 0; opacity: 0.9;">{label}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Executive dashboard with enhanced visualizations
    st.markdown('<h2 class="sub-header">📊 Executive Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced risk distribution with custom colors
        risk_dist = filtered_df['risk_category'].value_counts()
        colors = ['#28a745', '#ffc107', '#fd7e14', '#dc3545']  # Green, Yellow, Orange, Red
        
        fig_risk = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            title="🎯 Risk Category Distribution",
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig_risk.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            textfont_size=12,
            marker=dict(line=dict(color='#FFFFFF', width=2))
        )
        fig_risk.update_layout(
            font=dict(size=14),
            title_font_size=18,
            showlegend=True,
            height=400
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Top cities by risk with enhanced styling
        city_risk = filtered_df.groupby('city_name').agg({
            'risk_severity_score': 'mean',
            'segment_id': 'count'
        }).round(2).sort_values('risk_severity_score', ascending=False).head(10)
        
        fig_cities = px.bar(
            x=city_risk['risk_severity_score'],
            y=city_risk.index,
            orientation='h',
            title="🏙️ Top 10 Cities by Average Risk Score",
            color=city_risk['risk_severity_score'],
            color_continuous_scale="Reds",
            text=city_risk['risk_severity_score'].round(1)
        )
        fig_cities.update_traces(textposition='inside')
        fig_cities.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=400,
            font=dict(size=12),
            title_font_size=18
        )
        
        st.plotly_chart(fig_cities, use_container_width=True)

elif selected == "📊 EDA Deep Dive":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis Deep Dive</h1>', 
                unsafe_allow_html=True)
    
    # EDA Questions with enhanced interactivity
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
    
    # Question selector with categories
    categories = list(set([q["category"] for q in eda_questions]))
    selected_category = st.selectbox("🔍 Filter by Category", ["All"] + categories)
    
    if selected_category != "All":
        filtered_questions = [q for q in eda_questions if q["category"] == selected_category]
    else:
        filtered_questions = eda_questions
    
    question_options = [f"{q['icon']} {q['question']}" for q in filtered_questions]
    selected_question_display = st.selectbox("📋 Select EDA Question to Explore:", question_options)
    
    # Find selected question
    selected_question_id = None
    for q in filtered_questions:
        if selected_question_display.endswith(q['question']):
            selected_question_id = q['id']
            break
    
    # Question analysis based on selection
    if selected_question_id == 1:  # Elevation vs Risk
        st.markdown('<h2 class="sub-header">🏔️ Elevation vs Risk Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced scatter plot with regression
            fig_scatter = px.scatter(
                filtered_df,
                x='elevation_m',
                y='risk_severity_score',
                color='city_name',
                size='historical_rainfall_intensity_mm_hr',
                hover_data=['admin_ward', 'primary_land_use', 'risk_category'],
                title="🏔️ Elevation vs Risk Severity by City",
                trendline="ols",
                trendline_scope="overall"
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Elevation distribution by risk category
            fig_violin = px.violin(
                filtered_df,
                x='risk_category',
                y='elevation_m',
                box=True,
                title="🎻 Elevation Distribution by Risk Category",
                color='risk_category',
                color_discrete_sequence=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
            )
            fig_violin.update_layout(height=500)
            st.plotly_chart(fig_violin, use_container_width=True)

elif selected == "🗺️ Geographic Intelligence":
    st.markdown('<h1 class="main-header">🗺️ Geographic Intelligence Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Enhanced geographic analysis
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Interactive risk map with multiple layers
        fig_map = px.scatter_mapbox(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color='risk_severity_score',
            size='climate_vulnerability',
            hover_name='city_name',
            hover_data=['admin_ward', 'primary_land_use', 'risk_category', 'infrastructure_score'],
            color_continuous_scale="Reds",
            size_max=20,
            zoom=1,
            title="🌍 Global Flood Risk Intelligence Map",
            mapbox_style="open-street-map"
        )
        fig_map.update_layout(height=700)
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Geographic Insights")
        
        # Regional analysis
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
    <div class="insight-box">
    <h3>🎭 Interactive Data Stories</h3>
    <p>Experience your flood risk data through animated visualizations that reveal patterns and insights over time and across dimensions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Animation selector
    animation_type = st.selectbox(
        "🎬 Choose Animation Type",
        [
            "🌊 Risk Analysis Journey",
            "⏰ Time Series Evolution", 
            "🗺️ Geographic Risk Flow"
        ]
    )
    
    if "Risk Analysis Journey" in animation_type:
        st.markdown('<h2 class="sub-header">🌊 Animated Risk Analysis Journey</h2>', unsafe_allow_html=True)
        
        # Create sample data for animation
        sample_df = filtered_df.sample(min(100, len(filtered_df)))
        
        fig = px.scatter(
            sample_df,
            x='elevation_m',
            y='risk_severity_score',
            color='primary_land_use',
            size='historical_rainfall_intensity_mm_hr',
            animation_frame='city_name',
            hover_data=['admin_ward'],
            title="Risk Analysis Animation - Elevation vs Risk by City"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    elif "Time Series Evolution" in animation_type:
        st.markdown('<h2 class="sub-header">⏰ Time Series Risk Evolution</h2>', unsafe_allow_html=True)
        
        # Create synthetic time series data
        cities = filtered_df['city_name'].unique()[:8]
        years = list(range(2020, 2025))
        
        time_data = []
        np.random.seed(42)
        
        for city in cities:
            city_base_risk = filtered_df[filtered_df['city_name'] == city]['risk_severity_score'].mean()
            
            for year in years:
                # Add yearly trend and random variation
                yearly_factor = 1 + 0.05 * (year - 2020)
                random_factor = 1 + np.random.normal(0, 0.1)
                
                risk_value = city_base_risk * yearly_factor * random_factor
                
                time_data.append({
                    'City': city,
                    'Year': year,
                    'Risk Score': max(0, risk_value)
                })
        
        time_df = pd.DataFrame(time_data)
        
        # Create animated line chart
        fig = px.line(
            time_df,
            x='Year',
            y='Risk Score',
            color='City',
            title="Risk Score Evolution Over Time (2020-2024)"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown('<h1 class="main-header">🎯 Strategic Insights & Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # Generate comprehensive insights
    insights = []
    
    # City insights
    city_risk = filtered_df.groupby('city_name')['risk_severity_score'].mean().sort_values(ascending=False)
    if len(city_risk) > 0:
        highest_risk_city = city_risk.index[0]
        insights.append({
            "title": "🏙️ Highest Risk City",
            "content": f"**{highest_risk_city}** shows the highest average flood risk score of {city_risk.iloc[0]:.2f}",
            "type": "warning"
        })
    
    # Display insights
    st.markdown('<h2 class="sub-header">💡 Key Insights</h2>', unsafe_allow_html=True)
    
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
        <h4>{insight['title']}</h4>
        <p>{insight['content']}</p>
        </div>
        """, unsafe_allow_html=True)

# Enhanced footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 30px; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 15px; margin-top: 2rem;'>
    <h3 style='color: #1f77b4; margin-bottom: 1rem;'>🌊 Urban Flood Risk Analytics Dashboard</h3>
    <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'>Built with Advanced Analytics | Streamlit • Plotly • Professional Insights</p>
    <p style='font-size: 0.9rem; opacity: 0.8;'>Empowering cities worldwide with data-driven flood resilience strategies</p>
</div>
""", unsafe_allow_html=True)
