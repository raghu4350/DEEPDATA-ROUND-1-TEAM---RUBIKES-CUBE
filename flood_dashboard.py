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
import json
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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 5px solid #007bff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 5px;
    }
    .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Load Lottie animation
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Cache data loading
@st.cache_data
def load_data():
    """Load and preprocess the flood risk dataset"""
    try:
        df = pd.read_csv('urban_pluvial_flood_risk_cleaned.csv')
        
        # Process risk labels
        df['risk_labels_list'] = df['risk_labels'].str.split('|')
        df['num_risk_labels'] = df['risk_labels_list'].apply(len)
        df['has_monitor'] = df['risk_labels'].str.contains('monitor')
        df['has_low_lying'] = df['risk_labels'].str.contains('low_lying')
        df['has_extreme_rain'] = df['risk_labels'].str.contains('extreme_rain_history')
        df['has_ponding'] = df['risk_labels'].str.contains('ponding_hotspot')
        
        # Create risk severity score
        risk_weights = {
            'monitor': 1,
            'low_lying': 2,
            'ponding_hotspot': 3,
            'extreme_rain_history': 4
        }
        
        def calculate_risk_score(risk_labels):
            labels = risk_labels.split('|')
            return sum(risk_weights.get(label.strip(), 0) for label in labels)
        
        df['risk_severity_score'] = df['risk_labels'].apply(calculate_risk_score)
        
        # Extract land use type
        land_use_cols = [col for col in df.columns if col.startswith('land_use_')]
        df['primary_land_use'] = df[land_use_cols].idxmax(axis=1).str.replace('land_use_', '')
        
        # Extract soil group
        soil_cols = [col for col in df.columns if col.startswith('soil_group_')]
        df['soil_type'] = df[soil_cols].idxmax(axis=1).str.replace('soil_group_', '')
        
        # Extract storm drain type
        drain_cols = [col for col in df.columns if col.startswith('storm_drain_type_')]
        df['drain_type'] = df[drain_cols].idxmax(axis=1).str.replace('storm_drain_type_', '')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data
df = load_data()

if df is None:
    st.error("Failed to load data. Please check if the CSV file exists.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/flood.png", width=80)
    st.title("🌊 Flood Analytics")
    
    # Navigation menu
    selected = option_menu(
        menu_title="Navigation",
        options=["🏠 Overview", "📊 EDA Questions", "🗺️ Geographic Analysis", 
                "⚠️ Risk Assessment", "📈 Advanced Analytics", "🎯 Insights"],
        icons=["house", "bar-chart", "geo-alt", "exclamation-triangle", 
               "graph-up", "lightbulb"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1f77b4", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
            "nav-link-selected": {"background-color": "#007bff"},
        }
    )
    
    st.markdown("---")
    
    # Filters
    st.subheader("🔍 Filters")
    
    # City filter
    cities = ['All'] + sorted(df['city_name'].unique().tolist())
    selected_city = st.selectbox("Select City", cities)
    
    # Risk severity filter
    risk_range = st.slider("Risk Severity Score", 
                          int(df['risk_severity_score'].min()), 
                          int(df['risk_severity_score'].max()), 
                          (int(df['risk_severity_score'].min()), int(df['risk_severity_score'].max())))
    
    # Land use filter
    land_uses = ['All'] + sorted(df['primary_land_use'].unique().tolist())
    selected_land_use = st.selectbox("Land Use Type", land_uses)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_city != 'All':
        filtered_df = filtered_df[filtered_df['city_name'] == selected_city]
    
    filtered_df = filtered_df[
        (filtered_df['risk_severity_score'] >= risk_range[0]) & 
        (filtered_df['risk_severity_score'] <= risk_range[1])
    ]
    
    if selected_land_use != 'All':
        filtered_df = filtered_df[filtered_df['primary_land_use'] == selected_land_use]
    
    st.markdown("---")
    st.markdown(f"**Filtered Records:** {len(filtered_df):,}")
    st.markdown(f"**Total Records:** {len(df):,}")

# Main content
if selected == "🏠 Overview":
    # Header with animation
    lottie_flood = load_lottie_url("https://assets5.lottiefiles.com/packages/lf20_kkflmtur.json")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if lottie_flood:
            st_lottie(lottie_flood, height=200, key="flood_animation")
    
    st.markdown('<h1 class="main-header">🌊 Urban Pluvial Flood Risk Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <h3>🎯 Mission Statement</h3>
    <p>As data analysts for a global urban resilience consortium, we analyze urban pluvial (rainfall-driven) 
    flood risk patterns across global cities. Our mission is to provide actionable insights that guide 
    strategies to reduce flood risks, enhance drainage systems, and build city-wide resilience.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(filtered_df['city_name'].unique())}</h3>
            <p>Cities Analyzed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>{len(filtered_df):,}</h3>
            <p>Urban Segments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        high_risk = len(filtered_df[filtered_df['risk_severity_score'] >= 6])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{high_risk:,}</h3>
            <p>High Risk Areas</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_elevation = filtered_df['elevation_m'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>{avg_elevation:.1f}m</h3>
            <p>Avg Elevation</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset overview
    st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_counts = filtered_df['risk_labels'].value_counts().head(10)
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Label Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # City distribution
        city_counts = filtered_df['city_name'].value_counts().head(10)
        fig_bar = px.bar(
            x=city_counts.values,
            y=city_counts.index,
            orientation='h',
            title="Top 10 Cities by Segments",
            color=city_counts.values,
            color_continuous_scale="viridis"
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Data quality summary
    st.markdown('<h2 class="sub-header">🔍 Data Quality Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Data Completeness", f"{(1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100:.1f}%")
    
    with col2:
        st.metric("Unique Cities", len(filtered_df['city_name'].unique()))
    
    with col3:
        st.metric("Risk Categories", len(set([label.strip() for labels in filtered_df['risk_labels'].str.split('|') for label in labels])))

elif selected == "📊 EDA Questions":
    st.markdown('<h1 class="main-header">📊 Exploratory Data Analysis Questions</h1>', 
                unsafe_allow_html=True)
    
    # EDA Question selector
    eda_questions = [
        "1. How does elevation correlate with risk labels across cities?",
        "2. Which cities have the highest concentration of high-risk areas?",
        "3. How does land use type influence flood risk severity?",
        "4. What's the relationship between drainage density and flood risk?",
        "5. How does storm drain proximity affect risk levels?",
        "6. Which soil types are most vulnerable to flooding?",
        "7. How does historical rainfall intensity correlate with risk?",
        "8. What's the geographic distribution of extreme risk areas?",
        "9. How do return periods vary across different risk categories?",
        "10. Which storm drain types are most effective?",
        "11. How does admin ward planning affect flood risk?",
        "12. What are the seasonal patterns in flood risk?"
    ]
    
    selected_question = st.selectbox("Select EDA Question to Explore:", eda_questions)
    
    if "elevation correlate with risk" in selected_question:
        st.markdown('<h2 class="sub-header">🏔️ Elevation vs Risk Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot with animation
            fig_scatter = px.scatter(
                filtered_df,
                x='elevation_m',
                y='risk_severity_score',
                color='city_name',
                size='historical_rainfall_intensity_mm_hr',
                hover_data=['admin_ward', 'primary_land_use'],
                title="Elevation vs Risk Severity by City",
                animation_frame='city_name' if len(filtered_df['city_name'].unique()) > 1 else None
            )
            fig_scatter.update_layout(height=500)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                filtered_df,
                x='risk_severity_score',
                y='elevation_m',
                color='has_low_lying',
                title="Elevation Distribution by Risk Score",
                labels={'has_low_lying': 'Low-lying Area'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Correlation analysis
        correlation = filtered_df['elevation_m'].corr(filtered_df['risk_severity_score'])
        st.markdown(f"""
        <div class="insight-box">
        <h4>📈 Key Insight</h4>
        <p>Correlation between elevation and risk severity: <strong>{correlation:.3f}</strong></p>
        <p>{'Lower elevations show higher flood risk as expected.' if correlation < -0.1 else 'Elevation shows weak correlation with risk, suggesting other factors are more important.'}</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif "highest concentration of high-risk" in selected_question:
        st.markdown('<h2 class="sub-header">🏙️ High-Risk Area Concentration by City</h2>', unsafe_allow_html=True)
        
        # Calculate high-risk concentration
        city_risk = filtered_df.groupby('city_name').agg({
            'risk_severity_score': ['mean', 'max', 'count'],
            'has_extreme_rain': 'sum',
            'has_ponding': 'sum'
        }).round(2)
        
        city_risk.columns = ['Avg_Risk', 'Max_Risk', 'Total_Segments', 'Extreme_Rain_Count', 'Ponding_Count']
        city_risk['High_Risk_Ratio'] = (city_risk['Avg_Risk'] > city_risk['Avg_Risk'].median()).astype(int)
        city_risk = city_risk.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Animated bar chart
            fig_bar = px.bar(
                city_risk.sort_values('Avg_Risk', ascending=False).head(15),
                x='city_name',
                y='Avg_Risk',
                color='Max_Risk',
                title="Average Risk Score by City",
                color_continuous_scale="Reds"
            )
            fig_bar.update_xaxes(tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Bubble chart
            fig_bubble = px.scatter(
                city_risk,
                x='Total_Segments',
                y='Avg_Risk',
                size='Max_Risk',
                color='Extreme_Rain_Count',
                hover_name='city_name',
                title="City Risk Profile (Size = Max Risk)",
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
    
    elif "land use type influence" in selected_question:
        st.markdown('<h2 class="sub-header">🏘️ Land Use Impact on Flood Risk</h2>', unsafe_allow_html=True)
        
        # Land use analysis
        land_use_risk = filtered_df.groupby('primary_land_use').agg({
            'risk_severity_score': ['mean', 'std', 'count'],
            'elevation_m': 'mean',
            'drainage_density_km_per_km2': 'mean'
        }).round(2)
        
        land_use_risk.columns = ['Avg_Risk', 'Risk_Std', 'Count', 'Avg_Elevation', 'Avg_Drainage']
        land_use_risk = land_use_risk.reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Violin plot
            fig_violin = px.violin(
                filtered_df,
                x='primary_land_use',
                y='risk_severity_score',
                box=True,
                title="Risk Distribution by Land Use Type"
            )
            fig_violin.update_xaxes(tickangle=45)
            st.plotly_chart(fig_violin, use_container_width=True)
        
        with col2:
            # Heatmap
            land_use_pivot = filtered_df.pivot_table(
                values='risk_severity_score',
                index='primary_land_use',
                columns='soil_type',
                aggfunc='mean'
            )
            
            fig_heatmap = px.imshow(
                land_use_pivot,
                title="Risk Heatmap: Land Use vs Soil Type",
                color_continuous_scale="RdYlBu_r"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    elif "drainage density and flood risk" in selected_question:
        st.markdown('<h2 class="sub-header">🚰 Drainage Density vs Flood Risk</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter with trendline
            fig_scatter = px.scatter(
                filtered_df,
                x='drainage_density_km_per_km2',
                y='risk_severity_score',
                color='primary_land_use',
                trendline="ols",
                title="Drainage Density vs Risk Severity"
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Binned analysis
            filtered_df['drainage_bin'] = pd.cut(filtered_df['drainage_density_km_per_km2'], 
                                               bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            
            drainage_risk = filtered_df.groupby('drainage_bin')['risk_severity_score'].agg(['mean', 'count']).reset_index()
            
            fig_bar = px.bar(
                drainage_risk,
                x='drainage_bin',
                y='mean',
                title="Average Risk by Drainage Density Level",
                color='mean',
                color_continuous_scale="RdYlGn_r"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Add more EDA questions...
    else:
        st.info("Select an EDA question from the dropdown to see detailed analysis.")

elif selected == "🗺️ Geographic Analysis":
    st.markdown('<h1 class="main-header">🗺️ Geographic Flood Risk Analysis</h1>', 
                unsafe_allow_html=True)
    
    # Geographic visualizations
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive map
        fig_map = px.scatter_mapbox(
            filtered_df,
            lat='latitude',
            lon='longitude',
            color='risk_severity_score',
            size='historical_rainfall_intensity_mm_hr',
            hover_name='city_name',
            hover_data=['admin_ward', 'primary_land_use', 'risk_labels'],
            color_continuous_scale="Reds",
            size_max=15,
            zoom=1,
            title="Global Flood Risk Distribution"
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            height=600
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Geographic Insights")
        
        # Risk by region analysis
        filtered_df['region'] = filtered_df['city_name'].str.extract(r', (.+)$')[0]
        region_risk = filtered_df.groupby('region')['risk_severity_score'].agg(['mean', 'count']).round(2)
        
        st.dataframe(region_risk.sort_values('mean', ascending=False))
        
        # Elevation analysis
        st.markdown("### 🏔️ Elevation Analysis")
        elevation_stats = filtered_df['elevation_m'].describe()
        st.write(elevation_stats)

elif selected == "⚠️ Risk Assessment":
    st.markdown('<h1 class="main-header">⚠️ Comprehensive Risk Assessment</h1>', 
                unsafe_allow_html=True)
    
    # Risk matrix
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk severity distribution
        fig_hist = px.histogram(
            filtered_df,
            x='risk_severity_score',
            color='primary_land_use',
            title="Risk Severity Distribution by Land Use",
            nbins=20
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Risk vs Infrastructure
        fig_scatter = px.scatter(
            filtered_df,
            x='storm_drain_proximity_m',
            y='risk_severity_score',
            color='drain_type',
            title="Storm Drain Proximity vs Risk"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Risk prediction factors
    st.markdown('<h2 class="sub-header">🔍 Risk Factor Analysis</h2>', unsafe_allow_html=True)
    
    # Correlation matrix
    numeric_cols = ['elevation_m', 'drainage_density_km_per_km2', 'storm_drain_proximity_m',
                   'historical_rainfall_intensity_mm_hr', 'return_period_years', 'risk_severity_score']
    
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Risk Factor Correlation Matrix",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif selected == "📈 Advanced Analytics":
    st.markdown('<h1 class="main-header">📈 Advanced Analytics & Predictions</h1>', 
                unsafe_allow_html=True)
    
    # Time series analysis (simulated)
    st.markdown('<h2 class="sub-header">📅 Temporal Risk Patterns</h2>', unsafe_allow_html=True)
    
    # Create synthetic time series data
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='M')
    np.random.seed(42)
    
    risk_trend = []
    for city in filtered_df['city_name'].unique()[:5]:  # Top 5 cities
        city_data = filtered_df[filtered_df['city_name'] == city]
        base_risk = city_data['risk_severity_score'].mean()
        
        # Add seasonal variation and trend
        seasonal_risk = [base_risk + 2*np.sin(2*np.pi*i/12) + np.random.normal(0, 0.5) 
                        for i in range(len(dates))]
        
        for i, date in enumerate(dates):
            risk_trend.append({
                'date': date,
                'city': city,
                'risk_score': max(0, seasonal_risk[i]),
                'segments': len(city_data)
            })
    
    risk_df = pd.DataFrame(risk_trend)
    
    # Animated time series
    fig_time = px.line(
        risk_df,
        x='date',
        y='risk_score',
        color='city',
        title="Risk Score Trends Over Time",
        animation_frame=risk_df['date'].dt.year
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Advanced statistical analysis
    st.markdown('<h2 class="sub-header">📊 Statistical Insights</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribution analysis
        fig_dist = px.box(
            filtered_df,
            x='soil_type',
            y='risk_severity_score',
            title="Risk Distribution by Soil Type"
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Multi-dimensional analysis
        fig_3d = px.scatter_3d(
            filtered_df.sample(min(500, len(filtered_df))),  # Sample for performance
            x='elevation_m',
            y='drainage_density_km_per_km2',
            z='historical_rainfall_intensity_mm_hr',
            color='risk_severity_score',
            title="3D Risk Factor Analysis",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_3d, use_container_width=True)

elif selected == "🎯 Insights":
    st.markdown('<h1 class="main-header">🎯 Key Insights & Recommendations</h1>', 
                unsafe_allow_html=True)
    
    # Generate insights based on data
    insights = []
    
    # City with highest risk
    city_risk = filtered_df.groupby('city_name')['risk_severity_score'].mean().sort_values(ascending=False)
    highest_risk_city = city_risk.index[0]
    insights.append(f"🏙️ **{highest_risk_city}** has the highest average flood risk score ({city_risk.iloc[0]:.2f})")
    
    # Land use insights
    land_use_risk = filtered_df.groupby('primary_land_use')['risk_severity_score'].mean().sort_values(ascending=False)
    highest_risk_land_use = land_use_risk.index[0]
    insights.append(f"🏘️ **{highest_risk_land_use}** areas show the highest flood vulnerability")
    
    # Elevation insights
    low_elevation_risk = filtered_df[filtered_df['elevation_m'] < filtered_df['elevation_m'].quantile(0.25)]['risk_severity_score'].mean()
    high_elevation_risk = filtered_df[filtered_df['elevation_m'] > filtered_df['elevation_m'].quantile(0.75)]['risk_severity_score'].mean()
    if low_elevation_risk > high_elevation_risk:
        insights.append(f"⛰️ Low-elevation areas have {((low_elevation_risk/high_elevation_risk - 1) * 100):.1f}% higher risk than high-elevation areas")
    
    # Drainage insights
    good_drainage = filtered_df[filtered_df['drainage_density_km_per_km2'] > filtered_df['drainage_density_km_per_km2'].median()]
    poor_drainage = filtered_df[filtered_df['drainage_density_km_per_km2'] <= filtered_df['drainage_density_km_per_km2'].median()]
    
    if poor_drainage['risk_severity_score'].mean() > good_drainage['risk_severity_score'].mean():
        insights.append(f"🚰 Areas with poor drainage density have {((poor_drainage['risk_severity_score'].mean()/good_drainage['risk_severity_score'].mean() - 1) * 100):.1f}% higher risk")
    
    # Display insights
    for i, insight in enumerate(insights, 1):
        st.markdown(f"""
        <div class="insight-box">
        <h4>Insight #{i}</h4>
        <p>{insight}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<h2 class="sub-header">💡 Strategic Recommendations</h2>', unsafe_allow_html=True)
    
    recommendations = [
        {
            "title": "🏗️ Infrastructure Investment",
            "description": "Prioritize storm drain improvements in areas with proximity > 200m from existing infrastructure",
            "priority": "High",
            "impact": "Immediate flood risk reduction"
        },
        {
            "title": "🌱 Green Infrastructure",
            "description": "Implement permeable surfaces and green roofs in high-density residential areas",
            "priority": "Medium",
            "impact": "Long-term sustainability"
        },
        {
            "title": "📊 Early Warning Systems",
            "description": "Deploy IoT sensors in areas with extreme rain history for real-time monitoring",
            "priority": "High",
            "impact": "Disaster preparedness"
        },
        {
            "title": "🏘️ Land Use Planning",
            "description": "Restrict development in low-lying areas with poor drainage",
            "priority": "Medium",
            "impact": "Prevention of future risks"
        }
    ]
    
    for rec in recommendations:
        st.markdown(f"""
        <div class="insight-box">
        <h4>{rec['title']}</h4>
        <p><strong>Description:</strong> {rec['description']}</p>
        <p><strong>Priority:</strong> {rec['priority']} | <strong>Impact:</strong> {rec['impact']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown('<h2 class="sub-header">📈 Summary Dashboard</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Risk Score", f"{filtered_df['risk_severity_score'].sum():,.0f}")
    
    with col2:
        st.metric("Average Risk", f"{filtered_df['risk_severity_score'].mean():.2f}")
    
    with col3:
        critical_areas = len(filtered_df[filtered_df['risk_severity_score'] >= 8])
        st.metric("Critical Areas", f"{critical_areas:,}")
    
    with col4:
        cities_analyzed = len(filtered_df['city_name'].unique())
        st.metric("Cities Covered", f"{cities_analyzed}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌊 Urban Flood Risk Analytics Dashboard | Built with Streamlit, Plotly & Advanced Analytics</p>
    <p>Data-driven insights for urban resilience and flood risk management</p>
</div>
""", unsafe_allow_html=True)
