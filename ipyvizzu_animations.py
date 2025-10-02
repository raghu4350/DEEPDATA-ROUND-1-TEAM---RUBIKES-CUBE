import streamlit as st
import pandas as pd
import json

# Try to import ipyvizzu components, fallback if not available
try:
    from streamlit_ipyvizzu import chart
    IPYVIZZU_AVAILABLE = True
except ImportError:
    IPYVIZZU_AVAILABLE = False
    chart = None

def create_animated_risk_analysis(df):
    """Create animated risk analysis using ipyvizzu"""
    
    # Prepare data for ipyvizzu
    chart_data = {
        "data": {
            "series": [
                {
                    "name": "City",
                    "values": df['city_name'].tolist()
                },
                {
                    "name": "Risk Score",
                    "values": df['risk_severity_score'].tolist()
                },
                {
                    "name": "Land Use",
                    "values": df['primary_land_use'].tolist()
                },
                {
                    "name": "Elevation",
                    "values": df['elevation_m'].tolist()
                },
                {
                    "name": "Rainfall Intensity",
                    "values": df['historical_rainfall_intensity_mm_hr'].tolist()
                }
            ]
        }
    }
    
    # Animation config
    config = {
        "channels": {
            "x": "City",
            "y": "Risk Score",
            "color": "Land Use",
            "size": "Elevation"
        },
        "title": "Urban Flood Risk Analysis - Animated Scatter Plot",
        "coordSystem": "cartesian"
    }
    
    # Animation steps
    animation_steps = [
        {
            "target": {
                "channels": {
                    "x": "City",
                    "y": "Risk Score",
                    "color": "Land Use"
                }
            },
            "options": {
                "duration": 2000,
                "title": "Risk Score by City and Land Use"
            }
        },
        {
            "target": {
                "channels": {
                    "x": "Elevation",
                    "y": "Risk Score",
                    "color": "Land Use",
                    "size": "Rainfall Intensity"
                }
            },
            "options": {
                "duration": 2000,
                "title": "Risk vs Elevation (Size = Rainfall Intensity)"
            }
        },
        {
            "target": {
                "channels": {
                    "x": "Land Use",
                    "y": "Risk Score",
                    "color": "City"
                },
                "coordSystem": "polar"
            },
            "options": {
                "duration": 2000,
                "title": "Risk Distribution by Land Use (Polar View)"
            }
        }
    ]
    
    return chart_data, config, animation_steps

def create_time_series_animation(df):
    """Create time series animation for flood risk trends"""
    
    # Create synthetic time series data
    import numpy as np
    
    cities = df['city_name'].unique()[:8]  # Top 8 cities
    years = list(range(2020, 2025))
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    time_data = []
    np.random.seed(42)
    
    for city in cities:
        city_base_risk = df[df['city_name'] == city]['risk_severity_score'].mean()
        
        for year in years:
            for i, month in enumerate(months):
                # Add seasonal variation
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * i / 12)
                # Add yearly trend
                yearly_factor = 1 + 0.05 * (year - 2020)
                # Add random variation
                random_factor = 1 + np.random.normal(0, 0.1)
                
                risk_value = city_base_risk * seasonal_factor * yearly_factor * random_factor
                
                time_data.append({
                    'City': city,
                    'Year': year,
                    'Month': month,
                    'Date': f"{year}-{i+1:02d}",
                    'Risk Score': max(0, risk_value),
                    'Season': 'Wet' if i in [5, 6, 7, 8] else 'Dry'
                })
    
    time_df = pd.DataFrame(time_data)
    
    chart_data = {
        "data": {
            "series": [
                {
                    "name": "City",
                    "values": time_df['City'].tolist()
                },
                {
                    "name": "Date",
                    "values": time_df['Date'].tolist()
                },
                {
                    "name": "Risk Score",
                    "values": time_df['Risk Score'].tolist()
                },
                {
                    "name": "Season",
                    "values": time_df['Season'].tolist()
                },
                {
                    "name": "Year",
                    "values": time_df['Year'].tolist()
                }
            ]
        }
    }
    
    return chart_data, time_df

def create_geographic_animation(df):
    """Create geographic animation showing risk distribution"""
    
    # Prepare geographic data
    geo_data = df.groupby(['city_name', 'primary_land_use']).agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'risk_severity_score': 'mean',
        'elevation_m': 'mean',
        'segment_id': 'count'
    }).reset_index()
    
    geo_data.rename(columns={'segment_id': 'segment_count'}, inplace=True)
    
    chart_data = {
        "data": {
            "series": [
                {
                    "name": "City",
                    "values": geo_data['city_name'].tolist()
                },
                {
                    "name": "Land Use",
                    "values": geo_data['primary_land_use'].tolist()
                },
                {
                    "name": "Latitude",
                    "values": geo_data['latitude'].tolist()
                },
                {
                    "name": "Longitude",
                    "values": geo_data['longitude'].tolist()
                },
                {
                    "name": "Risk Score",
                    "values": geo_data['risk_severity_score'].tolist()
                },
                {
                    "name": "Elevation",
                    "values": geo_data['elevation_m'].tolist()
                },
                {
                    "name": "Segment Count",
                    "values": geo_data['segment_count'].tolist()
                }
            ]
        }
    }
    
    return chart_data, geo_data

def render_ipyvizzu_chart(chart_data, config, animation_steps=None):
    """Render ipyvizzu chart in Streamlit"""
    
    if not IPYVIZZU_AVAILABLE:
        st.warning("⚠️ ipyvizzu not available. Showing alternative Plotly visualization instead.")
        return False
    
    try:
        # Create the chart
        chart_config = {
            "data": chart_data,
            "config": config
        }
        
        if animation_steps:
            chart_config["animationSteps"] = animation_steps
        
        # Render using streamlit-ipyvizzu
        chart(chart_config, key=f"chart_{hash(str(chart_config))}")
        
    except Exception as e:
        st.error(f"Error rendering ipyvizzu chart: {e}")
        st.info("Falling back to alternative visualization...")
        return False
    
    return True

def create_risk_matrix_animation(df):
    """Create animated risk matrix visualization"""
    
    # Create risk matrix data
    risk_matrix = df.groupby(['primary_land_use', 'soil_type']).agg({
        'risk_severity_score': ['mean', 'count', 'std']
    }).round(2)
    
    risk_matrix.columns = ['avg_risk', 'count', 'std_risk']
    risk_matrix = risk_matrix.reset_index()
    
    chart_data = {
        "data": {
            "series": [
                {
                    "name": "Land Use",
                    "values": risk_matrix['primary_land_use'].tolist()
                },
                {
                    "name": "Soil Type",
                    "values": risk_matrix['soil_type'].tolist()
                },
                {
                    "name": "Average Risk",
                    "values": risk_matrix['avg_risk'].tolist()
                },
                {
                    "name": "Count",
                    "values": risk_matrix['count'].tolist()
                },
                {
                    "name": "Risk Std",
                    "values": risk_matrix['std_risk'].fillna(0).tolist()
                }
            ]
        }
    }
    
    config = {
        "channels": {
            "x": "Land Use",
            "y": "Soil Type",
            "color": "Average Risk",
            "size": "Count"
        },
        "title": "Risk Matrix: Land Use vs Soil Type",
        "coordSystem": "cartesian"
    }
    
    return chart_data, config

def create_infrastructure_analysis(df):
    """Create infrastructure impact analysis animation"""
    
    # Bin drainage density and storm drain proximity
    df['drainage_category'] = pd.cut(df['drainage_density_km_per_km2'], 
                                   bins=3, labels=['Low', 'Medium', 'High'])
    df['proximity_category'] = pd.cut(df['storm_drain_proximity_m'], 
                                    bins=3, labels=['Near', 'Medium', 'Far'])
    
    infra_data = df.groupby(['drainage_category', 'proximity_category', 'drain_type']).agg({
        'risk_severity_score': 'mean',
        'segment_id': 'count'
    }).reset_index()
    
    infra_data.rename(columns={'segment_id': 'segment_count'}, inplace=True)
    
    chart_data = {
        "data": {
            "series": [
                {
                    "name": "Drainage Category",
                    "values": infra_data['drainage_category'].astype(str).tolist()
                },
                {
                    "name": "Proximity Category",
                    "values": infra_data['proximity_category'].astype(str).tolist()
                },
                {
                    "name": "Drain Type",
                    "values": infra_data['drain_type'].tolist()
                },
                {
                    "name": "Risk Score",
                    "values": infra_data['risk_severity_score'].tolist()
                },
                {
                    "name": "Segment Count",
                    "values": infra_data['segment_count'].tolist()
                }
            ]
        }
    }
    
    return chart_data, infra_data
