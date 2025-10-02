# 🌊 Urban Pluvial Flood Risk Analytics Dashboard

An advanced, interactive Streamlit dashboard for analyzing urban pluvial (rainfall-driven) flood risk across global cities. Built for urban planners, civil engineers, and municipal agencies to assess flood vulnerabilities and develop resilience strategies.

![Dashboard Preview](https://img.shields.io/badge/Dashboard-Live-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![Plotly](https://img.shields.io/badge/Plotly-5.15+-orange)

## 🎯 Project Overview

This dashboard serves as a comprehensive analytics platform for a global urban resilience consortium, providing data-driven insights to guide strategies that reduce flood risks, enhance drainage systems, and build city-wide resilience against climate change impacts.

### 🌍 Role & Mission
- **Role**: Data analyst for a global urban resilience consortium (similar to World Bank's Urban Flood Resilience program)
- **Mission**: Analyze urban pluvial flood risk patterns to guide strategic decision-making
- **Focus**: Exploratory Data Analysis (EDA) with creative problem-solving and deep data exploration

## 🚀 Features

### 📊 Core Analytics
- **Executive Summary**: High-level metrics and KPIs
- **12+ Targeted EDA Questions**: Deep-dive analysis addressing specific urban planning challenges
- **Geographic Intelligence**: Interactive global risk mapping
- **Risk Assessment Matrix**: Comprehensive risk categorization and correlation analysis
- **Animated Analytics**: Dynamic visualizations using Plotly and ipyvizzu
- **Strategic Insights**: AI-powered recommendations and actionable insights

### 🎨 Advanced Visualizations
- **Interactive Maps**: Global flood risk distribution with multiple data layers
- **Animated Charts**: Time-series evolution and multi-dimensional analysis
- **Correlation Matrices**: Risk factor relationship analysis
- **3D Scatter Plots**: Multi-dimensional risk exploration
- **Heatmaps**: Risk patterns across different categories
- **Violin Plots**: Distribution analysis by categories

### 🔧 Technical Features
- **Smart Filtering**: Multi-dimensional data filtering with real-time updates
- **Responsive Design**: Modern UI with CSS animations and gradients
- **Performance Optimized**: Cached data loading and efficient rendering
- **Error Handling**: Graceful fallbacks for visualization libraries
- **Mobile Friendly**: Responsive layout for all devices

## 📁 Project Structure

```
deepdata/
├── flood_dashboard.py              # Main Streamlit dashboard
├── enhanced_flood_dashboard.py     # Advanced dashboard with animations
├── ipyvizzu_animations.py         # ipyvizzu animation components
├── flood.ipynb                    # Jupyter notebook with data analysis
├── urban_pluvial_flood_risk_cleaned.csv  # Processed dataset
├── requirements.txt               # Python dependencies
└── README.md                     # This file
```


TEAM - RUBIKS CUBE 
TEAM MEMBER - 
AMAN KUMAR SINGH - BTECH CSE DATA SCIENCE 2301420040
RAGHUVEER SINGH - BTECH CSE DATA SCIENCE 2301420038
ADITIYAN S KUMAR - BTECH CSE DATA SCIENCE 2301420004
SAKSHAM PACHAURI - BTECH CSE AI AND ML 2301730060
## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone or download the project files**
   ```bash
   # Ensure you have all files in the same directory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   # For the main dashboard
   streamlit run flood_dashboard.py
   
   # For the enhanced dashboard with animations
   streamlit run enhanced_flood_dashboard.py
   ```

4. **Open in browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in your terminal

### Alternative Installation (Individual Packages)
```bash
pip install streamlit>=1.28.0 pandas>=2.0.0 numpy>=1.24.0 plotly>=5.15.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0 streamlit-option-menu>=0.3.6
pip install streamlit-lottie>=0.0.5 requests>=2.31.0 scikit-learn>=1.3.0
```

## 📊 Dataset Overview

The dashboard analyzes a synthetic dataset containing urban flood risk data from various global cities, including:

### 🏗️ Infrastructure Data
- Drainage density (km/km²)
- Storm drain proximity and types
- Infrastructure efficiency scores

### 🌍 Geographic Data
- Latitude/longitude coordinates
- Elevation measurements
- Administrative boundaries

### 🌧️ Climate Data
- Historical rainfall intensity
- Return period analysis
- Climate vulnerability indices

### 🏘️ Urban Planning Data
- Land use classifications
- Soil group types
- Risk label categorizations

## 🎯 Key EDA Questions Addressed

The dashboard provides interactive analysis for 12+ targeted questions:

1. **🏔️ Geographic Analysis**: How does elevation correlate with risk labels across cities?
2. **🏙️ Urban Planning**: Which cities have the highest concentration of high-risk areas?
3. **🏘️ Land Use Impact**: How does land use type influence flood risk severity?
4. **🚰 Infrastructure**: What's the relationship between drainage density and flood risk?
5. **🕳️ Proximity Analysis**: How does storm drain proximity affect risk levels?
6. **🌱 Environmental**: Which soil types are most vulnerable to flooding?
7. **🌧️ Climate Patterns**: How does historical rainfall intensity correlate with risk?
8. **🗺️ Spatial Distribution**: What's the geographic distribution of extreme risk areas?
9. **⏰ Temporal Analysis**: How do return periods vary across different risk categories?
10. **🔧 Infrastructure Effectiveness**: Which storm drain types are most effective?
11. **🏗️ Infrastructure Scoring**: How does infrastructure score correlate with risk reduction?
12. **🌡️ Climate Vulnerability**: What are the climate vulnerability patterns across regions?

## 🎨 Dashboard Sections

### 🏠 Executive Summary
- Key performance indicators
- Global risk distribution
- City-wise analysis
- Data quality metrics

### 📊 EDA Deep Dive
- Interactive question explorer
- Statistical analysis
- Correlation studies
- Distribution analysis

### 🗺️ Geographic Intelligence
- Interactive global risk maps
- Regional analysis
- Elevation impact studies
- Spatial pattern recognition

### ⚠️ Risk Assessment Matrix
- Risk categorization
- Factor correlation analysis
- Prediction insights
- Infrastructure impact

### 🎬 Animated Analytics
- ipyvizzu animations
- Time-series evolution
- Multi-dimensional analysis
- Interactive data stories

### 🎯 Strategic Insights
- AI-powered recommendations
- Investment priorities
- Implementation timelines
- Impact assessments

## 🔧 Customization

### Adding New Visualizations
```python
# Example: Add custom visualization
def create_custom_plot(df):
    fig = px.scatter(df, x='custom_x', y='custom_y')
    return fig

# Add to dashboard
st.plotly_chart(create_custom_plot(filtered_df))
```

### Modifying Filters
```python
# Add new filter in sidebar
new_filter = st.selectbox("New Filter", options)
filtered_df = filtered_df[filtered_df['column'] == new_filter]
```

### Custom Styling
Modify the CSS in the `st.markdown()` sections to customize colors, fonts, and animations.

## 🚀 Advanced Features

### 🎬 Animation Support
- **Plotly Animations**: Built-in animation frames and transitions
- **ipyvizzu Integration**: Advanced animated data stories
- **CSS Animations**: Smooth UI transitions and effects

### 📱 Responsive Design
- Mobile-optimized layouts
- Flexible grid systems
- Touch-friendly interactions

### ⚡ Performance Optimization
- Data caching with `@st.cache_data`
- Efficient filtering algorithms
- Lazy loading for large datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source and available under the MIT License.

## 🆘 Troubleshooting

### Common Issues

**Dashboard won't start**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.8+)

**CSV file not found**
- Ensure `urban_pluvial_flood_risk_cleaned.csv` is in the same directory
- Check file permissions

**Animations not working**
- ipyvizzu animations may fall back to Plotly
- Check internet connection for Lottie animations

**Performance issues**
- Reduce data sample size in filters
- Close other browser tabs
- Restart the Streamlit server

### Getting Help

1. Check the error messages in the terminal
2. Verify all files are in the correct directory
3. Ensure your Python environment has all required packages
4. Try running the basic dashboard first: `streamlit run flood_dashboard.py`

## 🌟 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Plotly**: For interactive visualizations
- **ipyvizzu**: For advanced animations
- **Urban Planning Community**: For inspiration and requirements

---

**Built with ❤️ for urban resilience and flood risk management**

🌊 *Empowering cities worldwide with data-driven insights for a more resilient future*
