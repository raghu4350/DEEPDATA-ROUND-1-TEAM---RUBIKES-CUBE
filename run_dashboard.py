#!/usr/bin/env python3
"""
🌊 Urban Flood Risk Analytics Dashboard Launcher
Simple script to launch the dashboard with proper setup checks
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all required files and dependencies exist"""
    print("🔍 Checking requirements...")
    
    # Check if CSV file exists
    csv_file = "urban_pluvial_flood_risk_cleaned.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found!")
        print("   Please ensure the CSV file is in the same directory as this script.")
        return False
    
    # Check if main dashboard files exist
    dashboard_files = ["flood_dashboard.py", "enhanced_flood_dashboard.py"]
    missing_files = [f for f in dashboard_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Error: Missing dashboard files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found!")
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    requirements = [
        "streamlit>=1.28.0",
        "pandas>=2.0.0", 
        "numpy>=1.24.0",
        "plotly>=5.15.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "streamlit-option-menu>=0.3.6",
        "streamlit-lottie>=0.0.5",
        "requests>=2.31.0",
        "scikit-learn>=1.3.0"
    ]
    
    try:
        for package in requirements:
            print(f"   Installing {package.split('>=')[0]}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        print("   Please run: pip install -r requirements.txt")
        return False

def launch_dashboard(dashboard_type="enhanced"):
    """Launch the selected dashboard"""
    dashboard_files = {
        "basic": "flood_dashboard.py",
        "enhanced": "enhanced_flood_dashboard.py"
    }
    
    dashboard_file = dashboard_files.get(dashboard_type, "enhanced_flood_dashboard.py")
    
    print(f"🚀 Launching {dashboard_type} dashboard...")
    print(f"   File: {dashboard_file}")
    print("   Opening browser at: http://localhost:8501")
    print("\n" + "="*50)
    print("🌊 URBAN FLOOD RISK ANALYTICS DASHBOARD")
    print("="*50)
    print("📊 Dashboard is starting...")
    print("🌐 Browser will open automatically")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("="*50 + "\n")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_file])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("🌊 Urban Flood Risk Analytics Dashboard Launcher")
    print("=" * 55)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Ask user for dashboard type
    print("\n📊 Choose dashboard version:")
    print("1. Enhanced Dashboard (Recommended) - Full features with animations")
    print("2. Basic Dashboard - Standard features")
    print("3. Install requirements only")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            # Check and install requirements
            try:
                import streamlit
                import pandas
                import plotly
                print("✅ Core packages already installed!")
            except ImportError:
                if not install_requirements():
                    return
            
            launch_dashboard("enhanced")
            break
            
        elif choice == "2":
            # Check and install requirements
            try:
                import streamlit
                import pandas
                import plotly
                print("✅ Core packages already installed!")
            except ImportError:
                if not install_requirements():
                    return
            
            launch_dashboard("basic")
            break
            
        elif choice == "3":
            install_requirements()
            print("✅ Requirements installed! Run this script again to launch the dashboard.")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
