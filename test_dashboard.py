#!/usr/bin/env python3
"""
Simple test script to identify dashboard issues
"""

import sys
import subprocess

def test_streamlit():
    """Test basic streamlit functionality"""
    print("Testing Streamlit...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False

def test_csv_loading():
    """Test CSV file loading"""
    print("Testing CSV loading...")
    try:
        import pandas as pd
        df = pd.read_csv('urban_pluvial_flood_risk_cleaned.csv')
        print(f"✅ CSV loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return True
    except Exception as e:
        print(f"❌ CSV loading failed: {e}")
        return False

def test_dependencies():
    """Test all required dependencies"""
    print("Testing dependencies...")
    dependencies = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'matplotlib', 
        'seaborn', 'streamlit_option_menu', 'streamlit_lottie', 'requests'
    ]
    
    failed = []
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep}")
            failed.append(dep)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("✅ All dependencies imported successfully")
        return True

def run_simple_test():
    """Run a simple streamlit app test"""
    print("Creating simple test app...")
    
    test_app_content = '''
import streamlit as st
import pandas as pd

st.title("🌊 Test Dashboard")
st.write("This is a simple test to verify Streamlit is working")

try:
    df = pd.read_csv('urban_pluvial_flood_risk_cleaned.csv')
    st.success(f"✅ CSV loaded: {len(df)} rows")
    st.dataframe(df.head())
except Exception as e:
    st.error(f"❌ Error loading CSV: {e}")

st.write("If you see this message, Streamlit is working properly!")
'''
    
    with open('test_app.py', 'w') as f:
        f.write(test_app_content)
    
    print("✅ Test app created: test_app.py")
    print("To run test app: streamlit run test_app.py")

def main():
    print("🔍 Dashboard Diagnostic Tool")
    print("=" * 40)
    
    # Run tests
    tests = [
        test_streamlit,
        test_csv_loading,
        test_dependencies
    ]
    
    all_passed = True
    for test in tests:
        if not test():
            all_passed = False
        print("-" * 40)
    
    if all_passed:
        print("✅ All tests passed! Dashboard should work.")
        run_simple_test()
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    print("\n🚀 To run the main dashboard:")
    print("   streamlit run modern_dashboard.py")
    print("   Then open: http://localhost:8501")

if __name__ == "__main__":
    main()
