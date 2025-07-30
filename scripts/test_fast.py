#!/usr/bin/env python3
"""
Test script for Fast Bias Detector
Validates setup and dependencies
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import asyncio
        print("✅ asyncio imported successfully")
    except ImportError as e:
        print(f"❌ asyncio import failed: {e}")
        return False
    
    try:
        import aiohttp
        print("✅ aiohttp imported successfully")
    except ImportError as e:
        print(f"❌ aiohttp import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

def test_csv_reading():
    """Test if CSV files can be read"""
    print("\nTesting CSV reading...")
    
    try:
        import pandas as pd
        df = pd.read_csv('example_resumes.csv')
        print(f"✅ CSV file read successfully: {len(df)} rows")
        
        # Check required columns
        required_columns = ['resume_id', 'resume_content', 'university']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            return False
        else:
            print("✅ All required columns present")
            return True
            
    except Exception as e:
        print(f"❌ CSV reading failed: {e}")
        return False

def test_api_key():
    """Test if API key is available"""
    print("\nTesting API key...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print("✅ OpenAI API key found in .env file")
        return True
    else:
        print("⚠️  OpenAI API key not found in .env file")
        print("   You can still run the script by passing --api-key parameter")
        return True  # Not a critical error

def test_script():
    """Test if the main script can be imported"""
    print("\nTesting main script...")
    
    try:
        from fast_bias_detector import FastBiasDetector
        print("✅ FastBiasDetector class imported successfully")
        return True
    except Exception as e:
        print(f"❌ Script import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("FAST BIAS DETECTOR - SETUP TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_csv_reading,
        test_api_key,
        test_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Prepare your CSV file with columns: resume_id, resume_content, university")
        print("3. Run: python fast_bias_detector.py your_resumes.csv")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements_fast.txt")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 