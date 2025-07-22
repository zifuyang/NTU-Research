"""
Test Script for Resume Bias Detection System
Tests all components without requiring OpenAI API calls
"""

import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# Import our modules
from sample_data_generator import SampleDataGenerator
from bias_analyzer import BiasAnalyzer
from config import UK_UNIVERSITIES, US_UNIVERSITIES

def test_sample_data_generator():
    """Test the sample data generator"""
    print("Testing Sample Data Generator...")
    
    try:
        generator = SampleDataGenerator()
        
        # Test generating a small dataset
        df = generator.generate_dataset(10)
        
        # Verify structure
        assert 'resume_id' in df.columns
        assert 'resume_content' in df.columns
        assert 'university' in df.columns
        assert 'quality' in df.columns
        
        # Verify data
        assert len(df) == 10
        assert all(df['resume_content'].str.len() > 100)  # Content should be substantial
        
        # Verify university distribution
        uk_count = sum(df['university'].isin(UK_UNIVERSITIES))
        us_count = sum(df['university'].isin(US_UNIVERSITIES))
        assert uk_count > 0 and us_count > 0
        
        print("✅ Sample Data Generator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Sample Data Generator: FAILED - {str(e)}")
        return False

def test_bias_analyzer():
    """Test the bias analyzer with mock data"""
    print("Testing Bias Analyzer...")
    
    try:
        # Create mock data
        mock_data = []
        for i in range(20):
            # Create balanced data with slight bias for testing
            if i < 10:
                university = "University of Oxford"
                category = "UK"
                rating = 7.5 + np.random.normal(0, 1)  # UK slightly higher
            else:
                university = "Harvard University"
                category = "US"
                rating = 7.0 + np.random.normal(0, 1)  # US slightly lower
            
            mock_data.append({
                'resume_id': f'TEST_{i:03d}',
                'university': university,
                'university_category': category,
                'overall_rating': max(1, min(10, rating)),
                'technical_skills': max(1, min(10, rating + np.random.normal(0, 0.5))),
                'experience_relevance': max(1, min(10, rating + np.random.normal(0, 0.5))),
                'education_quality': max(1, min(10, rating + np.random.normal(0, 0.5))),
                'communication_skills': max(1, min(10, rating + np.random.normal(0, 0.5))),
                'recommendation': 'HIRE' if rating > 7 else 'MAYBE' if rating > 5 else 'REJECT',
                'confidence_level': max(1, min(10, 8 + np.random.normal(0, 1))),
                'key_strengths': ['Technical skills', 'Experience'],
                'areas_of_concern': [],
                'university_bias_detected': False,
                'bias_explanation': '',
                'raw_response': 'Mock response',
                'processing_timestamp': '2024-01-01T00:00:00'
            })
        
        # Save mock data
        mock_df = pd.DataFrame(mock_data)
        os.makedirs('results', exist_ok=True)
        mock_df.to_csv('results/bias_analysis_results.csv', index=False)
        
        # Test analyzer
        analyzer = BiasAnalyzer('results/bias_analysis_results.csv')
        analyzer.load_results()
        analyzer.clean_data()
        
        # Test descriptive statistics
        stats = analyzer.descriptive_statistics()
        assert 'UK' in stats
        assert 'US' in stats
        assert stats['UK']['count'] == 10
        assert stats['US']['count'] == 10
        
        # Test statistical tests
        tests = analyzer.statistical_tests()
        assert 'overall_rating_ttest' in tests
        
        # Test bias patterns
        patterns = analyzer.detect_bias_patterns()
        assert 'technical_skills' in patterns
        
        print("✅ Bias Analyzer: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Bias Analyzer: FAILED - {str(e)}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("Testing Configuration...")
    
    try:
        from config import (
            OPENAI_MODEL, MAX_TOKENS, TEMPERATURE,
            BATCH_SIZE, DELAY_BETWEEN_REQUESTS, MAX_RETRIES,
            UK_UNIVERSITIES, US_UNIVERSITIES
        )
        
        # Verify configuration values
        assert OPENAI_MODEL in ["gpt-4", "gpt-3.5-turbo"]
        assert 1 <= MAX_TOKENS <= 4000
        assert 0 <= TEMPERATURE <= 1
        assert BATCH_SIZE > 0
        assert DELAY_BETWEEN_REQUESTS >= 0
        assert MAX_RETRIES > 0
        assert len(UK_UNIVERSITIES) > 0
        assert len(US_UNIVERSITIES) > 0
        
        print("✅ Configuration: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Configuration: FAILED - {str(e)}")
        return False

def test_data_structures():
    """Test data structures and imports"""
    print("Testing Data Structures...")
    
    try:
        # Test university lists
        assert "University of Oxford" in UK_UNIVERSITIES
        assert "Harvard University" in US_UNIVERSITIES
        
        # Test that lists don't overlap
        uk_set = set(UK_UNIVERSITIES)
        us_set = set(US_UNIVERSITIES)
        assert len(uk_set.intersection(us_set)) == 0
        
        # Test pandas and numpy imports
        test_df = pd.DataFrame({'test': [1, 2, 3]})
        assert len(test_df) == 3
        
        test_array = np.array([1, 2, 3])
        assert len(test_array) == 3
        
        print("✅ Data Structures: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data Structures: FAILED - {str(e)}")
        return False

def test_file_operations():
    """Test file operations"""
    print("Testing File Operations...")
    
    try:
        # Test creating results directory
        os.makedirs('results', exist_ok=True)
        assert os.path.exists('results')
        
        # Test saving and loading CSV
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        test_file = 'results/test.csv'
        test_data.to_csv(test_file, index=False)
        assert os.path.exists(test_file)
        
        loaded_data = pd.read_csv(test_file)
        assert len(loaded_data) == 3
        
        # Cleanup
        os.remove(test_file)
        
        print("✅ File Operations: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ File Operations: FAILED - {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 50)
    print("SYSTEM TESTING")
    print("=" * 50)
    
    tests = [
        test_configuration,
        test_data_structures,
        test_file_operations,
        test_sample_data_generator,
        test_bias_analyzer
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
        print("🎉 ALL TESTS PASSED! System is ready to use.")
        print("\nNext steps:")
        print("1. Set your OpenAI API key in a .env file")
        print("2. Run: python main.py --step full --num-resumes 10")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 