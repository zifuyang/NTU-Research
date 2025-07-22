"""
Excel File Preparation Helper
Validates and prepares Excel files for the bias detection system
"""

import pandas as pd
import os
from config import UK_UNIVERSITIES, US_UNIVERSITIES

def validate_excel_file(file_path: str = "resume_dataset.xlsx") -> bool:
    """Validate that the Excel file has the correct format"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File '{file_path}' not found")
            return False
        
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Check required columns
        required_columns = ['resume_id', 'resume_content', 'university']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"Required columns: {required_columns}")
            return False
        
        # Check for empty data
        if len(df) == 0:
            print("❌ Excel file is empty")
            return False
        
        # Check for missing values in key columns
        missing_resume_ids = df['resume_id'].isna().sum()
        missing_content = df['resume_content'].isna().sum()
        missing_universities = df['university'].isna().sum()
        
        if missing_resume_ids > 0:
            print(f"⚠️  Warning: {missing_resume_ids} missing resume IDs")
        if missing_content > 0:
            print(f"⚠️  Warning: {missing_content} missing resume content")
        if missing_universities > 0:
            print(f"⚠️  Warning: {missing_universities} missing universities")
        
        # Classify universities
        df['university_category'] = df['university'].apply(classify_university)
        
        # Show university distribution
        uk_count = len(df[df['university_category'] == 'UK'])
        us_count = len(df[df['university_category'] == 'US'])
        other_count = len(df[df['university_category'] == 'Other'])
        
        print(f"\n📊 University Distribution:")
        print(f"  UK Universities: {uk_count}")
        print(f"  US Universities: {us_count}")
        print(f"  Other Universities: {other_count}")
        print(f"  Total: {len(df)}")
        
        # Show sample of universities
        print(f"\n🏫 Sample Universities:")
        for category in ['UK', 'US', 'Other']:
            category_unis = df[df['university_category'] == category]['university'].unique()[:5]
            print(f"  {category}: {', '.join(category_unis)}")
        
        print(f"\n✅ Excel file validation successful!")
        print(f"File: {file_path}")
        print(f"Total resumes: {len(df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating Excel file: {str(e)}")
        return False

def classify_university(university: str) -> str:
    """Classify university as UK, US, or Other"""
    if pd.isna(university):
        return "Other"
    
    university_lower = str(university).lower()
    
    for uk_uni in UK_UNIVERSITIES:
        if uk_uni.lower() in university_lower:
            return "UK"
    
    for us_uni in US_UNIVERSITIES:
        if us_uni.lower() in university_lower:
            return "US"
    
    return "Other"

def create_sample_excel_template():
    """Create a sample Excel template for reference"""
    sample_data = {
        'resume_id': ['UK_001', 'US_001', 'UK_002', 'US_002'],
        'resume_content': [
            'EDUCATION\nUniversity of Oxford\nComputer Science Degree\nGraduation: 2023\n\nEXPERIENCE\nSoftware Engineer at Google\n- Developed web applications\n- Led team of 5 developers\n\nSKILLS\nPython, JavaScript, React, AWS',
            'EDUCATION\nHarvard University\nComputer Science Degree\nGraduation: 2023\n\nEXPERIENCE\nSoftware Engineer at Microsoft\n- Built cloud infrastructure\n- Managed database systems\n\nSKILLS\nJava, C++, Azure, Docker',
            'EDUCATION\nUniversity of Cambridge\nComputer Science Degree\nGraduation: 2023\n\nEXPERIENCE\nData Scientist at Amazon\n- Implemented ML models\n- Analyzed large datasets\n\nSKILLS\nPython, TensorFlow, SQL, Spark',
            'EDUCATION\nStanford University\nComputer Science Degree\nGraduation: 2023\n\nEXPERIENCE\nFull Stack Developer at Apple\n- Created mobile apps\n- Optimized performance\n\nSKILLS\nSwift, React Native, iOS, Git'
        ],
        'university': ['University of Oxford', 'Harvard University', 'University of Cambridge', 'Stanford University']
    }
    
    df = pd.DataFrame(sample_data)
    template_file = 'resume_dataset_template.xlsx'
    df.to_excel(template_file, index=False)
    
    print(f"✅ Created sample template: {template_file}")
    print("Use this as a reference for your Excel file format")
    
    return template_file

def main():
    """Main function to validate Excel file"""
    print("=" * 50)
    print("EXCEL FILE VALIDATION")
    print("=" * 50)
    
    # Check for existing file
    if os.path.exists("resume_dataset.xlsx"):
        print("Found existing 'resume_dataset.xlsx' file")
        if validate_excel_file():
            print("\n🎉 Your Excel file is ready for processing!")
            print("Run: python3 main.py --step full --skip-data-generation")
        else:
            print("\n❌ Please fix the issues above before proceeding")
    else:
        print("No 'resume_dataset.xlsx' file found")
        print("\nOptions:")
        print("1. Create your own Excel file with columns: resume_id, resume_content, university")
        print("2. Generate sample data: python3 main.py --step generate_data")
        print("3. Create template: python3 prepare_excel.py --create-template")
        
        # Create template if requested
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == '--create-template':
            create_sample_excel_template()

if __name__ == "__main__":
    main() 