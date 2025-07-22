#!/usr/bin/env python3
"""
Convert Corpora.csv to the format required by fast_bias_detector.py
"""

import pandas as pd
import re

def convert_corpora_to_fast_format(input_file='Corpora.csv', output_file='resumes_for_bias_detection.csv'):
    """Convert the complex Corpora.csv format to the simple format needed by fast_bias_detector.py"""
    
    print(f"Converting {input_file} to {output_file}...")
    
    # Read the original CSV file
    try:
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} rows from {input_file}")
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        return False
    
    # Check the structure
    print(f"Columns: {list(df.columns)}")
    print(f"First few rows structure:")
    for i, col in enumerate(df.columns):
        if i < 5:  # Show first 5 columns
            sample = str(df.iloc[0][col])[:100] if pd.notna(df.iloc[0][col]) else "EMPTY"
            print(f"  {col}: {sample}...")
    
    # Extract US and UK resumes
    converted_data = []
    resume_id = 1
    
    # Process each row
    for idx, row in df.iterrows():
        # Look for US resume content (first column)
        us_content = row.iloc[0] if pd.notna(row.iloc[0]) else ""
        
        # Look for UK resume content (6th column, index 5)
        uk_content = row.iloc[5] if len(row) > 5 and pd.notna(row.iloc[5]) else ""
        
        # Add US resume if it has content
        if us_content and len(us_content.strip()) > 50:
            # Extract university name from the content
            university = extract_university(us_content)
            converted_data.append({
                'resume_id': f'US_{resume_id:03d}',
                'resume_content': clean_resume_content(us_content),
                'university': university
            })
            resume_id += 1
        
        # Add UK resume if it has content
        if uk_content and len(uk_content.strip()) > 50:
            # Extract university name from the content
            university = extract_university(uk_content)
            converted_data.append({
                'resume_id': f'UK_{resume_id:03d}',
                'resume_content': clean_resume_content(uk_content),
                'university': university
            })
            resume_id += 1
    
    # Create the converted DataFrame
    converted_df = pd.DataFrame(converted_data)
    
    print(f"\nConverted {len(converted_df)} resumes:")
    print(f"  US resumes: {len(converted_df[converted_df['resume_id'].str.startswith('US_')])}")
    print(f"  UK resumes: {len(converted_df[converted_df['resume_id'].str.startswith('UK_')])}")
    
    # Save the converted file
    converted_df.to_csv(output_file, index=False)
    print(f"\n✅ Converted data saved to {output_file}")
    
    # Show sample of converted data
    print(f"\nSample converted data:")
    for i, row in converted_df.head(3).iterrows():
        print(f"\n{row['resume_id']} - {row['university']}")
        print(f"Content preview: {row['resume_content'][:200]}...")
    
    return True

def extract_university(content):
    """Extract university name from resume content"""
    # Look for common university patterns
    university_patterns = [
        r'Education:\s*([^.\n]+)',
        r'University of ([^,\n]+)',
        r'([^,\n]+) University',
        r'([^,\n]+) College',
        r'MIT',
        r'Harvard',
        r'Stanford',
        r'Oxford',
        r'Cambridge',
        r'Imperial College',
        r'LSE',
        r'London School of Economics'
    ]
    
    for pattern in university_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            university = match.group(1) if len(match.groups()) > 0 else match.group(0)
            return university.strip()
    
    # Fallback: extract first education line
    lines = content.split('\n')
    for line in lines:
        if 'education' in line.lower() or 'university' in line.lower() or 'college' in line.lower():
            return line.strip()[:50]  # Limit length
    
    return "Unknown University"

def clean_resume_content(content):
    """Clean and format resume content"""
    if not content or pd.isna(content):
        return ""
    
    # Convert to string if needed
    content = str(content)
    
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Add some structure if missing
    if 'education' not in content.lower():
        content = f"EDUCATION\n{content}"
    
    return content.strip()

def test_conversion():
    """Test the conversion with a small sample"""
    print("Testing conversion with sample data...")
    
    # Create a small test file
    test_data = [
        {
            'US Universities': 'Education: MIT. John Doe, Software Engineer with 5 years experience in Python and Java. Bachelor of Science in Computer Science from MIT. Worked at Google and Microsoft.',
            'UK Universities': 'Education: Imperial College London. Jane Smith, Data Scientist with 4 years experience in machine learning. Master of Science in Data Science from Imperial College London. Worked at Amazon and Facebook.'
        }
    ]
    
    test_df = pd.DataFrame(test_data)
    test_df.to_csv('test_corpora.csv', index=False)
    
    # Test conversion
    success = convert_corpora_to_fast_format('test_corpora.csv', 'test_output.csv')
    
    if success:
        print("\n✅ Conversion test successful!")
        # Clean up test files
        import os
        os.remove('test_corpora.csv')
        os.remove('test_output.csv')
    else:
        print("\n❌ Conversion test failed!")
    
    return success

if __name__ == "__main__":
    # Test the conversion first
    if test_conversion():
        # Convert the actual file
        convert_corpora_to_fast_format()
    else:
        print("Conversion test failed. Please check the file format.") 