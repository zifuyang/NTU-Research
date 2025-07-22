# Fast Resume Bias Detector

A simple, optimized Python script for detecting hiring bias between UK and US universities using ChatGPT.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_fast.txt
   ```

2. **Set up API key:**
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run bias detection:**
   ```bash
   python fast_bias_detector.py resumes_for_bias_detection.csv
   ```

## 📁 Files

- `fast_bias_detector.py` - Main bias detection script
- `resumes_for_bias_detection.csv` - Your converted resume data (400 resumes)
- `Corpora.csv` - Your original data
- `convert_corpora.py` - Data conversion script (if you need to reconvert)
- `requirements_fast.txt` - Python dependencies

## 📊 Output

The script outputs a CSV file with:
- `resume_id`: US_001, UK_002, etc.
- `resume_content`: Original resume text
- `university`: University name
- `ai_response`: Full ChatGPT response
- `verdict`: US or UK
- `reasoning`: ChatGPT's explanation
- `status`: success/error

## ⚙️ Options

```bash
# Custom output file
python fast_bias_detector.py resumes_for_bias_detection.csv --output results.csv

# Adjust processing speed
python fast_bias_detector.py resumes_for_bias_detection.csv --batch-size 20 --delay 0.2

# Use different model
python fast_bias_detector.py resumes_for_bias_detection.csv --model gpt-4
```

## 💰 Cost Estimate

- **400 resumes**: ~$2-4 USD
- **Processing time**: ~10-15 minutes 