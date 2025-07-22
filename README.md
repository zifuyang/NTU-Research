# Comparison Resume Bias Detector

A Python script that compares US vs UK resumes side by side and asks AI models to choose which applicant to hire, detecting potential hiring bias.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements_fast.txt
   ```

2. **Set up API key:**
   Create a `.env` file with either:
   ```
   # For OpenAI (GPT models)
   OPENAI_API_KEY=your_openai_api_key_here
   
   # For Google Gemini
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

3. **Run comparison bias detection:**
   ```bash
   # Using OpenAI GPT models
   python comparison_bias_detector.py resumes_for_bias_detection.csv
   
   # Using Google Gemini
   python gemini_comparison_bias_detector.py resumes_for_bias_detection.csv
   ```

## 📁 Files

- `comparison_bias_detector.py` - OpenAI GPT version
- `gemini_comparison_bias_detector.py` - Google Gemini version
- `resumes_for_bias_detection.csv` - Your converted resume data (200 US + 200 UK resumes)
- `Corpora.csv` - Your original data
- `requirements_fast.txt` - Python dependencies

## 📊 Output

Both scripts output a CSV file with:
- `comparison_id`: COMP_001, COMP_002, etc.
- `us_resume_id`: US resume identifier
- `us_university`: US candidate's university
- `us_resume_content`: US candidate's resume
- `uk_resume_id`: UK resume identifier
- `uk_university`: UK candidate's university
- `uk_resume_content`: UK candidate's resume
- `ai_response`: Full AI response
- `verdict`: US or UK (which candidate AI chose)
- `reasoning`: AI's explanation
- `status`: success/error

## ⚙️ Options

```bash
# Custom output file
python comparison_bias_detector.py resumes_for_bias_detection.csv --output results.csv
python gemini_comparison_bias_detector.py resumes_for_bias_detection.csv --output gemini_results.csv

# Adjust processing speed (for rate limiting)
python comparison_bias_detector.py resumes_for_bias_detection.csv --batch-size 10 --delay 0.5
python gemini_comparison_bias_detector.py resumes_for_bias_detection.csv --batch-size 10 --delay 0.5

# Use different models
python comparison_bias_detector.py resumes_for_bias_detection.csv --model gpt-4o
python gemini_comparison_bias_detector.py resumes_for_bias_detection.csv --model gemini-1.5-pro
```

## 💰 Cost Estimate

**OpenAI Models:**
- **200 comparisons with GPT-4o-mini**: ~$2-3 USD
- **200 comparisons with GPT-4o**: ~$4-6 USD

**Google Gemini Models:**
- **200 comparisons with gemini-1.5-flash**: ~$1-2 USD
- **200 comparisons with gemini-1.5-pro**: ~$3-4 USD

**Processing time**: ~15-20 minutes (with rate limiting)

## 🎯 How It Works

1. **Pairs resumes**: Matches US resume #1 with UK resume #1, US #2 with UK #2, etc.
2. **Side-by-side comparison**: Presents both resumes to AI simultaneously
3. **Hiring decision**: Asks AI to choose which candidate to hire
4. **Bias analysis**: Analyzes whether there's a pattern in US vs UK preferences

## 📈 Expected Results

The system will show:
- How many times AI chose US candidates
- How many times AI chose UK candidates
- Detailed reasoning for each decision
- Potential bias patterns in hiring preferences

## 🔄 Model Comparison

Compare results between different AI models:
- **OpenAI GPT**: Industry standard, extensive training data
- **Google Gemini**: Alternative perspective, potentially different biases 