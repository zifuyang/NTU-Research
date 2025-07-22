# Comparison Resume Bias Detector

A Python script that compares US vs UK resumes side by side and asks ChatGPT to choose which applicant to hire, detecting potential hiring bias.

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

3. **Run comparison bias detection:**
   ```bash
   python comparison_bias_detector.py resumes_for_bias_detection.csv
   ```

## 📁 Files

- `comparison_bias_detector.py` - Main comparison bias detection script
- `resumes_for_bias_detection.csv` - Your converted resume data (200 US + 200 UK resumes)
- `Corpora.csv` - Your original data
- `requirements_fast.txt` - Python dependencies

## 📊 Output

The script outputs a CSV file with:
- `comparison_id`: COMP_001, COMP_002, etc.
- `us_resume_id`: US resume identifier
- `us_university`: US candidate's university
- `us_resume_content`: US candidate's resume
- `uk_resume_id`: UK resume identifier
- `uk_university`: UK candidate's university
- `uk_resume_content`: UK candidate's resume
- `ai_response`: Full ChatGPT response
- `verdict`: US or UK (which candidate ChatGPT chose)
- `reasoning`: ChatGPT's explanation
- `status`: success/error

## ⚙️ Options

```bash
# Custom output file
python comparison_bias_detector.py resumes_for_bias_detection.csv --output results.csv

# Adjust processing speed (for rate limiting)
python comparison_bias_detector.py resumes_for_bias_detection.csv --batch-size 2 --delay 25

# Use different model
python comparison_bias_detector.py resumes_for_bias_detection.csv --model gpt-4o
```

## 💰 Cost Estimate

- **200 comparisons with GPT-4o-mini**: ~$2-3 USD
- **200 comparisons with GPT-4o**: ~$4-6 USD
- **Processing time**: ~15-20 minutes (with rate limiting)

## 🎯 How It Works

1. **Pairs resumes**: Matches US resume #1 with UK resume #1, US #2 with UK #2, etc.
2. **Side-by-side comparison**: Presents both resumes to ChatGPT simultaneously
3. **Hiring decision**: Asks ChatGPT to choose which candidate to hire
4. **Bias analysis**: Analyzes whether there's a pattern in US vs UK preferences

## 📈 Expected Results

The system will show:
- How many times ChatGPT chose US candidates
- How many times ChatGPT chose UK candidates
- Detailed reasoning for each decision
- Potential bias patterns in hiring preferences 