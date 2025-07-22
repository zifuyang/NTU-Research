# 🚀 Fast Bias Detector - Setup Guide

## ✅ What's Ready

Your bias detection system is **fully set up and tested**! Here's what we've accomplished:

1. **✅ Data Conversion**: Your `Corpora.csv` has been converted to the proper format
2. **✅ System Tested**: All components are working correctly
3. **✅ Demo Completed**: Successfully processed 10 sample resumes
4. **✅ Dependencies Installed**: All required packages are ready

## 📊 Your Data Summary

- **Total Resumes**: 400 (200 US + 200 UK)
- **Converted File**: `resumes_for_bias_detection.csv`
- **Format**: Ready for bias detection processing

## 🎯 Next Steps to Use Real ChatGPT API

### 1. Get OpenAI API Key
1. Go to [OpenAI Platform](https://platform.openai.com/api-keys)
2. Create a new API key
3. Copy the key (starts with `sk-`)

### 2. Set Up API Key
Create a `.env` file in your project directory:
```bash
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

**OR** pass it directly when running:
```bash
python3 fast_bias_detector.py resumes_for_bias_detection.csv --api-key your_api_key_here
```

### 3. Run the Bias Detection
```bash
# Process all 400 resumes
python3 fast_bias_detector.py resumes_for_bias_detection.csv

# Or process with custom settings
python3 fast_bias_detector.py resumes_for_bias_detection.csv \
  --output bias_results.csv \
  --batch-size 20 \
  --delay 0.2
```

## 📈 Expected Results

The system will output a CSV file with:
- `resume_id`: US_001, UK_002, etc.
- `resume_content`: Original resume text
- `university`: Extracted university name
- `ai_response`: Full ChatGPT response
- `verdict`: US or UK
- `reasoning`: ChatGPT's explanation
- `status`: success/error

## 💰 Cost Estimate

- **Model**: GPT-3.5-turbo (recommended for cost efficiency)
- **400 resumes**: ~$2-4 USD
- **Processing time**: ~10-15 minutes

## ⚙️ Customization Options

### Change the Prompt
Create a custom prompt file:
```txt
You are evaluating resumes for bias between US and UK universities.
Based on the candidate's background, determine if they would be better suited for US or UK work environments.

Respond in this format:
Verdict: [US/UK] | Reasoning: [explanation]

Resume:
{resume_content}
```

Then use:
```bash
python3 fast_bias_detector.py resumes_for_bias_detection.csv --prompt my_prompt.txt
```

### Adjust Processing Speed
```bash
# Faster processing (may hit rate limits)
python3 fast_bias_detector.py resumes_for_bias_detection.csv --batch-size 30 --delay 0.1

# Slower, more reliable
python3 fast_bias_detector.py resumes_for_bias_detection.csv --batch-size 10 --delay 0.5
```

## 🔍 Sample Output Format

```csv
resume_id,resume_content,university,ai_response,verdict,reasoning,status
US_001,"Education: MIT. John Doe...",MIT,"Verdict: US | Reasoning: Strong technical background...",US,Strong technical background...,success
UK_002,"Education: Imperial College...",Imperial College London,"Verdict: UK | Reasoning: Excellent academic credentials...",UK,Excellent academic credentials...,success
```

## 🚨 Troubleshooting

### API Key Issues
```bash
# Check if API key is set
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', 'SET' if os.getenv('OPENAI_API_KEY') else 'NOT SET')"
```

### Rate Limiting
If you get rate limit errors, increase the delay:
```bash
python3 fast_bias_detector.py resumes_for_bias_detection.csv --delay 1.0
```

### Memory Issues
Process smaller batches:
```bash
python3 fast_bias_detector.py resumes_for_bias_detection.csv --batch-size 5
```

## 📞 Support

If you encounter any issues:
1. Check the error messages
2. Verify your API key is correct
3. Try with smaller batch sizes
4. Check your internet connection

## 🎉 Ready to Go!

Your system is fully prepared. Just add your OpenAI API key and run the bias detection on all 400 resumes! 