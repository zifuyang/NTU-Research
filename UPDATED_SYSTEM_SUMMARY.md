# Updated Resume Bias Detection System

## 🎯 System Overview

The system has been successfully updated to work with your specific requirements:

1. **Your existing Excel spreadsheet** with resumes from US and UK universities
2. **Your specific prompt format**: "Recommendation: [SCHOOL NAME] | Reasoning: [explanation]"
3. **Output format**: AI Verdict | Reasoning

## ✅ What's Been Updated

### 1. Prompt Configuration
- **Updated** `config.py` to use your exact prompt format
- **Modified** the evaluation prompt to match your specifications
- **Changed** from JSON output to structured text output

### 2. Response Processing
- **Updated** `resume_processor.py` to parse your specific format
- **Added** `extract_verdict_from_response()` function
- **Modified** to extract "AI Verdict" and "Reasoning" from responses

### 3. Statistical Analysis
- **Updated** `bias_analyzer.py` to work with school recommendations instead of ratings
- **Added** Chi-square tests for recommendation patterns
- **Added** Fisher's exact test for UK bias detection
- **Modified** visualizations to show recommendation distributions

### 4. Excel File Support
- **Added** `prepare_excel.py` to validate your existing Excel files
- **Created** template generation for reference
- **Updated** main script to skip data generation when using existing data

## 🚀 How to Use Your Existing Excel Data

### Step 1: Prepare Your Excel File
Your Excel file should have these columns:
- `resume_id`: Unique identifier for each resume
- `resume_content`: Full text content of the resume
- `university`: University name

### Step 2: Validate Your File
```bash
python3 prepare_excel.py
```

### Step 3: Run the Analysis
```bash
python3 main.py --step full --skip-data-generation
```

## 📊 What the System Does

### 1. Resume Processing
- Reads your Excel file with resumes
- Sends each resume to ChatGPT with your prompt
- Extracts the AI verdict (school name) and reasoning
- Saves structured results

### 2. Bias Analysis
- Analyzes whether UK candidates are more likely to be recommended UK schools
- Analyzes whether US candidates are more likely to be recommended US schools
- Performs statistical tests to detect significant bias
- Generates visualizations and reports

### 3. Output Files
- `results/bias_analysis_results.csv`: All AI evaluations
- `results/bias_analysis_report.txt`: Statistical analysis
- `results/bias_analysis_visualizations.png`: Charts and graphs

## 🔍 Statistical Tests Used

### 1. Chi-Square Test
- Tests for overall bias in recommendation patterns
- Compares UK vs US candidates and their recommendations

### 2. Fisher's Exact Test
- Tests specifically for UK bias
- Determines if UK candidates are favored for UK schools

### 3. Confidence Intervals
- Wilson confidence intervals for recommendation proportions
- Shows uncertainty in bias estimates

## 📈 Example Output

### AI Response Format
```
Recommendation: University of Oxford | Reasoning: Strong academic background with relevant experience in software development. The candidate demonstrates excellent technical skills and leadership abilities that align well with the role requirements.
```

### Analysis Results
- **UK Candidates → UK Schools**: 65%
- **UK Candidates → US Schools**: 25%
- **US Candidates → UK Schools**: 30%
- **US Candidates → US Schools**: 60%

### Statistical Significance
- Chi-square test: p < 0.05 (significant bias detected)
- Fisher's test: p < 0.01 (strong UK bias)

## 🛠️ Configuration Options

### Key Settings in `config.py`
```python
# OpenAI Configuration
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for cost savings
MAX_TOKENS = 1000
TEMPERATURE = 0.1  # Low temperature for consistent responses

# Processing Configuration
BATCH_SIZE = 10  # Process 10 resumes at a time
DELAY_BETWEEN_REQUESTS = 1  # 1 second between API calls
MAX_RETRIES = 3  # Retry failed requests up to 3 times
```

## 💰 Cost Optimization

### Estimated Costs for 100 Resumes
- **GPT-3.5-turbo**: ~$2-5
- **GPT-4**: ~$20-50

### Cost Reduction Tips
1. Use GPT-3.5-turbo instead of GPT-4
2. Reduce MAX_TOKENS if responses are too long
3. Increase BATCH_SIZE for fewer API calls
4. Test with a small sample first

## 🔧 Troubleshooting

### Common Issues

1. **Excel File Not Found**
   ```bash
   python3 prepare_excel.py
   ```

2. **Missing Columns**
   - Ensure your Excel has: resume_id, resume_content, university
   - Use the template: `python3 prepare_excel.py --create-template`

3. **API Key Issues**
   - Check your `.env` file has the correct OpenAI API key
   - Ensure the API key has sufficient credits

4. **Rate Limiting**
   - Increase `DELAY_BETWEEN_REQUESTS` in `config.py`
   - Reduce `BATCH_SIZE` for slower processing

## 📋 Quick Start Checklist

- [ ] Install dependencies: `./install.sh`
- [ ] Set OpenAI API key in `.env` file
- [ ] Place your Excel file as `resume_dataset.xlsx`
- [ ] Validate your file: `python3 prepare_excel.py`
- [ ] Run analysis: `python3 main.py --step full --skip-data-generation`
- [ ] Check results in `results/` directory

## 🎉 Expected Results

The system will:
1. Process all resumes in your Excel file
2. Generate AI evaluations using your prompt format
3. Analyze for bias between UK and US universities
4. Provide statistical evidence of bias (if any)
5. Create visualizations and detailed reports

This updated system is specifically designed to work with your existing data and prompt format, making it easy to detect hiring bias between UK and US universities using your preferred evaluation approach. 