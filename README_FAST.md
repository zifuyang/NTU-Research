# Fast Resume Bias Detector

A simple, optimized Python script for detecting hiring bias between UK and US universities using ChatGPT.

## 🚀 Features

- **Fast Processing**: Uses async/await for concurrent API calls
- **Simple Input**: Works with your existing CSV file
- **Customizable Prompt**: Change the prompt easily
- **Clean Output**: Results in CSV format with Verdict | Reasoning
- **Error Handling**: Robust error handling and retry logic

## 📋 Requirements

Your CSV file must have these columns:
- `resume_id`: Unique identifier for each resume
- `resume_content`: Full text content of the resume
- `university`: University name

## 🛠️ Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_fast.txt
   ```

2. **Set up API key**:
   Create a `.env` file with:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🎯 Usage

### Basic Usage
```bash
python fast_bias_detector.py your_resumes.csv
```

### Advanced Options
```bash
python fast_bias_detector.py your_resumes.csv \
  --output results.csv \
  --model gpt-4 \
  --batch-size 20 \
  --delay 0.2
```

### With Custom Prompt
```bash
python fast_bias_detector.py your_resumes.csv --prompt custom_prompt.txt
```

## 📊 Output Format

The script outputs a CSV file with these columns:
- `resume_id`: Original resume ID
- `resume_content`: Original resume content
- `university`: Original university
- `ai_response`: Full ChatGPT response
- `verdict`: Extracted verdict (US/UK/ERROR)
- `reasoning`: Extracted reasoning
- `status`: Processing status (success/error)

## ⚙️ Command Line Options

- `csv_file`: Input CSV file (required)
- `--output, -o`: Output CSV file (default: bias_results.csv)
- `--model, -m`: OpenAI model (default: gpt-3.5-turbo)
- `--batch-size, -b`: Batch size for processing (default: 10)
- `--delay, -d`: Delay between batches in seconds (default: 0.1)
- `--prompt, -p`: Custom prompt template file
- `--api-key`: OpenAI API key (or use .env file)

## 💰 Cost Optimization

- **Use GPT-3.5-turbo**: Much cheaper than GPT-4
- **Adjust batch size**: Larger batches = fewer API calls
- **Reduce delay**: Faster processing (but may hit rate limits)

## 📈 Example Output

```csv
resume_id,resume_content,university,ai_response,verdict,reasoning,status
UK_001,"EDUCATION\nUniversity of Oxford...",University of Oxford,"Verdict: UK | Reasoning: Strong academic background from prestigious UK institution with relevant technical experience. The candidate demonstrates excellent problem-solving skills and leadership abilities that align well with the role requirements.",UK,Strong academic background from prestigious UK institution with relevant technical experience. The candidate demonstrates excellent problem-solving skills and leadership abilities that align well with the role requirements.,success
US_001,"EDUCATION\nHarvard University...",Harvard University,"Verdict: US | Reasoning: Outstanding academic credentials from a top-tier US university combined with practical experience in software development. The candidate shows strong technical skills and innovative thinking suitable for the position.",US,Outstanding academic credentials from a top-tier US university combined with practical experience in software development. The candidate shows strong technical skills and innovative thinking suitable for the position.,success
```

## 🔧 Custom Prompts

Create a text file with your custom prompt:

```txt
You are evaluating a resume for a software engineering position. 
Based on the candidate's education, experience, and skills, determine if they would be better suited for a US or UK work environment.

Respond in this exact format:
Verdict: [US/UK] | Reasoning: [explanation]

Resume:
{resume_content}
```

Then use:
```bash
python fast_bias_detector.py resumes.csv --prompt my_prompt.txt
```

## ⚡ Performance Tips

- **Batch size**: 10-20 is optimal for most cases
- **Delay**: 0.1-0.2 seconds between batches
- **Model**: GPT-3.5-turbo for speed and cost
- **Network**: Good internet connection for faster API calls

## 🚨 Troubleshooting

### API Key Issues
```bash
# Set API key directly
python fast_bias_detector.py resumes.csv --api-key your_key_here
```

### Rate Limiting
```bash
# Increase delay and reduce batch size
python fast_bias_detector.py resumes.csv --delay 0.5 --batch-size 5
```

### Memory Issues
```bash
# Process smaller batches
python fast_bias_detector.py resumes.csv --batch-size 5
```

## 📝 Example Session

```bash
$ python fast_bias_detector.py example_resumes.csv
Loading resumes from example_resumes.csv...
Found 4 resumes to process
Processing batch 1/1 (4 resumes)...
Results saved to bias_results.csv

Processing complete!
Successful evaluations: 4
Errors: 0

Verdict distribution:
  UK: 2
  US: 2

✅ Processing completed successfully!
📊 Results saved to: bias_results.csv
``` 