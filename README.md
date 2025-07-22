# Resume Bias Detection Project

A comprehensive Python system for detecting hiring bias between UK and US universities through automated resume evaluation using OpenAI's GPT models.

## 🎯 Project Overview

This project automates the submission and evaluation of 100 resumes to ChatGPT for detecting hiring bias between UK and US universities. The system provides:

- **Batch processing** of resumes from Excel datasets
- **Consistent evaluation** using structured prompts
- **Statistical analysis** for bias detection
- **Comprehensive reporting** with visualizations
- **Cost and time optimization** through batch processing

## 📋 Features

### Core Functionality
- ✅ Process 100+ resumes without manual intervention
- ✅ Maintain consistent evaluation conditions
- ✅ Generate structured output for statistical analysis
- ✅ Complete processing within reasonable time/cost constraints
- ✅ Robust error handling and retry mechanisms
- ✅ Intermediate result saving for fault tolerance

### Analysis Capabilities
- ✅ Statistical tests (T-test, Mann-Whitney U, Chi-square)
- ✅ Confidence interval calculations
- ✅ Bias pattern detection
- ✅ Comprehensive visualizations
- ✅ Detailed analysis reports

### Data Management
- ✅ Excel file input/output
- ✅ Sample data generation for testing
- ✅ Structured JSON responses from OpenAI
- ✅ Progress tracking and logging
- ✅ Batch processing with rate limiting

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd NTU-Research

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run the Complete Workflow

```bash
# Run the entire pipeline (generates data, processes resumes, analyzes bias)
python main.py --step full --num-resumes 100
```

## 📁 Project Structure

```
NTU-Research/
├── main.py                 # Main orchestration script
├── config.py              # Configuration settings
├── resume_processor.py    # Core resume processing logic
├── bias_analyzer.py       # Statistical bias analysis
├── sample_data_generator.py # Sample data generation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .env                  # Environment variables (create this)
├── resume_dataset.xlsx   # Input dataset (generated or provided)
└── results/              # Output directory (created automatically)
    ├── bias_analysis_results.csv
    ├── bias_analysis_report.txt
    ├── bias_analysis_visualizations.png
    └── processing_summary.json
```

## 🔧 Usage Examples

### Generate Sample Data Only

```bash
python main.py --step generate_data --num-resumes 100
```

### Process Existing Resume Data

```bash
python main.py --step process_resumes
```

### Analyze Results for Bias

```bash
python main.py --step analyze_bias
```

### Run with Custom Number of Resumes

```bash
python main.py --step full --num-resumes 50
```

### Skip Data Generation (Use Existing Data)

```bash
python main.py --step full --skip-data-generation
```

## 📊 Input Data Format

The system expects an Excel file (`resume_dataset.xlsx`) with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `resume_id` | String | Unique identifier for each resume |
| `resume_content` | String | Full text content of the resume |
| `university` | String | University name |

### Example Input Data

```python
import pandas as pd

# Sample data structure
data = {
    'resume_id': ['UK_001', 'US_001', 'UK_002'],
    'resume_content': [
        'EDUCATION\nUniversity of Oxford\nComputer Science...',
        'EDUCATION\nHarvard University\nComputer Science...',
        'EDUCATION\nUniversity of Cambridge\nComputer Science...'
    ],
    'university': ['University of Oxford', 'Harvard University', 'University of Cambridge']
}

df = pd.DataFrame(data)
df.to_excel('resume_dataset.xlsx', index=False)
```

### Using Your Existing Excel File

If you already have an Excel file with resume data:

1. **Validate your file**:
   ```bash
   python3 prepare_excel.py
   ```

2. **Create a template** (if needed):
   ```bash
   python3 prepare_excel.py --create-template
   ```

3. **Run with existing data**:
   ```bash
   python3 main.py --step full --skip-data-generation
   ```

## 📈 Output Results

### 1. Evaluation Results (`bias_analysis_results.csv`)

Contains structured evaluation data for each resume:

- `resume_id`: Original resume identifier
- `university`: University name
- `university_category`: UK/US/Other classification
- `ai_verdict`: School name recommended by AI
- `reasoning`: AI's reasoning for the recommendation
- `raw_response`: Complete AI response
- `processing_timestamp`: When the evaluation was processed

### 2. Analysis Report (`bias_analysis_report.txt`)

Comprehensive statistical analysis including:

- Descriptive statistics by university category
- Statistical test results (Chi-square, Fisher's exact test)
- Bias pattern detection in school recommendations
- Confidence intervals for recommendation proportions
- Conclusions and recommendations

### 3. Visualizations (`bias_analysis_visualizations.png`)

Four-panel visualization showing:

- Recommendation distribution by candidate origin
- Recommendation rates (percentages)
- Top 10 most recommended schools
- Cross-category bias analysis

## ⚙️ Configuration

Key settings can be modified in `config.py`:

```python
# OpenAI Configuration
OPENAI_MODEL = "gpt-4"  # or "gpt-3.5-turbo"
MAX_TOKENS = 1000
TEMPERATURE = 0.1

# Processing Configuration
BATCH_SIZE = 10
DELAY_BETWEEN_REQUESTS = 1
MAX_RETRIES = 3

# Statistical Analysis
SIGNIFICANCE_LEVEL = 0.05
CONFIDENCE_INTERVAL = 0.95
```

## 🔍 Statistical Analysis Methods

The system employs multiple statistical tests for robust bias detection:

### 1. T-Test
- Compares mean overall ratings between UK and US universities
- Tests for significant differences in evaluation scores

### 2. Mann-Whitney U Test
- Non-parametric alternative to T-test
- Robust against non-normal distributions

### 3. Chi-Square Test
- Analyzes recommendation patterns (HIRE/REJECT rates)
- Detects bias in hiring decisions

### 4. Effect Size Calculation
- Cohen's d for standardized mean differences
- Quantifies practical significance

## 🛠️ Error Handling

The system includes comprehensive error handling:

- **API Rate Limiting**: Automatic delays between requests
- **Retry Logic**: Up to 3 attempts for failed API calls
- **Intermediate Saving**: Results saved after each batch
- **Logging**: Detailed logs for debugging
- **Graceful Degradation**: Continues processing even if some resumes fail

## 💰 Cost Optimization

To minimize OpenAI API costs:

1. **Use GPT-3.5-turbo** instead of GPT-4 (modify `config.py`)
2. **Reduce MAX_TOKENS** for shorter responses
3. **Increase BATCH_SIZE** for fewer API calls
4. **Use sample data** for testing before running on real data

## 🧪 Testing

### Generate and Test with Sample Data

```bash
# Generate 10 sample resumes for testing
python main.py --step generate_data --num-resumes 10

# Run complete workflow with sample data
python main.py --step full --num-resumes 10
```

### Validate Results

Check the generated files in the `results/` directory:
- Verify `bias_analysis_results.csv` contains expected columns
- Review `bias_analysis_report.txt` for statistical validity
- Examine visualizations for data quality

## 📝 Logging

The system provides comprehensive logging:

- **File Logging**: `resume_processing.log`
- **Console Output**: Real-time progress updates
- **Error Tracking**: Detailed error messages and stack traces
- **Performance Metrics**: Processing times and success rates

## 🔒 Privacy and Ethics

### Data Privacy
- No personal information is stored or transmitted
- Sample data uses fictional universities and content
- API calls are made with minimal data exposure

### Ethical Considerations
- System designed to detect, not perpetuate bias
- Transparent evaluation criteria
- Reproducible and auditable results
- Configurable for different evaluation contexts

## 🚨 Troubleshooting

### Common Issues

1. **OpenAI API Key Not Found**
   ```bash
   # Ensure .env file exists with:
   OPENAI_API_KEY=your_key_here
   ```

2. **Rate Limiting Errors**
   - Increase `DELAY_BETWEEN_REQUESTS` in `config.py`
   - Reduce `BATCH_SIZE` for slower processing

3. **Memory Issues with Large Datasets**
   - Process in smaller batches
   - Use `--num-resumes` to limit dataset size

4. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Debug Mode

Enable detailed logging by modifying the logging level in `resume_processor.py`:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 API Reference

### ResumeProcessor Class

```python
processor = ResumeProcessor()
results_df = processor.run()
```

### BiasAnalyzer Class

```python
analyzer = BiasAnalyzer()
results = analyzer.run_analysis()
```

### SampleDataGenerator Class

```python
generator = SampleDataGenerator()
df = generator.generate_dataset(100)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for providing the GPT API
- The research community for bias detection methodologies
- Contributors and testers

---

**Note**: This system is designed for research purposes. Always ensure compliance with relevant data protection and privacy regulations when using real resume data. 