# NTU Research Project Structure

## Overview
This project analyzes hiring bias in Large Language Models (LLMs) by comparing selection rates between US and UK candidates across different AI models.

## Directory Structure

```
NTU-Research/
├── analysis/                          # Analysis scripts organized by type
│   ├── equal_opportunity/            # Equal Opportunity Gap analysis
│   │   └── equal_opportunity_analysis.py
│   ├── comparison/                   # Model comparison analysis
│   │   └── create_comparison_figure.py
│   └── tpr_analysis/                 # True Positive Rate analysis
│       ├── Claude Tests.py
│       └── Claude Tests Improved.py
├── data/                             # Raw data files
│   ├── Corpora.csv                   # Corpus data
│   ├── Resumes.csv                   # Resume data
│   ├── example_resumes.csv           # Example resume data
│   └── claude_results_trial*.csv     # Claude test results
├── figures/                          # Generated visualizations
│   ├── equal_opportunity/            # Equal Opportunity Gap figures
│   │   ├── detailed_equal_opportunity_analysis.png
│   │   └── [other equal opportunity figures]
│   ├── comparison/                   # Model comparison figures
│   │   └── [comparison figures]
│   └── tpr_analysis/                 # TPR analysis figures
│       └── [TPR analysis figures]
├── scripts/                          # Utility scripts
│   ├── fast_bias_detector.py         # Fast bias detection utility
│   └── test_fast.py                  # Test script for fast bias detector
├── docs/                             # Documentation
│   ├── README.md                     # Main project documentation
│   ├── README_FAST.md                # Fast bias detector documentation
│   └── requirements_fast.txt         # Python dependencies
├── Results/                          # LLM test results (organized by model)
│   ├── Claude/                       # Claude model results
│   ├── OpenAI/                       # OpenAI/GPT model results
│   ├── Gemini/                       # Gemini model results
│   └── FAILED/                       # Failed test results
└── PROJECT_STRUCTURE.md              # This file

```

## Key Components

### Analysis Scripts
- **Equal Opportunity Analysis**: Measures bias using TPR gaps between US/UK candidates
- **Comparison Analysis**: Compares different LLM models side-by-side
- **TPR Analysis**: Analyzes True Positive Rates for hiring decisions

### Data Files
- **Corpora.csv**: Contains corpus data for analysis
- **Resumes.csv**: Resume data used in hiring simulations
- **Example_resumes.csv**: Sample resume data for testing
- **Claude Results**: Multiple trial results from Claude model tests

### Results Organization
- **Claude/**: Results from Claude Sonnet 4 tests
- **OpenAI/**: Results from GPT model tests (various trials)
- **Gemini/**: Results from Gemini model tests
- **FAILED/**: Failed test runs for reference

### Generated Figures
- **Equal Opportunity**: Bias analysis visualizations
- **Comparison**: Model comparison charts
- **TPR Analysis**: True Positive Rate visualizations

## Usage

### Running Equal Opportunity Analysis
```bash
cd analysis/equal_opportunity/
python3 equal_opportunity_analysis.py
```

### Running Model Comparison
```bash
cd analysis/comparison/
python3 create_comparison_figure.py
```

### Running TPR Analysis
```bash
cd analysis/tpr_analysis/
python3 "Claude Tests Improved.py"
```

## Key Findings

### Equal Opportunity Gap Results
- **Claude**: 77.0% gap (highest bias)
- **Gemini**: 42.0% gap (moderate bias)
- **OpenAI**: 18.0% gap (lowest bias)
- **Aggregated**: 45.0% gap (overall system bias)

All models exceed the 15% fairness gap threshold, indicating substantial Equal Opportunity violations.

## Dependencies
- Python 3.x
- pandas
- matplotlib
- seaborn
- scipy
- numpy

## Notes
- All PNG figures are high-resolution (300 DPI) suitable for publication
- Analysis scripts include statistical significance testing
- Results are organized by model and trial for easy comparison 