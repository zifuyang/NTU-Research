# NTU Research: LLM Hiring Bias Analysis

## Project Overview
This research project analyzes hiring bias in Large Language Models (LLMs) by comparing selection rates between US and UK candidates across different AI models. The study focuses on Equal Opportunity violations and provides comprehensive statistical analysis with visualizations.

## Key Research Questions
1. Do LLMs exhibit bias in hiring decisions between US and UK candidates?
2. Which models show the highest/lowest bias levels?
3. Are the observed differences statistically significant?
4. Do the models exceed established fairness thresholds?

## Models Analyzed
- **Claude Sonnet 4** (Anthropic)
- **GPT-4o Mini** (OpenAI) 
- **Gemini 2.5 Flash** (Google)

## Key Findings

### Equal Opportunity Gap Results
| Model | US Selection Rate | UK Selection Rate | Gap | Statistical Significance |
|-------|------------------|------------------|-----|-------------------------|
| Claude | 88.5% | 11.5% | 77.0% | p ≤ 0.05 |
| Gemini | 71.0% | 29.0% | 42.0% | p ≤ 0.05 |
| OpenAI | 59.0% | 41.0% | 18.0% | p ≤ 0.05 |
| **Aggregated** | **72.5%** | **27.5%** | **45.0%** | **p ≤ 0.05** |

### Critical Insights
- **All models exceed the 15% fairness gap threshold**
- **Claude shows the highest bias (77% gap)**
- **OpenAI shows the lowest bias but still significant (18% gap)**
- **Statistical significance confirmed for all models**

## Project Structure
```
NTU-Research/
├── analysis/          # Analysis scripts
├── data/             # Raw data files
├── figures/          # Generated visualizations
├── scripts/          # Utility scripts
├── docs/             # Documentation
└── Results/          # LLM test results
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed organization.

## Quick Start

### Prerequisites
```bash
pip install pandas matplotlib seaborn scipy numpy
```

### Run Equal Opportunity Analysis
```bash
cd analysis/equal_opportunity/
python3 equal_opportunity_analysis.py
```

### Run Model Comparison
```bash
cd analysis/comparison/
python3 create_comparison_figure.py
```

## Analysis Types

### 1. Equal Opportunity Gap Analysis
- Measures True Positive Rate (TPR) differences between US/UK candidates
- Tests statistical significance using Fisher's Exact Test
- Visualizes gaps with confidence intervals
- Flags violations of 15% fairness threshold

### 2. Model Comparison Analysis
- Side-by-side comparison of all three models
- Grouped bar charts showing selection rates
- Statistical significance indicators
- Aggregate analysis across all models

### 3. TPR Analysis
- Detailed True Positive Rate analysis
- Confusion matrix visualization
- P-value plots with log scale
- Combined heatmap analysis

## Visualizations Generated

### Figure 1: Grouped Bar Chart
- TPR comparison for UK vs US candidates
- Error bars showing 95% confidence intervals
- Statistical significance markers (* p ≤ 0.05)
- Fairness gap threshold line

### Figure 2: TPR Gap Plot
- Gap visualization with threshold overlay
- Statistical significance indicators
- Exact gap annotations

### Figure 3: P-value Plot
- Log-scale p-value visualization
- Significance threshold highlighting
- Model comparison of statistical evidence

### Figure 4: Confusion Matrix
- Detailed breakdown for selected model
- True Positive/False Negative counts
- Visual representation of selection patterns

### Figure 5: Combined Heatmap
- Cross-model and cross-group comparison
- Color-coded selection rates
- Comprehensive bias overview

## Data Sources
- **Resumes.csv**: Resume data for hiring simulations
- **Corpora.csv**: Corpus data for analysis
- **Results/**: Organized test results by model and trial

## Statistical Methods
- **Fisher's Exact Test**: For statistical significance
- **Wilson Score Interval**: For confidence intervals
- **Equal Opportunity Gap**: TPR difference measurement
- **15% Threshold**: Fairness violation flag

## Research Implications
1. **Substantial Bias**: All tested models show significant bias
2. **Model Variation**: Different models exhibit varying bias levels
3. **Fairness Violations**: All models exceed established thresholds
4. **Need for Mitigation**: Urgent need for bias reduction techniques

## Future Work
- Bias mitigation strategies
- Additional model testing
- Cross-cultural bias analysis
- Fairness-aware model training

## Contact
For questions about this research, please refer to the project documentation in the `docs/` directory.

## License
This research project is for academic purposes. Please cite appropriately if using these findings. 