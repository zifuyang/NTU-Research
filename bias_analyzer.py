"""
Bias Analysis Module
Statistical analysis of resume evaluation results to detect hiring bias
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
import warnings
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime

from config import SIGNIFICANCE_LEVEL, CONFIDENCE_INTERVAL, RESULTS_DIR

warnings.filterwarnings('ignore')

class BiasAnalyzer:
    """Class for analyzing bias in resume evaluation results"""
    
    def __init__(self, results_file: str = None):
        """Initialize the bias analyzer"""
        if results_file is None:
            results_file = os.path.join(RESULTS_DIR, "bias_analysis_results.csv")
        
        self.results_file = results_file
        self.df = None
        self.analysis_results = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def load_results(self) -> pd.DataFrame:
        """Load the evaluation results"""
        try:
            self.df = pd.read_csv(self.results_file)
            print(f"Loaded {len(self.df)} evaluation results")
            return self.df
        except FileNotFoundError:
            print(f"Results file {self.results_file} not found")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        if self.df is None:
            self.load_results()
        
        # Filter for UK and US universities only
        self.df = self.df[self.df['university_category'].isin(['UK', 'US'])]
        
        # Remove rows with missing values in key columns
        self.df = self.df.dropna(subset=['university_category'])
        
        # Create binary recommendation column for analysis
        if 'ai_verdict' in self.df.columns:
            # Extract school names from AI verdicts for analysis
            self.df['recommended_school'] = self.df['ai_verdict'].apply(
                lambda x: str(x).strip() if pd.notna(x) else 'UNKNOWN'
            )
        
        print(f"Cleaned data: {len(self.df)} valid evaluations")
        return self.df
    
    def descriptive_statistics(self) -> Dict:
        """Calculate descriptive statistics by university category"""
        stats_dict = {}
        
        for category in ['UK', 'US']:
            category_data = self.df[self.df['university_category'] == category]
            
            stats_dict[category] = {
                'count': len(category_data),
                'total_evaluations': len(category_data),
                'uk_recommendations': len(category_data[category_data['recommended_school'].isin(UK_UNIVERSITIES)]),
                'us_recommendations': len(category_data[category_data['recommended_school'].isin(US_UNIVERSITIES)]),
                'other_recommendations': len(category_data[~category_data['recommended_school'].isin(UK_UNIVERSITIES + US_UNIVERSITIES)])
            }
        
        self.analysis_results['descriptive_stats'] = stats_dict
        return stats_dict
    
    def statistical_tests(self) -> Dict:
        """Perform statistical tests for bias detection"""
        uk_data = self.df[self.df['university_category'] == 'UK']
        us_data = self.df[self.df['university_category'] == 'US']
        
        tests_results = {}
        
        # Test for recommendation bias using Chi-square test
        if len(uk_data) > 0 and len(us_data) > 0:
            # Create contingency table for UK vs US recommendations
            uk_recommendations = uk_data['recommended_school'].apply(
                lambda x: 'UK' if x in UK_UNIVERSITIES else ('US' if x in US_UNIVERSITIES else 'Other')
            )
            us_recommendations = us_data['recommended_school'].apply(
                lambda x: 'UK' if x in UK_UNIVERSITIES else ('US' if x in US_UNIVERSITIES else 'Other')
            )
            
            # Chi-square test for recommendation patterns
            recommendation_contingency = pd.crosstab(
                pd.concat([uk_recommendations, us_recommendations]),
                pd.concat([pd.Series(['UK'] * len(uk_recommendations)), 
                          pd.Series(['US'] * len(us_recommendations))])
            )
            
            if recommendation_contingency.shape[0] >= 2 and recommendation_contingency.shape[1] >= 2:
                chi2_stat, chi2_p_value, _, _ = chi2_contingency(recommendation_contingency)
                tests_results['recommendation_bias_chi2'] = {
                    'chi2_statistic': chi2_stat,
                    'p_value': chi2_p_value,
                    'significant': chi2_p_value < SIGNIFICANCE_LEVEL
                }
            
            # Test for UK bias (UK candidates recommended more UK schools)
            uk_uk_recommendations = len(uk_data[uk_data['recommended_school'].isin(UK_UNIVERSITIES)])
            us_uk_recommendations = len(us_data[us_data['recommended_school'].isin(UK_UNIVERSITIES)])
            
            # Fisher's exact test for 2x2 contingency table
            from scipy.stats import fisher_exact
            contingency_table = [[uk_uk_recommendations, len(uk_data) - uk_uk_recommendations],
                               [us_uk_recommendations, len(us_data) - us_uk_recommendations]]
            
            fisher_stat, fisher_p_value = fisher_exact(contingency_table)
            tests_results['uk_bias_fisher'] = {
                'fisher_statistic': fisher_stat,
                'p_value': fisher_p_value,
                'significant': fisher_p_value < SIGNIFICANCE_LEVEL
            }
        
        self.analysis_results['statistical_tests'] = tests_results
        return tests_results
    
    def calculate_confidence_intervals(self) -> Dict:
        """Calculate confidence intervals for key metrics"""
        ci_results = {}
        
        for category in ['UK', 'US']:
            category_data = self.df[self.df['university_category'] == category]
            
            if len(category_data) > 0:
                n = len(category_data)
                
                # Calculate proportion of UK recommendations for each category
                uk_recommendations = len(category_data[category_data['recommended_school'].isin(UK_UNIVERSITIES)])
                uk_proportion = uk_recommendations / n
                
                # Wilson confidence interval for proportion
                from statsmodels.stats.proportion import proportion_confint
                ci_lower, ci_upper = proportion_confint(uk_recommendations, n, alpha=1-CONFIDENCE_INTERVAL, method='wilson')
                
                ci_results[category] = {
                    'uk_recommendation_proportion': uk_proportion,
                    'uk_recommendation_ci': (ci_lower, ci_upper),
                    'sample_size': n
                }
        
        self.analysis_results['confidence_intervals'] = ci_results
        return ci_results
    
    def detect_bias_patterns(self) -> Dict:
        """Detect specific bias patterns in the data"""
        bias_patterns = {}
        
        # Check for bias in recommendation patterns
        uk_data = self.df[self.df['university_category'] == 'UK']
        us_data = self.df[self.df['university_category'] == 'US']
        
        if len(uk_data) > 0 and len(us_data) > 0:
            # UK candidates recommended UK schools
            uk_uk_recommendations = len(uk_data[uk_data['recommended_school'].isin(UK_UNIVERSITIES)])
            uk_uk_rate = uk_uk_recommendations / len(uk_data)
            
            # US candidates recommended UK schools
            us_uk_recommendations = len(us_data[us_data['recommended_school'].isin(UK_UNIVERSITIES)])
            us_uk_rate = us_uk_recommendations / len(us_data)
            
            # UK candidates recommended US schools
            uk_us_recommendations = len(uk_data[uk_data['recommended_school'].isin(US_UNIVERSITIES)])
            uk_us_rate = uk_us_recommendations / len(uk_data)
            
            # US candidates recommended US schools
            us_us_recommendations = len(us_data[us_data['recommended_school'].isin(US_UNIVERSITIES)])
            us_us_rate = us_us_recommendations / len(us_data)
            
            bias_patterns['recommendation_bias'] = {
                'uk_candidates_uk_recommendations': uk_uk_rate,
                'us_candidates_uk_recommendations': us_uk_rate,
                'uk_candidates_us_recommendations': uk_us_rate,
                'us_candidates_us_recommendations': us_us_rate,
                'uk_favored_by_uk': uk_uk_rate > us_uk_rate,
                'us_favored_by_us': us_us_rate > uk_us_rate
            }
        
        self.analysis_results['bias_patterns'] = bias_patterns
        return bias_patterns
    
    def generate_visualizations(self):
        """Generate visualizations for bias analysis"""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Resume Evaluation Bias Analysis: UK vs US Universities', fontsize=16, fontweight='bold')
        
        # 1. Recommendation Distribution by Candidate Origin
        ax1 = axes[0, 0]
        uk_data = self.df[self.df['university_category'] == 'UK']
        us_data = self.df[self.df['university_category'] == 'US']
        
        uk_uk_rec = len(uk_data[uk_data['recommended_school'].isin(UK_UNIVERSITIES)])
        uk_us_rec = len(uk_data[uk_data['recommended_school'].isin(US_UNIVERSITIES)])
        uk_other_rec = len(uk_data) - uk_uk_rec - uk_us_rec
        
        us_uk_rec = len(us_data[us_data['recommended_school'].isin(UK_UNIVERSITIES)])
        us_us_rec = len(us_data[us_data['recommended_school'].isin(US_UNIVERSITIES)])
        us_other_rec = len(us_data) - us_uk_rec - us_us_rec
        
        categories = ['UK Candidates', 'US Candidates']
        uk_recs = [uk_uk_rec, us_uk_rec]
        us_recs = [uk_us_rec, us_us_rec]
        other_recs = [uk_other_rec, us_other_rec]
        
        x = np.arange(len(categories))
        width = 0.25
        
        ax1.bar(x - width, uk_recs, width, label='UK Schools Recommended', color='blue', alpha=0.7)
        ax1.bar(x, us_recs, width, label='US Schools Recommended', color='red', alpha=0.7)
        ax1.bar(x + width, other_recs, width, label='Other Schools', color='gray', alpha=0.7)
        
        ax1.set_xlabel('Candidate Origin')
        ax1.set_ylabel('Number of Recommendations')
        ax1.set_title('Recommendation Distribution by Candidate Origin')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Recommendation Rates (Percentages)
        ax2 = axes[0, 1]
        uk_total = len(uk_data)
        us_total = len(us_data)
        
        uk_uk_rate = uk_uk_rec / uk_total * 100 if uk_total > 0 else 0
        uk_us_rate = uk_us_rec / uk_total * 100 if uk_total > 0 else 0
        us_uk_rate = us_uk_rec / us_total * 100 if us_total > 0 else 0
        us_us_rate = us_us_rec / us_total * 100 if us_total > 0 else 0
        
        categories = ['UK Candidates', 'US Candidates']
        uk_rates = [uk_uk_rate, us_uk_rate]
        us_rates = [uk_us_rate, us_us_rate]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax2.bar(x - width/2, uk_rates, width, label='UK Schools (%)', color='blue', alpha=0.7)
        ax2.bar(x + width/2, us_rates, width, label='US Schools (%)', color='red', alpha=0.7)
        
        ax2.set_xlabel('Candidate Origin')
        ax2.set_ylabel('Recommendation Rate (%)')
        ax2.set_title('Recommendation Rates by Candidate Origin')
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Top Recommended Schools
        ax3 = axes[1, 0]
        school_counts = self.df['recommended_school'].value_counts().head(10)
        school_counts.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_xlabel('School Name')
        ax3.set_ylabel('Number of Recommendations')
        ax3.set_title('Top 10 Most Recommended Schools')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias Analysis Summary
        ax4 = axes[1, 1]
        bias_metrics = ['UK Candidates\n→ UK Schools', 'UK Candidates\n→ US Schools', 
                       'US Candidates\n→ UK Schools', 'US Candidates\n→ US Schools']
        bias_values = [uk_uk_rate, uk_us_rate, us_uk_rate, us_us_rate]
        colors = ['blue', 'red', 'lightblue', 'lightcoral']
        
        bars = ax4.bar(bias_metrics, bias_values, color=colors, alpha=0.7)
        ax4.set_ylabel('Recommendation Rate (%)')
        ax4.set_title('Bias Analysis: Cross-Category Recommendations')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, bias_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = os.path.join(RESULTS_DIR, 'bias_analysis_visualizations.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualizations saved to {plot_file}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive bias analysis report"""
        report = []
        report.append("=" * 60)
        report.append("RESUME EVALUATION BIAS ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total evaluations analyzed: {len(self.df)}")
        report.append("")
        
        # Descriptive Statistics
        report.append("DESCRIPTIVE STATISTICS")
        report.append("-" * 30)
        for category, stats in self.analysis_results.get('descriptive_stats', {}).items():
            report.append(f"\n{category} Candidates (n={stats['count']}):")
            report.append(f"  UK Schools Recommended: {stats['uk_recommendations']}")
            report.append(f"  US Schools Recommended: {stats['us_recommendations']}")
            report.append(f"  Other Schools Recommended: {stats['other_recommendations']}")
        
        # Statistical Tests
        report.append("\n\nSTATISTICAL TESTS")
        report.append("-" * 30)
        tests = self.analysis_results.get('statistical_tests', {})
        
        if 'recommendation_bias_chi2' in tests:
            chi2_test = tests['recommendation_bias_chi2']
            report.append(f"\nRecommendation Bias Chi-Square Test:")
            report.append(f"  Chi-square statistic: {chi2_test['chi2_statistic']:.3f}")
            report.append(f"  P-value: {chi2_test['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if chi2_test['significant'] else 'No'}")
        
        if 'uk_bias_fisher' in tests:
            fisher_test = tests['uk_bias_fisher']
            report.append(f"\nUK Bias Fisher's Exact Test:")
            report.append(f"  Fisher's statistic: {fisher_test['fisher_statistic']:.3f}")
            report.append(f"  P-value: {fisher_test['p_value']:.4f}")
            report.append(f"  Significant: {'Yes' if fisher_test['significant'] else 'No'}")
        
        # Bias Patterns
        report.append("\n\nBIAS PATTERNS")
        report.append("-" * 30)
        patterns = self.analysis_results.get('bias_patterns', {})
        
        if 'recommendation_bias' in patterns:
            rec_bias = patterns['recommendation_bias']
            report.append(f"\nRecommendation Bias Analysis:")
            report.append(f"  UK Candidates → UK Schools: {rec_bias['uk_candidates_uk_recommendations']:.1%}")
            report.append(f"  UK Candidates → US Schools: {rec_bias['uk_candidates_us_recommendations']:.1%}")
            report.append(f"  US Candidates → UK Schools: {rec_bias['us_candidates_uk_recommendations']:.1%}")
            report.append(f"  US Candidates → US Schools: {rec_bias['us_candidates_us_recommendations']:.1%}")
            
            if rec_bias['uk_favored_by_uk']:
                report.append(f"  → UK candidates favor UK schools")
            if rec_bias['us_favored_by_us']:
                report.append(f"  → US candidates favor US schools")
        
        # Conclusions
        report.append("\n\nCONCLUSIONS")
        report.append("-" * 30)
        
        # Determine if bias is detected
        bias_detected = False
        for test_name, test_result in tests.items():
            if test_result.get('significant', False):
                bias_detected = True
                break
        
        if bias_detected:
            report.append("⚠️  STATISTICALLY SIGNIFICANT BIAS DETECTED")
            report.append("The analysis reveals significant differences in evaluation outcomes")
            report.append("between UK and US universities, suggesting potential hiring bias.")
        else:
            report.append("✅ NO STATISTICALLY SIGNIFICANT BIAS DETECTED")
            report.append("The analysis does not reveal significant differences in evaluation")
            report.append("outcomes between UK and US universities.")
        
        report.append("\n" + "=" * 60)
        
        report_text = "\n".join(report)
        
        # Save report
        report_file = os.path.join(RESULTS_DIR, 'bias_analysis_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"Analysis report saved to {report_file}")
        return report_text
    
    def run_analysis(self) -> Dict:
        """Run complete bias analysis"""
        print("Starting bias analysis...")
        
        # Load and clean data
        self.load_results()
        self.clean_data()
        
        # Perform analyses
        self.descriptive_statistics()
        self.statistical_tests()
        self.calculate_confidence_intervals()
        self.detect_bias_patterns()
        
        # Generate visualizations and report
        self.generate_visualizations()
        report = self.generate_report()
        
        print("Bias analysis completed!")
        return self.analysis_results


def main():
    """Main function to run bias analysis"""
    analyzer = BiasAnalyzer()
    results = analyzer.run_analysis()
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    # Check for significant bias
    tests = results.get('statistical_tests', {})
    bias_detected = any(test.get('significant', False) for test in tests.values())
    
    if bias_detected:
        print("⚠️  STATISTICALLY SIGNIFICANT BIAS DETECTED")
    else:
        print("✅ NO STATISTICALLY SIGNIFICANT BIAS DETECTED")
    
    print(f"Results saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main() 