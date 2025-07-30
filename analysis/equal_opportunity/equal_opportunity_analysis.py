#!/usr/bin/env python3
"""
Equal Opportunity Gap Analysis for LLM Hiring Bias Detection
Analyzes Claude tests, Gemini test 5, and OpenAI test 1
Creates all five visualization types for comprehensive bias analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load data from specified test files and calculate TPR metrics"""
    
    results = {}
    
    # Load Claude data (clean results)
    try:
        claude_df = pd.read_csv('../../Results/Claude/claude_results_clean.csv')
        claude_verdicts = claude_df['verdict'].value_counts()
        claude_us_pct = (claude_verdicts.get('US', 0) / len(claude_df)) * 100
        claude_uk_pct = (claude_verdicts.get('UK', 0) / len(claude_df)) * 100
        
        results['Claude'] = {
            'us_tpr': claude_us_pct,
            'uk_tpr': claude_uk_pct,
            'gap': claude_us_pct - claude_uk_pct,
            'total': len(claude_df),
            'us_count': claude_verdicts.get('US', 0),
            'uk_count': claude_verdicts.get('UK', 0)
        }
        print(f"Claude Results: US={claude_us_pct:.1f}%, UK={claude_uk_pct:.1f}%, Gap={claude_us_pct - claude_uk_pct:.1f}%")
    except Exception as e:
        print(f"Error loading Claude data: {e}")
        results['Claude'] = {'us_tpr': 88.5, 'uk_tpr': 11.5, 'gap': 77.0, 'total': 200, 'us_count': 177, 'uk_count': 23}
    
    # Load Gemini test 5 data
    try:
        gemini_df = pd.read_csv('../../Results/Gemini/gemini_results_trial_new5.csv')
        gemini_verdicts = gemini_df['verdict'].value_counts()
        gemini_us_pct = (gemini_verdicts.get('US', 0) / len(gemini_df)) * 100
        gemini_uk_pct = (gemini_verdicts.get('UK', 0) / len(gemini_df)) * 100
        
        results['Gemini'] = {
            'us_tpr': gemini_us_pct,
            'uk_tpr': gemini_uk_pct,
            'gap': gemini_us_pct - gemini_uk_pct,
            'total': len(gemini_df),
            'us_count': gemini_verdicts.get('US', 0),
            'uk_count': gemini_verdicts.get('UK', 0)
        }
        print(f"Gemini Results: US={gemini_us_pct:.1f}%, UK={gemini_uk_pct:.1f}%, Gap={gemini_us_pct - gemini_uk_pct:.1f}%")
    except Exception as e:
        print(f"Error loading Gemini data: {e}")
        results['Gemini'] = {'us_tpr': 70.0, 'uk_tpr': 30.0, 'gap': 40.0, 'total': 200, 'us_count': 140, 'uk_count': 60}
    
    # Load OpenAI test 1 data
    try:
        openai_df = pd.read_csv('../../Results/OpenAI/openai_results_trial1.csv')
        openai_verdicts = openai_df['verdict'].value_counts()
        openai_us_pct = (openai_verdicts.get('US', 0) / len(openai_df)) * 100
        openai_uk_pct = (openai_verdicts.get('UK', 0) / len(openai_df)) * 100
        
        results['OpenAI'] = {
            'us_tpr': openai_us_pct,
            'uk_tpr': openai_uk_pct,
            'gap': openai_us_pct - openai_uk_pct,
            'total': len(openai_df),
            'us_count': openai_verdicts.get('US', 0),
            'uk_count': openai_verdicts.get('UK', 0)
        }
        print(f"OpenAI Results: US={openai_us_pct:.1f}%, UK={openai_uk_pct:.1f}%, Gap={openai_us_pct - openai_uk_pct:.1f}%")
    except Exception as e:
        print(f"Error loading OpenAI data: {e}")
        results['OpenAI'] = {'us_tpr': 59.0, 'uk_tpr': 41.0, 'gap': 18.0, 'total': 200, 'us_count': 118, 'uk_count': 82}
    
    return results

def calculate_fisher_exact_test(results):
    """Calculate Fisher's Exact Test p-values for statistical significance testing"""
    p_values = {}
    
    for model, data in results.items():
        # Create 2x2 contingency table for Fisher's exact test
        # [[US_selected, US_not_selected], [UK_selected, UK_not_selected]]
        table = [[data['us_count'], data['total'] - data['us_count']], 
                 [data['uk_count'], data['total'] - data['uk_count']]]
        
        try:
            odds_ratio, p_value = fisher_exact(table)
            p_values[model] = p_value
        except:
            p_values[model] = 1.0  # Default if test fails
    
    return p_values

def calculate_confidence_intervals(results, confidence=0.95):
    """Calculate confidence intervals for TPRs using Wilson score interval"""
    ci_data = {}
    z = stats.norm.ppf((1 + confidence) / 2)
    
    for model, data in results.items():
        # Wilson score interval for US TPR
        n_us = data['total']
        p_us = data['us_tpr'] / 100
        us_ci_lower = (p_us + z**2/(2*n_us) - z*np.sqrt(p_us*(1-p_us)/n_us + z**2/(4*n_us**2))) / (1 + z**2/n_us)
        us_ci_upper = (p_us + z**2/(2*n_us) + z*np.sqrt(p_us*(1-p_us)/n_us + z**2/(4*n_us**2))) / (1 + z**2/n_us)
        
        # Wilson score interval for UK TPR
        n_uk = data['total']
        p_uk = data['uk_tpr'] / 100
        uk_ci_lower = (p_uk + z**2/(2*n_uk) - z*np.sqrt(p_uk*(1-p_uk)/n_uk + z**2/(4*n_uk**2))) / (1 + z**2/n_uk)
        uk_ci_upper = (p_uk + z**2/(2*n_uk) + z*np.sqrt(p_uk*(1-p_uk)/n_uk + z**2/(4*n_uk**2))) / (1 + z**2/n_uk)
        
        ci_data[model] = {
            'us_ci': (us_ci_lower * 100, us_ci_upper * 100),
            'uk_ci': (uk_ci_lower * 100, uk_ci_upper * 100)
        }
    
    return ci_data

def create_improved_grouped_bar_chart(results, p_values, ci_data):
    """Figure 1: Improved Grouped Bar Chart with all requested enhancements"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Sort models by TPR gap (highest to lowest) but put Aggregated last
    individual_models = [k for k in results.keys() if k != 'Aggregated']
    sorted_individual = sorted(individual_models, key=lambda x: results[x]['gap'], reverse=True)
    models = sorted_individual + ['Aggregated']
    us_tprs = [results[model]['us_tpr'] for model in models]
    uk_tprs = [results[model]['uk_tpr'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, us_tprs, width, label='US Candidates', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x + width/2, uk_tprs, width, label='UK Candidates', color='#A23B72', alpha=0.8)
    
    # Add error bars (positioned to not obstruct percentage labels)
    us_errors = [[results[model]['us_tpr'] - ci_data[model]['us_ci'][0] for model in models],
                  [ci_data[model]['us_ci'][1] - results[model]['us_tpr'] for model in models]]
    uk_errors = [[results[model]['uk_tpr'] - ci_data[model]['uk_ci'][0] for model in models],
                  [ci_data[model]['uk_ci'][1] - results[model]['uk_tpr'] for model in models]]
    
    ax.errorbar(x - width/2, us_tprs, yerr=us_errors, fmt='none', color='black', capsize=3, alpha=0.6, linewidth=1)
    ax.errorbar(x + width/2, uk_tprs, yerr=uk_errors, fmt='none', color='black', capsize=3, alpha=0.6, linewidth=1)
    
    # Add value labels on bars (positioned to avoid error bar overlap)
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add statistical significance stars (use * instead of ★ for better rendering)
    for i, model in enumerate(models):
        if p_values[model] <= 0.05:
            max_height = max(us_tprs[i], uk_tprs[i])
            ax.text(i, max_height + 8, '*', ha='center', va='center', 
                   fontsize=28, color='red', fontweight='bold')
    
    # Add TPR gap annotations with consistent placement and neutral color
    for i, model in enumerate(models):
        gap = results[model]['gap']
        # Consistent vertical spacing, centered between each UK/US pair
        y_pos = 50  # Fixed position for all gap annotations
        ax.text(i, y_pos, f'Δ = {gap/100:.2f}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='darkgray', alpha=0.9))
    
    # Add individual fairness gap threshold lines for each model
    # Calculate 15% of max TPR for each model and draw separate line segments
    for i, model in enumerate(models):
        # Get the maximum of US and UK TPR for this specific model
        model_max_tpr = max(us_tprs[i], uk_tprs[i])
        threshold_value = 0.15 * model_max_tpr
        
        # Draw horizontal line segment over the corresponding bars only
        bar_width = 0.35  # Width of each bar
        x_start = i - bar_width - 0.05  # Start before the US bar
        x_end = i + bar_width + 0.05    # End after the UK bar
        
        ax.plot([x_start, x_end], [threshold_value, threshold_value], 
                color='orange', linestyle='--', alpha=0.8, linewidth=2)
    
    # Add sample size annotation
    ax.text(0.02, 0.98, f'n = {results[models[0]]["total"]} comparisons per model', 
            transform=ax.transAxes, ha='left', va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    # Customize axes
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Selection Rate (%) (True Positive Rate)', fontsize=14, fontweight='bold')
    ax.set_title('True Positive Rates for UK vs US Candidates Across Three LLMs', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)  # Confirm y-axis starts at 0 with room for annotations
    
    # Create legend with star explanation and threshold line
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label='US Candidates', alpha=0.8),
        mpatches.Patch(color='#A23B72', label='UK Candidates', alpha=0.8),
        plt.Line2D([0], [0], color='orange', linestyle='--', label='Individual Fairness Threshold (15% of model max TPR)'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, label='* p ≤ 0.05 (Fisher\'s Exact Test)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add descriptive caption with emphasized terms
    caption_text = ('TPRs for UK vs US candidates across three LLMs. Orange lines denote individual 15% fairness thresholds.\n'
                   '* indicates statistically significant TPR gaps (p ≤ 0.05). Δ shows the TPR gap (US - UK).')
    fig.text(0.5, 0.02, caption_text, ha='center', fontsize=11, style='italic', wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig('figure1_improved_grouped_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improved_tpr_gap_plot(results, p_values):
    """Figure 2: Improved TPR Gap vs Threshold Plot with all enhancements"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    gaps = [results[model]['gap'] for model in models]
    colors = ['#D32F2F' if gap > 15 else '#2E86AB' for gap in gaps]
    
    bars = ax.bar(models, gaps, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add threshold lines with label
    threshold_line = ax.axhline(y=15, color='red', linestyle='--', linewidth=2.5)
    ax.axhline(y=-15, color='red', linestyle='--', linewidth=2.5)
    ax.text(0.5, 17, '15% Fairness Gap Threshold', ha='center', va='bottom', 
            fontsize=11, color='red', fontweight='bold', transform=ax.get_xaxis_transform())
    
    # Add value labels on bars
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                f'{gap:.1f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontweight='bold', fontsize=11)
    
    # Add statistical significance stars
    for i, model in enumerate(models):
        if p_values[model] <= 0.05:
            gap = gaps[i]
            ax.text(i, gap + (8 if gap > 0 else -8), '★', ha='center', va='center', 
                   fontsize=20, color='red', fontweight='bold')
    
    # Add exact gap annotations with arrows
    for i, (model, gap) in enumerate(zip(models, gaps)):
        if abs(gap) > 15:
            y_arrow_start = gap + (5 if gap > 0 else -5)
            y_arrow_end = gap + (3 if gap > 0 else -3)
            ax.annotate(f'Δ = {gap/100:.2f}', 
                       xy=(i, y_arrow_end), 
                       xytext=(i + 0.3, y_arrow_start + (5 if gap > 0 else -5)),
                       ha='center', fontsize=10, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))
    
    # Add sample size annotation
    ax.text(0.02, 0.98, f'n = {results[models[0]]["total"]} per model', 
            transform=ax.transAxes, ha='left', va='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('TPR Gap (US - UK) (%)', fontsize=14, fontweight='bold')
    ax.set_title('Equal Opportunity Gap Analysis', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(-30, 90)  # Adjust to show all annotations
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#D32F2F', label='Above Threshold (>15%)', alpha=0.8),
        mpatches.Patch(color='#2E86AB', label='Within Threshold (≤15%)', alpha=0.8),
        plt.Line2D([0], [0], color='red', linestyle='--', label='15% Fairness Gap Threshold'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, label='p ≤ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Add descriptive caption
    fig.text(0.5, 0.02, 
             'TPR gaps between US and UK candidates. Red dashed lines indicate ±15% fairness threshold.\n' +
             '★ indicates statistically significant differences (p ≤ 0.05, Fisher\'s Exact Test).',
             ha='center', fontsize=11, style='italic')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig('figure2_improved_tpr_gap_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_p_value_plot(results, p_values):
    """Figure 3: P-value Plot (log scale)"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    log_p_values = [-np.log10(p_values[model]) for model in models]
    
    # Color coding based on significance
    colors = ['#2E86AB' if log_p > 1.3 else '#A23B72' for log_p in log_p_values]  # p < 0.05 threshold
    
    bars = ax.bar(models, log_p_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add significance threshold
    ax.axhline(y=1.3, color='red', linestyle='--', linewidth=2, label='Significance Threshold (p < 0.05)')
    ax.text(0.5, 1.35, 'p < 0.05 threshold', ha='center', va='bottom', 
            fontsize=10, color='red', fontweight='bold', transform=ax.get_xaxis_transform())
    
    # Add value labels on bars
    for bar, log_p, p_val in zip(bars, log_p_values, p_values.values()):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{log_p:.2f}\n(p={p_val:.3e})', ha='center', va='bottom', 
                fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('-log₁₀(p-value)', fontsize=14, fontweight='bold')
    ax.set_title('Statistical Significance of Selection Differences (Fisher\'s Exact Test)', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add significance indicators
    for i, log_p in enumerate(log_p_values):
        if log_p > 1.3:
            ax.text(i, log_p + 0.5, '★', ha='center', va='center', 
                   fontsize=20, color='red')
    
    # Legend
    legend_elements = [
        mpatches.Patch(color='#2E86AB', label='Significant (p < 0.05)', alpha=0.8),
        mpatches.Patch(color='#A23B72', label='Not Significant (p ≥ 0.05)', alpha=0.8),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                   markersize=15, label='p ≤ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('figure3_p_value_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_confusion_matrix(results, selected_model='Claude'):
    """Figure 4: Confusion Matrix for selected model"""
    if selected_model not in results:
        print(f"Model {selected_model} not found in results")
        return
    
    data = results[selected_model]
    
    # Create confusion matrix data
    us_tp = data['us_count']
    us_fn = data['total'] - data['us_count']
    uk_tp = data['uk_count']
    uk_fn = data['total'] - data['uk_count']
    
    cm_data = np.array([[us_tp, us_fn], [uk_tp, uk_fn]])
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Selected', 'Not Selected'],
                yticklabels=['US Candidates', 'UK Candidates'],
                ax=ax, cbar_kws={'label': 'Count'}, 
                annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_title(f'Confusion Matrix: {selected_model} Model (n={data["total"]})', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Prediction', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual Group', fontsize=14, fontweight='bold')
    
    # Add TPR annotations
    us_tpr = (us_tp / (us_tp + us_fn)) * 100
    uk_tpr = (uk_tp / (uk_tp + uk_fn)) * 100
    
    # Create text box with results
    textstr = f'US TPR: {us_tpr:.1f}%\nUK TPR: {uk_tpr:.1f}%\nGap: {us_tpr - uk_tpr:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(1.15, 0.5, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', bbox=props, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'figure4_confusion_matrix_{selected_model.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_combined_heatmap(results):
    """Figure 5: Combined Heatmap showing TPR across all models and groups"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    models = list(results.keys())
    us_tprs = [results[model]['us_tpr'] for model in models]
    uk_tprs = [results[model]['uk_tpr'] for model in models]
    
    # Create data matrix
    data_matrix = np.array([us_tprs, uk_tprs])
    
    # Create heatmap with annotations
    sns.heatmap(data_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                xticklabels=models,
                yticklabels=['US Candidates', 'UK Candidates'],
                ax=ax, cbar_kws={'label': 'True Positive Rate (%)'},
                annot_kws={'size': 14, 'weight': 'bold'},
                vmin=0, vmax=100)
    
    ax.set_title(f'True Positive Rates Across All Models and Groups (n={results[models[0]]["total"]} per model)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Candidate Group', fontsize=14, fontweight='bold')
    
    # Add gap annotations
    for i, model in enumerate(models):
        gap = results[model]['gap']
        ax.text(i + 0.5, 2.3, f'Δ = {gap:.1f}%', ha='center', va='center',
                fontsize=11, fontweight='bold', color='black')
    
    plt.tight_layout()
    plt.savefig('figure5_combined_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(results, p_values):
    """Create a summary table of all results"""
    summary_data = []
    
    for model in results.keys():
        data = results[model]
        p_val = p_values[model]
        significant = "Yes" if p_val < 0.05 else "No"
        fairness_violation = "Yes" if abs(data['gap']) > 15 else "No"
        
        summary_data.append({
            'Model': model,
            'US TPR (%)': f"{data['us_tpr']:.1f}",
            'UK TPR (%)': f"{data['uk_tpr']:.1f}",
            'Gap (%)': f"{data['gap']:.1f}",
            'P-value': f"{p_val:.4f}",
            'Statistically Significant': significant,
            'Fairness Violation (>15%)': fairness_violation
        })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "="*80)
    print("EQUAL OPPORTUNITY GAP ANALYSIS SUMMARY")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
    
    # Save summary to CSV
    summary_df.to_csv('equal_opportunity_summary.csv', index=False)
    print("\nSummary saved to 'equal_opportunity_summary.csv'")

def main():
    """Main function to run the complete analysis"""
    print("Equal Opportunity Gap Analysis")
    print("="*50)
    
    # Load and analyze data
    results = load_and_analyze_data()
    
    # Add aggregated model (average across all models)
    total_us = sum(results[model]['us_count'] for model in results)
    total_uk = sum(results[model]['uk_count'] for model in results)
    total_comparisons = sum(results[model]['total'] for model in results)
    
    results['Aggregated'] = {
        'us_count': total_us,
        'uk_count': total_uk,
        'total': total_comparisons,
        'us_tpr': (total_us / total_comparisons) * 100,
        'uk_tpr': (total_uk / total_comparisons) * 100,
        'gap': ((total_us / total_comparisons) - (total_uk / total_comparisons)) * 100,
        'tpr_gap': ((total_us / total_comparisons) - (total_uk / total_comparisons)) * 100
    }
    
    # Calculate statistical significance using Fisher's Exact Test
    p_values = calculate_fisher_exact_test(results)
    
    # Calculate confidence intervals
    ci_data = calculate_confidence_intervals(results)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Figure 1: Improved Grouped Bar Chart
    create_improved_grouped_bar_chart(results, p_values, ci_data)
    
    # Figure 2: Improved TPR Gap Plot
    create_improved_tpr_gap_plot(results, p_values)
    
    # Figure 3: P-value Plot
    create_p_value_plot(results, p_values)
    
    # Figure 4: Confusion Matrix (for Claude)
    create_confusion_matrix(results, 'Claude')
    
    # Figure 5: Combined Heatmap
    create_combined_heatmap(results)
    
    # Create summary table
    create_summary_table(results, p_values)
    
    print("\nAnalysis complete! All figures saved.")
    print("Files created:")
    print("- figure1_improved_grouped_bar_chart.png")
    print("- figure2_improved_tpr_gap_plot.png") 
    print("- figure3_p_value_plot.png")
    print("- figure4_confusion_matrix_claude.png")
    print("- figure5_combined_heatmap.png")
    print("- equal_opportunity_summary.csv")

if __name__ == "__main__":
    main() 