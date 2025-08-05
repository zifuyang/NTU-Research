import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

def create_model_evaluation_figure():
    """Create a clean horizontal flow diagram: Resumes → Prompt → LLMs → Verdict"""
    
    # Set up the figure with more height for vertical stacking
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Color scheme
    resume_fill = '#E2E8F0'  # light grey
    resume_edge = '#6B7280'  # medium grey
    prompt_fill = '#FFFFFF'  # white
    prompt_edge = '#1E40AF'  # dark blue
    llm_fill = '#DBEAFE'    # very light blue
    llm_edge = '#1E40AF'    # dark blue
    verdict_fill = '#FEF3C7' # light yellow
    verdict_edge = '#92400E' # dark yellow
    arrow_color = '#374151'  # charcoal
    
    # Box parameters
    box_radius = 0.15
    stroke_width = 1.25
    
    # Step 1: Two Resumes (stacked vertically on left)
    resume1_y = 5.5
    resume2_y = 2.5
    resume_x = 1
    resume_width = 2.5
    resume_height = 1.5
    
    # Resume 1 (US)
    resume1_box = FancyBboxPatch((resume_x, resume1_y), resume_width, resume_height,
                                boxstyle=f"round,pad={box_radius}",
                                facecolor=resume_fill,
                                edgecolor=resume_edge,
                                linewidth=stroke_width)
    ax.add_patch(resume1_box)
    ax.text(resume_x + resume_width/2, resume1_y + resume_height/2, 
            "Resume 1 (US)", fontsize=14, ha='center', va='center', weight='semibold')
    
    # Resume 2 (UK)
    resume2_box = FancyBboxPatch((resume_x, resume2_y), resume_width, resume_height,
                                boxstyle=f"round,pad={box_radius}",
                                facecolor=resume_fill,
                                edgecolor=resume_edge,
                                linewidth=stroke_width)
    ax.add_patch(resume2_box)
    ax.text(resume_x + resume_width/2, resume2_y + resume_height/2,
            "Resume 2 (UK)", fontsize=14, ha='center', va='center', weight='semibold')
    
    # Step 2: Prompt (centered)
    prompt_x = 5.5
    prompt_y = 2.75
    prompt_width = 3
    prompt_height = 2.5
    
    prompt_box = FancyBboxPatch((prompt_x, prompt_y), prompt_width, prompt_height,
                               boxstyle=f"round,pad={box_radius}",
                               facecolor=prompt_fill,
                               edgecolor=prompt_edge,
                               linewidth=stroke_width)
    ax.add_patch(prompt_box)
    ax.text(prompt_x + prompt_width/2, prompt_y + prompt_height/2 + 0.4,
            "Hiring-decision prompt", fontsize=14, ha='center', va='center', weight='semibold')
    ax.text(prompt_x + prompt_width/2, prompt_y + prompt_height/2 - 0.1,
            "You are acting as a professional hiring manager...", fontsize=9, ha='center', va='center', 
            style='italic', color='#4B5563')
    ax.text(prompt_x + prompt_width/2, prompt_y + prompt_height/2 - 0.4,
            "Format: Verdict: [US/UK] | Reasoning: [...]", fontsize=8, ha='center', va='center', 
            style='italic', color='#4B5563')
    
    # Step 3: Three LLMs (stacked vertically)
    llm_x = 10
    llm_width = 2.8
    llm_height = 1.2
    llm_spacing = 0.3
    
    # Calculate starting y position to center the LLMs
    total_llm_height = 3 * llm_height + 2 * llm_spacing
    llm_start_y = 4 - total_llm_height/2
    
    llm_names = ["Claude 3.5 Sonnet", "Gemini 2.0 Flash", "GPT-4o Mini"]
    
    for i, name in enumerate(llm_names):
        llm_y = llm_start_y + i * (llm_height + llm_spacing)
        llm_box = FancyBboxPatch((llm_x, llm_y), llm_width, llm_height,
                                boxstyle=f"round,pad={box_radius}",
                                facecolor=llm_fill,
                                edgecolor=llm_edge,
                                linewidth=stroke_width)
        ax.add_patch(llm_box)
        ax.text(llm_x + llm_width/2, llm_y + llm_height/2,
                name, fontsize=12, ha='center', va='center')
    
    # Step 4: Verdict (far right)
    verdict_x = 13.5
    verdict_y = 3.25
    verdict_width = 2
    verdict_height = 1.5
    
    verdict_box = FancyBboxPatch((verdict_x, verdict_y), verdict_width, verdict_height,
                                boxstyle=f"round,pad={box_radius}",
                                facecolor=verdict_fill,
                                edgecolor=verdict_edge,
                                linewidth=stroke_width)
    ax.add_patch(verdict_box)
    ax.text(verdict_x + verdict_width/2, verdict_y + verdict_height/2 + 0.2,
            "Model verdict", fontsize=14, ha='center', va='center', weight='semibold')
    ax.text(verdict_x + verdict_width/2, verdict_y + verdict_height/2 - 0.3,
            "yes / no", fontsize=10, ha='center', va='center', color='#4B5563')
    
    # Arrows (all perfectly horizontal)
    arrow_props = dict(arrowstyle='->', lw=1.25, color=arrow_color)
    
    # From resumes to prompt (merge into single arrow)
    # Arrow from resume 1
    ax.annotate('', xy=(prompt_x, prompt_y + prompt_height/2 + 0.3), 
                xytext=(resume_x + resume_width, resume1_y + resume_height/2),
                arrowprops=arrow_props)
    
    # Arrow from resume 2
    ax.annotate('', xy=(prompt_x, prompt_y + prompt_height/2 - 0.3), 
                xytext=(resume_x + resume_width, resume2_y + resume_height/2),
                arrowprops=arrow_props)
    
    # From prompt to LLMs (single arrow that splits)
    # Main horizontal arrow
    ax.annotate('', xy=(llm_x - 0.3, 4), 
                xytext=(prompt_x + prompt_width, prompt_y + prompt_height/2),
                arrowprops=arrow_props)
    
    # Vertical line connecting to all LLMs
    ax.plot([llm_x - 0.3, llm_x - 0.3], 
            [llm_start_y + llm_height/2, llm_start_y + total_llm_height - llm_height/2], 
            color=arrow_color, lw=1.25)
    
    # Small arrows to each LLM
    for i in range(3):
        llm_y = llm_start_y + i * (llm_height + llm_spacing) + llm_height/2
        ax.annotate('', xy=(llm_x, llm_y), 
                    xytext=(llm_x - 0.3, llm_y),
                    arrowprops=arrow_props)
    
    # From LLMs to verdict (merge from all LLMs)
    # Vertical line collecting from all LLMs
    ax.plot([llm_x + llm_width + 0.3, llm_x + llm_width + 0.3], 
            [llm_start_y + llm_height/2, llm_start_y + total_llm_height - llm_height/2], 
            color=arrow_color, lw=1.25)
    
    # Small arrows from each LLM
    for i in range(3):
        llm_y = llm_start_y + i * (llm_height + llm_spacing) + llm_height/2
        ax.annotate('', xy=(llm_x + llm_width + 0.3, llm_y), 
                    xytext=(llm_x + llm_width, llm_y),
                    arrowprops=arrow_props)
    
    # Final arrow to verdict
    ax.annotate('', xy=(verdict_x, verdict_y + verdict_height/2), 
                xytext=(llm_x + llm_width + 0.3, 4),
                arrowprops=arrow_props)
    
    # Title
    plt.title('Model Evaluation Process', fontsize=18, fontweight='bold', pad=20)
    
    # Subtitle with experiment details
    ax.text(8, 0.5, '200 matched-pair resumes • accounting profession • MIT vs Imperial College London',
            fontsize=12, ha='center', va='center', color='#6B7280')
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "NTU-Research/figures"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "model_evaluation_methodology.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Model evaluation methodology figure saved to: {output_path}")
    
    # Don't show the plot to avoid blocking
    # plt.show()

def save_model_evaluation_figure():
    create_model_evaluation_figure()

if __name__ == "__main__":
    save_model_evaluation_figure()