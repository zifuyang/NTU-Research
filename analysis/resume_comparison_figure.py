import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_resume_comparison_figure():
    """
    Create a figure showing two identical resumes with different universities
    and their QS rankings for fair comparison.
    """
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    fig.suptitle('Resume Comparison: Same Content, Different Universities', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Resume content (same for both) - using actual content from Corpora.csv
    resume_content = """EDUCATION
Bachelor of Science in Computer Science
GPA: 3.8/4.0
Graduation: 2023

EXPERIENCE
Software Engineer at Google (2023-Present)
• Developed scalable web applications using React and Node.js
• Led team of 5 developers on major project
• Implemented CI/CD pipelines reducing deployment time by 40%

Software Engineering Intern at Microsoft (2022)
• Built cloud infrastructure using Azure
• Managed database systems and optimization
• Developed machine learning models for data analysis

SKILLS
Programming: Python, JavaScript, React, Java, C++
Cloud & DevOps: AWS, Azure, Docker, Git, CI/CD
Databases: SQL, MongoDB, Redis
Other: Machine Learning, Data Analysis, Agile"""
    
    # University information - using actual universities from Corpora.csv
    us_university = "MIT"
    uk_university = "Imperial College London"
    
    # QS Rankings removed as requested
    
    # Create US Resume (left side)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # US Resume box
    us_box = FancyBboxPatch((0.5, 0.5), 9, 9, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#f0f8ff', 
                           edgecolor='#0066cc', 
                           linewidth=2)
    ax1.add_patch(us_box)
    
    # US University header
    ax1.text(5, 8.8, f"EDUCATION\n{us_university}", 
             fontsize=11, fontweight='bold', ha='center', va='top',
             color='#0066cc')
    
    # US Resume content - shorter version to fit better
    short_resume_content = """EDUCATION
Bachelor of Science in Computer Science
GPA: 3.8/4.0
Graduation: 2023

EXPERIENCE
Software Engineer at Google (2023-Present)
• Developed scalable web applications using React and Node.js
• Led team of 5 developers on major project
• Implemented CI/CD pipelines reducing deployment time by 40%

SKILLS
Programming: Python, JavaScript, React, Java, C++
Cloud & DevOps: AWS, Azure, Docker, Git, CI/CD
Databases: SQL, MongoDB, Redis"""
    
    ax1.text(1, 7.2, short_resume_content, fontsize=8, ha='left', va='top',
             fontfamily='monospace')
    
    # US Label
    ax1.text(5, 9.5, "Resume 1 (US)", fontsize=12, fontweight='bold', 
             ha='center', va='top', color='#0066cc')
    
    # Create UK Resume (right side)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # UK Resume box
    uk_box = FancyBboxPatch((0.5, 0.5), 9, 9, 
                           boxstyle="round,pad=0.1", 
                           facecolor='#fff8f0', 
                           edgecolor='#cc6600', 
                           linewidth=2)
    ax2.add_patch(uk_box)
    
    # UK University header
    ax2.text(5, 8.8, f"EDUCATION\n{uk_university}", 
             fontsize=11, fontweight='bold', ha='center', va='top',
             color='#cc6600')
    
    # UK Resume content - shorter version to fit better
    ax2.text(1, 7.2, short_resume_content, fontsize=8, ha='left', va='top',
             fontfamily='monospace')
    
    # UK Label
    ax2.text(5, 9.5, "Resume 2 (UK)", fontsize=12, fontweight='bold', 
             ha='center', va='top', color='#cc6600')
    
    # Add comparison note
    fig.text(0.5, 0.02, 
             "Note: Both resumes contain identical content except for the university name.",
             fontsize=10, ha='center', va='bottom', 
             bbox=dict(boxstyle="round,pad=0.5", facecolor='#f9f9f9', edgecolor='#cccccc'))
    
    # Add equal sign between resumes
    fig.text(0.5, 0.6, "=", fontsize=36, fontweight='bold', 
             ha='center', va='center', color='#666666')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, bottom=0.08)
    
    return fig

def save_resume_comparison_figure():
    """
    Create and save the resume comparison figure.
    """
    fig = create_resume_comparison_figure()
    
    # Save the figure
    output_path = "NTU-Research/figures/resume_comparison_figure.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Resume comparison figure saved to: {output_path}")
    
    plt.show()
    return output_path

if __name__ == "__main__":
    save_resume_comparison_figure() 