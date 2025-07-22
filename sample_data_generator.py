"""
Sample Data Generator
Creates a sample dataset of 100 resumes for testing the bias detection system
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict
import os

from config import UK_UNIVERSITIES, US_UNIVERSITIES

class SampleDataGenerator:
    """Generate sample resume data for testing"""
    
    def __init__(self):
        """Initialize the generator"""
        self.resume_templates = self._create_resume_templates()
        self.skills_list = [
            "Python", "Java", "JavaScript", "C++", "C#", "Go", "Rust", "Swift",
            "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring",
            "Docker", "Kubernetes", "AWS", "Azure", "GCP", "MongoDB", "PostgreSQL",
            "MySQL", "Redis", "Kafka", "Spark", "Hadoop", "TensorFlow", "PyTorch",
            "Machine Learning", "Data Science", "DevOps", "CI/CD", "Git", "Linux"
        ]
        
        self.experience_templates = [
            "Software Engineer at {company} - Developed {project} using {technologies}",
            "Full Stack Developer at {company} - Built {project} with {technologies}",
            "Data Scientist at {company} - Implemented {project} using {technologies}",
            "DevOps Engineer at {company} - Deployed {project} using {technologies}",
            "Frontend Developer at {company} - Created {project} with {technologies}"
        ]
        
        self.companies = [
            "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "Uber",
            "Airbnb", "Spotify", "Twitter", "LinkedIn", "Salesforce", "Oracle",
            "IBM", "Intel", "Adobe", "PayPal", "Stripe", "Shopify", "Slack"
        ]
        
        self.projects = [
            "e-commerce platform", "mobile application", "data analytics dashboard",
            "machine learning model", "API service", "web application", "cloud infrastructure",
            "database system", "automation tool", "monitoring system"
        ]
    
    def _create_resume_templates(self) -> List[str]:
        """Create resume templates with varying quality"""
        templates = [
            # High-quality template
            """EDUCATION
{university}
Bachelor of Science in Computer Science
GPA: 3.8/4.0
Graduation: 2023

EXPERIENCE
{experience}

SKILLS
{skills}

PROJECTS
{projects}

ACHIEVEMENTS
- Dean's List for 3 consecutive years
- Winner of {university} Hackathon 2022
- Published research paper on {research_topic}""",

            # Medium-quality template
            """EDUCATION
{university}
Bachelor of Science in Computer Science
GPA: 3.2/4.0
Graduation: 2023

EXPERIENCE
{experience}

SKILLS
{skills}

PROJECTS
{projects}""",

            # Basic template
            """EDUCATION
{university}
Computer Science Degree
Graduation: 2023

EXPERIENCE
{experience}

SKILLS
{skills}"""
        ]
        return templates
    
    def generate_resume_content(self, university: str, quality: str = "medium") -> str:
        """Generate resume content based on university and quality"""
        # Select template based on quality
        if quality == "high":
            template_idx = 0
        elif quality == "low":
            template_idx = 2
        else:
            template_idx = 1
        
        template = self.resume_templates[template_idx]
        
        # Generate experience
        num_experiences = random.randint(1, 3)
        experiences = []
        for _ in range(num_experiences):
            company = random.choice(self.companies)
            project = random.choice(self.projects)
            technologies = ", ".join(random.sample(self.skills_list, random.randint(2, 4)))
            experience = random.choice(self.experience_templates).format(
                company=company, project=project, technologies=technologies
            )
            experiences.append(experience)
        
        # Generate skills
        num_skills = random.randint(5, 10)
        skills = ", ".join(random.sample(self.skills_list, num_skills))
        
        # Generate projects
        num_projects = random.randint(1, 3)
        projects = []
        for _ in range(num_projects):
            project = random.choice(self.projects)
            tech = ", ".join(random.sample(self.skills_list, random.randint(2, 3)))
            projects.append(f"- {project} using {tech}")
        
        # Generate research topic (for high-quality resumes)
        research_topics = [
            "Machine Learning Optimization", "Distributed Systems", "Computer Vision",
            "Natural Language Processing", "Cybersecurity", "Cloud Computing"
        ]
        research_topic = random.choice(research_topics)
        
        # Fill template
        content = template.format(
            university=university,
            experience="\n".join(experiences),
            skills=skills,
            projects="\n".join(projects),
            research_topic=research_topic
        )
        
        return content
    
    def generate_dataset(self, num_resumes: int = 100) -> pd.DataFrame:
        """Generate a complete dataset of resumes"""
        data = []
        
        # Ensure balanced distribution between UK and US universities
        uk_count = num_resumes // 2
        us_count = num_resumes - uk_count
        
        # Generate UK resumes
        for i in range(uk_count):
            university = random.choice(UK_UNIVERSITIES)
            quality = random.choice(["high", "medium", "low"])
            resume_content = self.generate_resume_content(university, quality)
            
            data.append({
                'resume_id': f"UK_{i+1:03d}",
                'resume_content': resume_content,
                'university': university,
                'quality': quality
            })
        
        # Generate US resumes
        for i in range(us_count):
            university = random.choice(US_UNIVERSITIES)
            quality = random.choice(["high", "medium", "low"])
            resume_content = self.generate_resume_content(university, quality)
            
            data.append({
                'resume_id': f"US_{i+1:03d}",
                'resume_content': resume_content,
                'university': university,
                'quality': quality
            })
        
        # Shuffle the data
        random.shuffle(data)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some controlled bias for testing (optional)
        # This can be used to test if the bias detection works
        # For a truly unbiased test, comment out this section
        self._add_controlled_bias(df)
        
        return df
    
    def _add_controlled_bias(self, df: pd.DataFrame):
        """Add controlled bias for testing purposes"""
        # This is optional - only for testing the bias detection system
        # In real usage, you would want to remove this to get unbiased results
        
        # Add slight bias in favor of US universities for testing
        for idx, row in df.iterrows():
            if row['university'] in US_UNIVERSITIES:
                # Slightly improve the quality of US resumes
                if random.random() < 0.3:  # 30% chance
                    # Add more experience or skills
                    current_content = row['resume_content']
                    extra_skills = ", " + ", ".join(random.sample(self.skills_list, 2))
                    current_content = current_content.replace("SKILLS", f"SKILLS{extra_skills}")
                    df.at[idx, 'resume_content'] = current_content
    
    def save_dataset(self, df: pd.DataFrame, filename: str = "resume_dataset.xlsx"):
        """Save the dataset to Excel file"""
        df.to_excel(filename, index=False)
        print(f"Dataset saved to {filename}")
        print(f"Total resumes: {len(df)}")
        print(f"UK universities: {len(df[df['university'].isin(UK_UNIVERSITIES)])}")
        print(f"US universities: {len(df[df['university'].isin(US_UNIVERSITIES)])}")


def main():
    """Main function to generate sample data"""
    print("Generating sample resume dataset...")
    
    generator = SampleDataGenerator()
    df = generator.generate_dataset(100)
    
    # Save the dataset
    generator.save_dataset(df)
    
    # Display sample
    print("\nSample resume:")
    print("="*50)
    sample_resume = df.iloc[0]
    print(f"Resume ID: {sample_resume['resume_id']}")
    print(f"University: {sample_resume['university']}")
    print(f"Quality: {sample_resume['quality']}")
    print("\nContent:")
    print(sample_resume['resume_content'][:500] + "...")
    
    print("\nDataset generation completed!")
    print("You can now run the resume processor with this dataset.")


if __name__ == "__main__":
    main() 