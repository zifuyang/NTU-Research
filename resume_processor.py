"""
Resume Bias Detection Processor
Main script for processing resumes through OpenAI API and detecting hiring bias
"""

import json
import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from tqdm import tqdm
import os
from datetime import datetime
import re

from config import (
    OPENAI_API_KEY, OPENAI_MODEL, MAX_TOKENS, TEMPERATURE,
    INPUT_EXCEL_FILE, OUTPUT_CSV_FILE, LOG_FILE, RESULTS_DIR,
    BATCH_SIZE, DELAY_BETWEEN_REQUESTS, MAX_RETRIES,
    RESUME_EVALUATION_PROMPT, UK_UNIVERSITIES, US_UNIVERSITIES
)

class ResumeProcessor:
    """Main class for processing resumes and detecting bias"""
    
    def __init__(self):
        """Initialize the processor with OpenAI client and logging"""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.setup_logging()
        self.results = []
        self.processed_count = 0
        self.failed_count = 0
        
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(LOG_FILE),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_resume_data(self) -> pd.DataFrame:
        """Load resume data from Excel file"""
        try:
            self.logger.info(f"Loading resume data from {INPUT_EXCEL_FILE}")
            df = pd.read_excel(INPUT_EXCEL_FILE)
            
            # Validate required columns
            required_columns = ['resume_id', 'resume_content', 'university']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            self.logger.info(f"Loaded {len(df)} resumes successfully")
            return df
            
        except FileNotFoundError:
            self.logger.error(f"Input file {INPUT_EXCEL_FILE} not found")
            raise
        except Exception as e:
            self.logger.error(f"Error loading resume data: {str(e)}")
            raise
    
    def classify_university(self, university: str) -> str:
        """Classify university as UK, US, or Other"""
        university_lower = university.lower()
        
        for uk_uni in UK_UNIVERSITIES:
            if uk_uni.lower() in university_lower:
                return "UK"
        
        for us_uni in US_UNIVERSITIES:
            if us_uni.lower() in university_lower:
                return "US"
        
        return "Other"
    
    def extract_verdict_from_response(self, response_text: str) -> Optional[Dict]:
        """Extract AI verdict and reasoning from OpenAI response"""
        try:
            # Parse the format: "Recommendation: [SCHOOL] | Reasoning: [explanation]"
            pattern = r'Recommendation:\s*(.+?)\s*\|\s*Reasoning:\s*(.+)'
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                school_name = match.group(1).strip()
                reasoning = match.group(2).strip()
                
                return {
                    'ai_verdict': school_name,
                    'reasoning': reasoning
                }
            else:
                # Fallback: try to extract any structured response
                self.logger.warning(f"Could not parse structured response: {response_text}")
                return {
                    'ai_verdict': 'UNPARSEABLE',
                    'reasoning': response_text
                }
        except Exception as e:
            self.logger.warning(f"Failed to parse response: {e}")
            return None
    
    def evaluate_resume(self, resume_content: str, resume_id: str) -> Optional[Dict]:
        """Evaluate a single resume using OpenAI API"""
        prompt = RESUME_EVALUATION_PROMPT.format(resume_content=resume_content)
        
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an unbiased hiring manager evaluating resumes."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE
                )
                
                response_text = response.choices[0].message.content
                evaluation = self.extract_verdict_from_response(response_text)
                
                if evaluation:
                    evaluation['resume_id'] = resume_id
                    evaluation['raw_response'] = response_text
                    return evaluation
                else:
                    self.logger.warning(f"Failed to extract verdict from response for resume {resume_id}")
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for resume {resume_id}: {str(e)}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(DELAY_BETWEEN_REQUESTS * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All attempts failed for resume {resume_id}")
        
        return None
    
    def process_resume_batch(self, batch_df: pd.DataFrame) -> List[Dict]:
        """Process a batch of resumes"""
        batch_results = []
        
        for _, row in batch_df.iterrows():
            resume_id = row['resume_id']
            resume_content = row['resume_content']
            university = row['university']
            
            self.logger.info(f"Processing resume {resume_id} from {university}")
            
            # Classify university
            university_category = self.classify_university(university)
            
            # Evaluate resume
            evaluation = self.evaluate_resume(resume_content, resume_id)
            
            if evaluation:
                evaluation['university'] = university
                evaluation['university_category'] = university_category
                evaluation['processing_timestamp'] = datetime.now().isoformat()
                batch_results.append(evaluation)
                self.processed_count += 1
                self.logger.info(f"Successfully processed resume {resume_id}")
            else:
                self.failed_count += 1
                self.logger.error(f"Failed to process resume {resume_id}")
            
            # Add delay between requests
            time.sleep(DELAY_BETWEEN_REQUESTS)
        
        return batch_results
    
    def process_all_resumes(self) -> pd.DataFrame:
        """Process all resumes in the dataset"""
        # Load data
        df = self.load_resume_data()
        
        # Process in batches
        total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
        
        self.logger.info(f"Starting to process {len(df)} resumes in {total_batches} batches")
        
        for i in range(0, len(df), BATCH_SIZE):
            batch_df = df.iloc[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            batch_results = self.process_resume_batch(batch_df)
            self.results.extend(batch_results)
            
            # Save intermediate results
            if batch_results:
                self.save_intermediate_results(batch_num)
        
        # Convert results to DataFrame
        if self.results:
            results_df = pd.DataFrame(self.results)
            self.logger.info(f"Processing complete. {self.processed_count} successful, {self.failed_count} failed")
            return results_df
        else:
            self.logger.error("No results generated")
            return pd.DataFrame()
    
    def save_intermediate_results(self, batch_num: int):
        """Save intermediate results after each batch"""
        if self.results:
            intermediate_df = pd.DataFrame(self.results)
            intermediate_file = os.path.join(RESULTS_DIR, f"intermediate_results_batch_{batch_num}.csv")
            intermediate_df.to_csv(intermediate_file, index=False)
            self.logger.info(f"Saved intermediate results to {intermediate_file}")
    
    def save_final_results(self, results_df: pd.DataFrame):
        """Save final results to CSV"""
        output_file = os.path.join(RESULTS_DIR, OUTPUT_CSV_FILE)
        results_df.to_csv(output_file, index=False)
        self.logger.info(f"Final results saved to {output_file}")
        
        # Save processing summary
        summary = {
            'total_resumes': len(results_df),
            'successful_processing': self.processed_count,
            'failed_processing': self.failed_count,
            'uk_universities': len(results_df[results_df['university_category'] == 'UK']),
            'us_universities': len(results_df[results_df['university_category'] == 'US']),
            'other_universities': len(results_df[results_df['university_category'] == 'Other']),
            'processing_timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(RESULTS_DIR, 'processing_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Processing summary saved to {summary_file}")
    
    def run(self):
        """Main execution method"""
        try:
            self.logger.info("Starting Resume Bias Detection Processing")
            
            # Process all resumes
            results_df = self.process_all_resumes()
            
            if not results_df.empty:
                # Save final results
                self.save_final_results(results_df)
                
                # Print summary
                self.print_summary(results_df)
            else:
                self.logger.error("No results to save")
                
        except Exception as e:
            self.logger.error(f"Processing failed: {str(e)}")
            raise
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print processing summary"""
        print("\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total resumes processed: {len(results_df)}")
        print(f"Successful evaluations: {self.processed_count}")
        print(f"Failed evaluations: {self.failed_count}")
        print(f"Success rate: {self.processed_count/(self.processed_count + self.failed_count)*100:.1f}%")
        
        print(f"\nUniversity Distribution:")
        uni_counts = results_df['university_category'].value_counts()
        for category, count in uni_counts.items():
            print(f"  {category}: {count}")
        
        print(f"\nResults saved to: {os.path.join(RESULTS_DIR, OUTPUT_CSV_FILE)}")
        print("="*50)


def main():
    """Main function to run the resume processor"""
    processor = ResumeProcessor()
    processor.run()


if __name__ == "__main__":
    main() 