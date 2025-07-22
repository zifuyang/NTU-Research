"""
Main Script for Resume Bias Detection Project
Orchestrates the complete workflow from data generation to bias analysis
"""

import os
import sys
import argparse
from datetime import datetime

from sample_data_generator import SampleDataGenerator
from resume_processor import ResumeProcessor
from bias_analyzer import BiasAnalyzer
from config import RESULTS_DIR

def setup_environment():
    """Setup the environment and create necessary directories"""
    print("Setting up environment...")
    
    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check if OpenAI API key is set
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment variable")
        return False
    
    print("✅ Environment setup completed")
    return True

def generate_sample_data(num_resumes: int = 100):
    """Generate sample resume data"""
    print(f"\nGenerating {num_resumes} sample resumes...")
    
    generator = SampleDataGenerator()
    df = generator.generate_dataset(num_resumes)
    generator.save_dataset(df)
    
    print("✅ Sample data generation completed")
    return df

def process_resumes():
    """Process resumes through OpenAI API"""
    print("\nProcessing resumes through OpenAI API...")
    
    processor = ResumeProcessor()
    results_df = processor.run()
    
    if not results_df.empty:
        print("✅ Resume processing completed")
        return results_df
    else:
        print("❌ Resume processing failed")
        return None

def analyze_bias():
    """Analyze results for bias detection"""
    print("\nAnalyzing results for bias detection...")
    
    analyzer = BiasAnalyzer()
    results = analyzer.run_analysis()
    
    print("✅ Bias analysis completed")
    return results

def run_complete_workflow(num_resumes: int = 100, skip_data_generation: bool = False):
    """Run the complete workflow"""
    print("=" * 60)
    print("RESUME BIAS DETECTION PROJECT")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Exiting.")
        return False
    
    try:
        # Step 1: Generate sample data (if not skipped)
        if not skip_data_generation:
            generate_sample_data(num_resumes)
        else:
            print("\nSkipping data generation (using existing Excel data)")
            print("Make sure you have 'resume_dataset.xlsx' in the project directory")
        
        # Step 2: Process resumes
        results_df = process_resumes()
        if results_df is None:
            print("❌ Workflow failed at resume processing step")
            return False
        
        # Step 3: Analyze bias
        analysis_results = analyze_bias()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("WORKFLOW COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Total resumes processed: {len(results_df)}")
        print(f"Results saved in: {RESULTS_DIR}")
        
        # Check for bias
        tests = analysis_results.get('statistical_tests', {})
        bias_detected = any(test.get('significant', False) for test in tests.values())
        
        if bias_detected:
            print("⚠️  STATISTICALLY SIGNIFICANT BIAS DETECTED")
        else:
            print("✅ NO STATISTICALLY SIGNIFICANT BIAS DETECTED")
        
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        return True
        
    except Exception as e:
        print(f"❌ Workflow failed with error: {str(e)}")
        return False

def run_individual_step(step: str, **kwargs):
    """Run an individual step of the workflow"""
    print(f"Running {step} step...")
    
    if step == "generate_data":
        num_resumes = kwargs.get('num_resumes', 100)
        generate_sample_data(num_resumes)
    
    elif step == "process_resumes":
        process_resumes()
    
    elif step == "analyze_bias":
        analyze_bias()
    
    else:
        print(f"Unknown step: {step}")
        return False
    
    return True

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Resume Bias Detection Project")
    parser.add_argument("--step", choices=["generate_data", "process_resumes", "analyze_bias", "full"],
                       default="full", help="Which step to run")
    parser.add_argument("--num-resumes", type=int, default=100,
                       help="Number of resumes to generate (default: 100)")
    parser.add_argument("--skip-data-generation", action="store_true",
                       help="Skip data generation step (use existing data)")
    
    args = parser.parse_args()
    
    if args.step == "full":
        success = run_complete_workflow(args.num_resumes, args.skip_data_generation)
    else:
        success = run_individual_step(args.step, num_resumes=args.num_resumes)
    
    if success:
        print("\n🎉 All operations completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some operations failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 