#!/usr/bin/env python3
"""
Syllabus Keyword Extraction with Random Sampling
================================================

This script randomly samples 200 rows from a CSV dataset and adds keywords
based on syllabus analysis using Gemini AI. This is useful for testing
and creating smaller keyword-enhanced datasets.

Usage:
    python keyword_extraction.py input.csv syllabus.pdf
    python keyword_extraction.py bio_dataset.csv biology.pdf
    python keyword_extraction.py chem_dataset.csv chemistry.pdf

The script will:
1. Load the input CSV dataset
2. Randomly sample 200 rows (or all rows if less than 200)
3. Load the syllabus PDF document
4. Extract keywords for each sampled row using Gemini
5. Save the output as {input_name}_sample_200_with_keywords.csv
"""

import os
import sys
import argparse
import pandas as pd
import json
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from tqdm import tqdm
import time
import random
import numpy as np

# Global constants
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
MODEL_NAME = "gemini-2.5-flash"
SAMPLE_SIZE = 200  # Number of rows to sample

def setup_vertex_ai():
    """Initialize Vertex AI with project settings."""
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(f"✅ Vertex AI initialized: Project={PROJECT_ID}, Location={LOCATION}")
        return True
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {e}")
        return False

def load_syllabus_pdf(pdf_path):
    """
    Load and prepare the syllabus PDF for Gemini processing.
    
    Args:
        pdf_path: Path to the syllabus PDF file
        
    Returns:
        Part object containing the PDF data
    """
    try:
        if not os.path.exists(pdf_path):
            print(f"❌ Error: Syllabus PDF not found at {pdf_path}")
            return None
        
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        pdf_part = Part.from_data(data=pdf_data, mime_type="application/pdf")
        print(f"✅ Loaded syllabus PDF: {pdf_path} ({len(pdf_data)} bytes)")
        return pdf_part
    except Exception as e:
        print(f"❌ Error loading syllabus PDF: {e}")
        return None

def create_keyword_extraction_prompt(context, question, answer):
    """
    Create a prompt for extracting keywords based on syllabus learning objectives.
    
    Args:
        context: Context text for the question
        question: Question text
        answer: Answer text
        
    Returns:
        String prompt for Gemini
    """
    prompt = f"""You are an exam marking assistant. Your task is to identify key scientific terms and concepts in the ANSWER that are essential for scoring marks, based on the syllabus requirements.

SYLLABUS DOCUMENT: The syllabus PDF is provided as context. Use it to understand what terminology and concepts are emphasized in the curriculum.

QUESTION-ANSWER PAIR TO ANALYZE:

Context: {context if context else "No context provided"}

Question: {question}

Answer: {answer}

TASK:
Analyze the ANSWER TEXT and extract 3-8 key scientific terms/concepts that:
1. Are present in or directly relevant to the answer
2. Match syllabus terminology and learning objectives
3. Would likely earn marks if included in a student's response
4. Would likely lose marks if omitted from a student's response
5. Represent core scientific concepts, not filler words

Keywords should be:
- Specific scientific terms, processes, or concepts from the answer
- Terms that appear in the syllabus as key learning points
- Essential for demonstrating subject knowledge
- Precise terminology (e.g., "osmosis" not "water movement")
- Content words that carry meaning (not "because", "therefore", "the")

IMPORTANT RULES:
- Focus on mark-scoring terminology in the ANSWER
- Match terms to syllabus requirements and learning objectives
- Include biological/chemical/physical processes, structures, equations, laws
- Do NOT include generic words like "correct", "answer", "because", "therefore"
- Do NOT include question numbers or reference markers
- If the answer contains no clear syllabus-aligned key terms, return empty array
- Prioritize scientific accuracy and syllabus alignment

BIOLOGY EXAMPLES:
Answer: "Osmosis is the movement of water molecules from high water potential to low water potential through a partially permeable membrane."
Keywords: ["osmosis", "water potential", "partially permeable membrane", "diffusion"]

Answer: "Mitochondria are the site of aerobic respiration, producing ATP through the breakdown of glucose."
Keywords: ["mitochondria", "aerobic respiration", "ATP", "glucose"]

Answer: "The enzyme amylase breaks down starch into maltose."
Keywords: ["amylase", "enzyme", "starch", "maltose", "digestion"]

PHYSICS EXAMPLES:
Answer: "The force acting on a current-carrying conductor in a magnetic field is given by F = BIL, where B is magnetic flux density, I is current, and L is length."
Keywords: ["magnetic force", "current-carrying conductor", "magnetic flux density", "Fleming's left-hand rule", "electromagnetic force"]

Answer: "When light passes from air into glass, it bends towards the normal because glass has a higher refractive index than air."
Keywords: ["refraction", "refractive index", "normal", "Snell's law", "optical density"]

Answer: "The kinetic energy of the moving object is converted to gravitational potential energy as it rises."
Keywords: ["kinetic energy", "gravitational potential energy", "energy conversion", "conservation of energy"]

Answer: "The resistance increases with temperature in metals due to increased collision between electrons and atoms."
Keywords: ["resistance", "temperature coefficient", "electron collision", "conductivity", "metallic conduction"]

CHEMISTRY EXAMPLES:
Answer: "When sodium reacts with chlorine gas, electrons are transferred from sodium atoms to chlorine atoms, forming sodium chloride."
Keywords: ["sodium", "chlorine gas", "electron transfer", "ionic bonding", "sodium chloride", "oxidation", "reduction"]

Answer: "The rate of reaction increases with temperature because particles have more kinetic energy and collide more frequently with greater energy."
Keywords: ["rate of reaction", "temperature", "kinetic energy", "collision frequency", "activation energy", "collision theory"]

Answer: "Ethene undergoes addition reactions with bromine water, causing the orange solution to turn colorless."
Keywords: ["ethene", "addition reaction", "bromine water", "alkene", "unsaturated hydrocarbon", "decolorization"]

Answer: "The pH of the solution decreases as hydrogen ion concentration increases, making it more acidic."
Keywords: ["pH", "hydrogen ion concentration", "acidic", "acid-base", "logarithmic scale"]

Answer: "During electrolysis of copper sulfate solution, copper ions are reduced at the cathode and deposited as copper metal."
Keywords: ["electrolysis", "copper sulfate", "reduction", "cathode", "copper deposition", "oxidation state"]

OUTPUT FORMAT:
Return ONLY a valid JSON array of keywords:
["keyword1", "keyword2", "keyword3"]
or
[]

Return your response as a JSON array:"""

    return prompt

def extract_keywords_with_gemini(model, syllabus_part, context, question, answer, max_retries=3):
    """
    Use Gemini to extract keywords based on syllabus content.
    
    Args:
        model: Gemini model instance
        syllabus_part: PDF Part object containing syllabus
        context: Context text
        question: Question text
        answer: Answer text
        max_retries: Number of retry attempts
        
    Returns:
        List of keywords or empty list if none found
    """
    prompt = create_keyword_extraction_prompt(context, question, answer)
    
    for attempt in range(max_retries):
        try:
            # Combine syllabus PDF with the prompt
            response = model.generate_content([syllabus_part, prompt])
            
            if not response or not response.text:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return []
            
            # Parse the JSON response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            response_text = response_text.strip()
            
            # Parse JSON
            keywords = json.loads(response_text)
            
            # Validate it's a list
            if not isinstance(keywords, list):
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                return []
            
            # Clean and validate keywords
            keywords = [str(k).strip() for k in keywords if k and str(k).strip()]
            
            return keywords
            
        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"⚠️ JSON parse error: {e}")
            return []
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"⚠️ Error extracting keywords: {e}")
            return []
    
    return []

def sample_and_process_csv_with_keywords(csv_path, syllabus_pdf_path, sample_size=SAMPLE_SIZE, output_path=None):
    """
    Sample CSV file and add keywords column based on syllabus analysis.
    
    Args:
        csv_path: Path to input CSV file
        syllabus_pdf_path: Path to syllabus PDF file
        sample_size: Number of rows to sample (default: 200)
        output_path: Optional output path (default: {input}_sample_200_with_keywords.csv)
    """
    # Load CSV
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"✅ Loaded CSV: {csv_path}")
        print(f"   Total rows: {len(df)}, Columns: {list(df.columns)}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return False
    
    # Validate required columns
    required_columns = ['context_text', 'question_text', 'answer_text']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Error: Missing required columns: {missing_columns}")
        return False
    
    # Random sampling
    if len(df) <= sample_size:
        print(f"⚠️ Dataset has {len(df)} rows, which is less than requested sample size {sample_size}")
        print(f"   Using all {len(df)} rows")
        sampled_df = df.copy()
    else:
        print(f"🎲 Randomly sampling {sample_size} rows from {len(df)} total rows")
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        sampled_df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"✅ Sampled {len(sampled_df)} rows")
    
    # Load syllabus PDF
    syllabus_part = load_syllabus_pdf(syllabus_pdf_path)
    if syllabus_part is None:
        return False
    
    # Initialize Gemini model
    try:
        model = GenerativeModel(MODEL_NAME)
        print(f"✅ Initialized Gemini model: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Error initializing Gemini model: {e}")
        return False
    
    # Add keywords column if it doesn't exist
    if 'keywords' not in sampled_df.columns:
        sampled_df['keywords'] = None
    
    # Process each sampled row
    print(f"\n🔄 Processing {len(sampled_df)} sampled rows to extract keywords...")
    keywords_added = 0
    keywords_empty = 0
    
    for idx in tqdm(range(len(sampled_df)), desc="Extracting keywords"):
        context = sampled_df.at[idx, 'context_text']
        question = sampled_df.at[idx, 'question_text']
        answer = sampled_df.at[idx, 'answer_text']
        
        # Handle None/NaN values
        context = "" if pd.isna(context) else str(context)
        question = "" if pd.isna(question) else str(question)
        answer = "" if pd.isna(answer) else str(answer)
        
        # Extract keywords
        keywords = extract_keywords_with_gemini(
            model, syllabus_part, context, question, answer
        )
        
        # Store keywords as JSON string (or empty string if no keywords)
        if keywords:
            sampled_df.at[idx, 'keywords'] = json.dumps(keywords)
            keywords_added += 1
        else:
            sampled_df.at[idx, 'keywords'] = ""
            keywords_empty += 1
        
        # Rate limiting for API calls
        time.sleep(0.5)
    
    # Generate output path
    if output_path is None:
        base_name = os.path.splitext(csv_path)[0]
        output_path = f"{base_name}_sample_{sample_size}_with_keywords.csv"
    
    # Save output CSV
    try:
        sampled_df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n✅ Output saved: {output_path}")
    except Exception as e:
        print(f"\n❌ Error saving output: {e}")
        return False
    
    # Print summary
    print(f"\n📊 KEYWORD EXTRACTION SUMMARY:")
    print(f"  ├─ Original dataset: {len(df)} rows")
    print(f"  ├─ Sampled rows: {len(sampled_df)}")
    print(f"  ├─ Rows with keywords: {keywords_added} ({keywords_added/len(sampled_df)*100:.1f}%)")
    print(f"  ├─ Rows without keywords: {keywords_empty} ({keywords_empty/len(sampled_df)*100:.1f}%)")
    print(f"  └─ Output file: {output_path}")
    
    # Show sample keywords
    sample_with_keywords = sampled_df[sampled_df['keywords'] != ""].head(5)
    if not sample_with_keywords.empty:
        print(f"\n📋 SAMPLE KEYWORDS:")
        for idx, row in sample_with_keywords.iterrows():
            keywords_list = json.loads(row['keywords'])
            print(f"  Q: {row['question_text'][:60]}...")
            print(f"  Keywords: {', '.join(keywords_list)}")
            print()
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Extract keywords from randomly sampled dataset using Gemini AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python syllabus_sampler.py bio_dataset.csv biology.pdf
  python syllabus_sampler.py chem_dataset.csv chemistry.pdf --sample 150
  python syllabus_sampler.py input.csv syllabus.pdf --output custom_output.csv

The script will:
1. Load the input CSV dataset
2. Randomly sample the specified number of rows (default: 200)
3. Load the syllabus PDF document
4. Use Gemini to extract keywords for each sampled row
5. Save the output as {input_name}_sample_200_with_keywords.csv

Keywords are stored as JSON arrays in the CSV:
  - Rows with keywords: ["osmosis", "water potential", "membrane"]
  - Rows without keywords: "" (empty string)
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Input CSV file containing question-answer pairs'
    )
    
    parser.add_argument(
        'syllabus_pdf',
        help='Syllabus PDF file for the subject (e.g., biology.pdf, chemistry.pdf, physics.pdf)'
    )
    
    parser.add_argument(
        '--sample', '-s',
        type=int,
        default=SAMPLE_SIZE,
        help=f'Number of rows to sample (default: {SAMPLE_SIZE})'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output CSV file path (default: {input}_sample_200_with_keywords.csv)'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 70)
    print("🎯 SYLLABUS KEYWORD EXTRACTION WITH SAMPLING")
    print("=" * 70)
    print(f"Input CSV: {args.csv_file}")
    print(f"Syllabus PDF: {args.syllabus_pdf}")
    print(f"Sample Size: {args.sample} rows")
    print(f"Model: {MODEL_NAME}")
    print("=" * 70)
    
    # Initialize Vertex AI
    if not setup_vertex_ai():
        print("\n❌ Failed to initialize Vertex AI. Please check your configuration.")
        return 1
    
    # Process CSV with sampling and keywords
    success = sample_and_process_csv_with_keywords(
        args.csv_file,
        args.syllabus_pdf,
        args.sample,
        args.output
    )
    
    if success:
        print("\n🎉 Keyword extraction completed successfully!")
        return 0
    else:
        print("\n❌ Keyword extraction failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())