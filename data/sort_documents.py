import os
import shutil
import re
import argparse

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Sort science exam papers into questions, answers, and default folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sort_documents.py bio/comb_bio
  python sort_documents.py bio/pure_bio 
  python sort_documents.py chem/pure_chem
  python sort_documents.py phy/pure_phy
        """
    )
    
    parser.add_argument('subject_filepath', 
                       help='Subject file path (e.g., "bio/comb_bio", "bio/pure_bio", "chem/pure_chem")')

    args = parser.parse_args()
    
    # --- Configuration from arguments ---
    subject_filepath = args.subject_filepath
    
    # Updated BASE_DIR to match the download structure
    BASE_DIR = f"papers/{subject_filepath}/all_papers"
    
    # Target directories where files will be sorted
    TARGET_BASE_DIR = f"papers/{subject_filepath}"
    DEFAULT_DIR = os.path.join(TARGET_BASE_DIR, "default")
    QUESTIONS_DIR = os.path.join(TARGET_BASE_DIR, "questions")
    ANSWERS_DIR = os.path.join(TARGET_BASE_DIR, "answers")

    # Enhanced Keywords for classification
    ANSWER_KEYWORDS = [
        # Standard answer keywords
        "ms", "marking scheme", "answer", "answers", "ans", 
        "solution", "solutions", "soln", "sol", "sols",
        # Additional patterns
        "mark scheme", "answer key", "answer sheet", "worked solutions",
        "model answers", "suggested answers", "specimen answers",
        "marking guide", "marking criteria", "model solution",
        # Language variations
        "jawapan", "skema", "scheme"
    ]
    
    QUESTION_KEYWORDS = [
        # Standard question keywords  
        "qp", "question paper", "questions", "qns", "question",
        # Additional patterns
        "paper", "exam", "test", "quiz", "assessment",
        "exam paper", "examination paper", "test paper",
        # Specific indicators (but be careful not to catch answer papers)
        "specimen paper", "sample paper", "practice paper"
    ]
    
    # Keywords that strongly indicate answers (highest priority)
    STRONG_ANSWER_INDICATORS = [
        "marking scheme", "mark scheme", "ms", "answer key", 
        "worked solutions", "model answers", "solutions", "marking guide"
    ]
    
    # Keywords that strongly indicate questions (highest priority)  
    STRONG_QUESTION_INDICATORS = [
        "question paper", "qp", "specimen paper", "sample paper", "exam paper"
    ]
    
    # Subject-specific patterns for science papers
    SCIENCE_PATTERNS = {
        'physics': ['phy', 'physics', 'physical'],
        'chemistry': ['chem', 'chemistry', 'chemical'],
        'biology': ['bio', 'biology', 'biological'],
        'science': ['sci', 'science', 'combined science']
    }
    
    # Paper type patterns
    PAPER_TYPE_PATTERNS = [
        r'p[1-4]', r'paper\s*[1-4]', r'section\s*[a-d]'
    ]
    
    # Ensure subdirectories exist
    os.makedirs(DEFAULT_DIR, exist_ok=True)
    os.makedirs(QUESTIONS_DIR, exist_ok=True)
    os.makedirs(ANSWERS_DIR, exist_ok=True)
    
    # Display configuration
    print("--- 📁 Starting Enhanced File Sorting ---")
    print(f"Base Directory: {BASE_DIR}")
    print("-" * 50)
    
    # Run the sorting
    sort_files(BASE_DIR, DEFAULT_DIR, QUESTIONS_DIR, ANSWERS_DIR, 
              ANSWER_KEYWORDS, QUESTION_KEYWORDS, STRONG_ANSWER_INDICATORS, 
              STRONG_QUESTION_INDICATORS, SCIENCE_PATTERNS, PAPER_TYPE_PATTERNS)

# Enhanced Keywords for classification (moved to global for reference)
ANSWER_KEYWORDS = [
    # Standard answer keywords
    "ms", "marking scheme", "answer", "answers", "ans", 
    "solution", "solutions", "soln", "sol", "sols",
    # Additional patterns
    "mark scheme", "answer key", "answer sheet", "worked solutions",
    "model answers", "suggested answers", "specimen answers",
    "marking guide", "marking criteria", "model solution",
    # Language variations
    "jawapan", "skema", "scheme"
]

QUESTION_KEYWORDS = [
    # Standard question keywords  
    "qp", "question paper", "questions", "qns", "question",
    # Additional patterns
    "paper", "exam", "test", "quiz", "assessment",
    "exam paper", "examination paper", "test paper",
    # Specific indicators (but be careful not to catch answer papers)
    "specimen paper", "sample paper", "practice paper"
]

# Keywords that strongly indicate answers (highest priority)
STRONG_ANSWER_INDICATORS = [
    "marking scheme", "mark scheme", "ms", "answer key", 
    "worked solutions", "model answers", "solutions", "marking guide"
]

# Keywords that strongly indicate questions (highest priority)  
STRONG_QUESTION_INDICATORS = [
    "question paper", "qp", "specimen paper", "sample paper", "exam paper"
]

# Subject-specific patterns for science papers
SCIENCE_PATTERNS = {
    'physics': ['phy', 'physics', 'physical'],
    'chemistry': ['chem', 'chemistry', 'chemical'],
    'biology': ['bio', 'biology', 'biological'],
    'science': ['sci', 'science', 'combined science']
}

# Paper type patterns
PAPER_TYPE_PATTERNS = [
    r'p[1-4]', r'paper\s*[1-4]', r'section\s*[a-d]'
]

def classify_file(filename, answer_keywords, question_keywords, strong_answer_indicators, 
                  strong_question_indicators, science_patterns, paper_type_patterns):
    """
    Enhanced classification with better pattern recognition and confidence scoring.
    Returns tuple: (classification, confidence_score, reasoning)
    """
    filename_lower = filename.lower()
    
    # Initialize scoring
    answer_score = 0
    question_score = 0
    reasoning = []
    
    # Strong indicators have highest weight (10 points)
    for indicator in strong_answer_indicators:
        if indicator in filename_lower:
            answer_score += 10
            reasoning.append(f"Strong answer indicator: '{indicator}'")
    
    for indicator in strong_question_indicators:
        if indicator in filename_lower:
            question_score += 10
            reasoning.append(f"Strong question indicator: '{indicator}'")
    
    # Regular keywords have medium weight (5 points)
    for keyword in answer_keywords:
        if keyword in filename_lower:
            answer_score += 5
            reasoning.append(f"Answer keyword: '{keyword}'")
    
    for keyword in question_keywords:
        if keyword in filename_lower:
            question_score += 5
            reasoning.append(f"Question keyword: '{keyword}'")
    
    # Subject-specific patterns (bonus 2 points)
    for subject, patterns in science_patterns.items():
        for pattern in patterns:
            if pattern in filename_lower:
                answer_score += 1  # slight bonus for both
                question_score += 1
                reasoning.append(f"Subject pattern: '{pattern}'")
                break
    
    # Paper type patterns (bonus 1 point for structure)
    for pattern in paper_type_patterns:
        if re.search(pattern, filename_lower):
            question_score += 2  # questions more likely to have paper numbers
            reasoning.append(f"Paper type pattern: '{pattern}'")
    
    # Special rules for disambiguation
    # "ms" appearing alone is very likely marking scheme
    if re.search(r'\bms\b', filename_lower):
        answer_score += 15
        reasoning.append("Standalone 'ms' detected (high confidence answer)")
    
    # "qp" appearing alone is very likely question paper
    if re.search(r'\bqp\b', filename_lower):
        question_score += 15
        reasoning.append("Standalone 'qp' detected (high confidence question)")
    
    # Handle conflicting patterns
    if "answer" in filename_lower and "question" in filename_lower:
        # Check which comes first or is more prominent
        answer_pos = filename_lower.find("answer")
        question_pos = filename_lower.find("question")
        if answer_pos < question_pos:
            answer_score += 3
            reasoning.append("'answer' appears before 'question'")
        else:
            question_score += 3
            reasoning.append("'question' appears before 'answer'")
    
    # Calculate final classification
    total_score = answer_score + question_score
    
    if answer_score > question_score:
        classification = "answer"
        confidence = min(answer_score / max(total_score, 1) * 100, 100)
    elif question_score > answer_score:
        classification = "question"  
        confidence = min(question_score / max(total_score, 1) * 100, 100)
    else:
        classification = "unknown"
        confidence = 0
    
    # Adjust confidence based on score magnitude
    if total_score >= 20:
        confidence = min(confidence * 1.2, 100)  # Boost high-scoring files
    elif total_score <= 5:
        confidence = max(confidence * 0.7, 0)    # Reduce low-scoring files
    
    return classification, round(confidence, 1), reasoning

def sort_files(base_dir, default_dir, questions_dir, answers_dir, 
               answer_keywords, question_keywords, strong_answer_indicators, 
               strong_question_indicators, science_patterns, paper_type_patterns):
    """
    Enhanced sorting with confidence scoring and detailed reporting.
    """
    files_moved = {"answers": 0, "questions": 0, "default": 0}
    classification_details = []
    
    # Get all files in the base directory (not in subdirectories)
    try:
        all_items = os.listdir(base_dir)
    except FileNotFoundError:
        print(f"❌ Error: Directory '{base_dir}' not found.")
        return
    
    # Filter only files (exclude directories) and PDF files
    files = [f for f in all_items if os.path.isfile(os.path.join(base_dir, f)) 
             and f.lower().endswith('.pdf')]
    
    if not files:
        print("ℹ️  No PDF files found to sort in the base directory.")
        return
    
    print(f"Found {len(files)} PDF files to sort.\n")
    
    for filename in files:
        source_path = os.path.join(base_dir, filename)
        classification, confidence, reasoning = classify_file(filename, answer_keywords, question_keywords, 
                                                             strong_answer_indicators, strong_question_indicators, 
                                                             science_patterns, paper_type_patterns)
        
        # Determine target directory based on classification
        if classification == "answer":
            target_dir = answers_dir
            category = "answers"
        elif classification == "question":
            target_dir = questions_dir
            category = "questions"
        else:
            target_dir = default_dir
            category = "default"
        
        target_path = os.path.join(target_dir, filename)
        
        # Store classification details
        classification_details.append({
            'filename': filename,
            'category': category,
            'confidence': confidence,
            'reasoning': reasoning
        })
        
        # Move the file
        try:
            # Check if target file already exists
            if os.path.exists(target_path):
                print(f"⚠️ File already exists in target: {filename}")
                base, ext = os.path.splitext(filename)
                counter = 1
                while os.path.exists(target_path):
                    new_filename = f"{base}_{counter}{ext}"
                    target_path = os.path.join(target_dir, new_filename)
                    counter += 1
                print(f"   Renamed to: {os.path.basename(target_path)}")
            
            shutil.move(source_path, target_path)
            # Update confidence indicators for percentage scale
            confidence_indicator = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
            print(f"✅ {confidence_indicator} Moved '{filename}' → {category}/ (confidence: {confidence:.1f}%)")
            files_moved[category] += 1
        except Exception as e:
            print(f"❌ Error moving '{filename}': {e}")
    
    # Enhanced Summary with detailed confidence analysis
    total_files = len(classification_details)
    high_conf_count = len([f for f in classification_details if f['confidence'] >= 70])
    med_conf_count = len([f for f in classification_details if 40 <= f['confidence'] < 70])
    low_conf_count = len([f for f in classification_details if f['confidence'] < 40])
    
    print("\n" + "=" * 80)
    print("📊 ENHANCED SORTING SUMMARY:")
    print("=" * 80)
    print(f"📂 FILE DISTRIBUTION:")
    print(f"  ├─ ✅ Answers:   {files_moved['answers']:3d} files ({files_moved['answers']/total_files*100:.1f}%)")
    print(f"  ├─ ❓ Questions: {files_moved['questions']:3d} files ({files_moved['questions']/total_files*100:.1f}%)")
    print(f"  └─ ❔ Default:   {files_moved['default']:3d} files ({files_moved['default']/total_files*100:.1f}%)")
    print(f"     Total: {total_files} files processed")
    
    print(f"\n🎯 CONFIDENCE ANALYSIS:")
    print(f"  ├─ 🟢 High (≥70%):   {high_conf_count:3d} files ({high_conf_count/total_files*100:.1f}%)")
    print(f"  ├─ 🟡 Medium (40-69%): {med_conf_count:3d} files ({med_conf_count/total_files*100:.1f}%)")
    print(f"  └─ 🔴 Low (<40%):     {low_conf_count:3d} files ({low_conf_count/total_files*100:.1f}%)")
    
    # Classification success rate
    classified_files = files_moved['answers'] + files_moved['questions']
    success_rate = classified_files / total_files * 100
    print(f"\n📈 PERFORMANCE METRICS:")
    print(f"  ├─ Classification Rate: {success_rate:.1f}% ({classified_files}/{total_files})")
    if success_rate >= 90:
        print(f"  └─ 🏆 Excellent performance!")
    elif success_rate >= 75:
        print(f"  └─ ✅ Good performance")
    else:
        print(f"  └─ ⚠️  Performance could be improved")
    
    # Show low confidence classifications for review
    low_confidence = [f for f in classification_details if f['confidence'] < 40]
    if low_confidence:
        print(f"\n⚠️  LOW CONFIDENCE CLASSIFICATIONS ({len(low_confidence)} files):")
        print("   (Manual review recommended)")
        for item in low_confidence[:8]:  # Show first 8
            print(f"   🔴 {item['filename'][:50]:<50} → {item['category']:<8} ({item['confidence']:.1f}%)")
        if len(low_confidence) > 8:
            print(f"   ... and {len(low_confidence) - 8} more files")
    
    # Show sample high confidence classifications
    high_confidence = [f for f in classification_details if f['confidence'] >= 70]
    if high_confidence:
        print(f"\n✅ HIGH CONFIDENCE SAMPLE ({len(high_confidence)} total files):")
        for item in high_confidence[:5]:  # Show first 5
            print(f"   🟢 {item['filename'][:50]:<50} → {item['category']:<8} ({item['confidence']:.1f}%)")
        if len(high_confidence) > 5:
            print(f"   ... and {len(high_confidence) - 5} more high-confidence files")
    
    print("=" * 80)
    print("🎉 Enhanced sorting complete!")
    print("🟢 = High confidence | 🟡 = Medium confidence | 🔴 = Low confidence")
    print("=" * 80)

if __name__ == "__main__":
    main()