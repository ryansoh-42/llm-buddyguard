#!/usr/bin/env python3
"""
Enhanced Context Filter for Question-Context-Answer Triplets (v2)
================================================================

This script filters out low-quality question-answer-context triplets from JSON datasets
that lack sufficient context. Designed for v5 data structure with context_text field.

Usage:
    python additional_filter.py input.json output.json
    python additional_filter.py input.json  # overwrites input file
"""

import os
import json
import sys
import argparse
from typing import List, Dict, Any
import re


def filter_low_context_triplets(qa_triplets: List[Dict[str, Any]], verbose: bool = True) -> List[Dict[str, Any]]:
    """
    Enhanced filtering to remove question-answer-context triplets that lack sufficient context.
    Designed for structure with context_text, question_text, and answer_text.
    
    Args:
        qa_triplets: List of dictionaries with 'context_text', 'question_text' and 'answer_text' keys
        verbose: Whether to print filtering details
        
    Returns:
        List of filtered Q&A triplets
    """
    filtered_triplets = []
    removed_count = 0
    removal_reasons = {
        'empty_fields': 0,
        'visual_reference_no_context': 0,
        'numbered_sequence_no_context': 0,
        'diagram_parts_no_context': 0,
        'experimental_setup_no_context': 0,
        'table_data_no_context': 0,
        'figure_reference_no_context': 0,
        'ordering_sequence_no_context': 0,
        'location_reference_no_context': 0,
        'comparative_reference_no_context': 0,
        'too_short_question': 0,
        'meaningless_short_answers': 0,
        'fallback_answers': 0,
        'incomplete_questions': 0,
        'question_fragments': 0,
        'insufficient_context_quality': 0,
        'mcq_without_options': 0
    }
    
    for triplet in qa_triplets:
        context = triplet.get("context_text", "")
        question = triplet.get("question_text", "").strip()
        answer = triplet.get("answer_text", "").strip()
        
        # Skip if question or answer is empty
        if not question or not answer:
            removed_count += 1
            removal_reasons['empty_fields'] += 1
            continue
            
        # Convert to lowercase for case-insensitive matching
        context_lower = context.lower() if context else ""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Flag for insufficient context
        insufficient_context = False
        removal_reason = None
        
        # 1. ENHANCED: Questions referencing visual elements without adequate context
        visual_element_patterns = [
            r'\bdiagram\b', r'\bfigure\b', r'\bchart\b', r'\bimage\b', r'\billustration\b',
            r'\bpicture\b', r'\bgraph\b', r'\btable\b', r'\bshown above\b', r'\bshown below\b',
            r'\bin the figure\b', r'\bfrom the diagram\b', r'\busing the chart\b',
            r'\baccording to the table\b', r'\bfrom the graph\b'
        ]
        
        has_visual_reference = any(re.search(pattern, question_lower) for pattern in visual_element_patterns)
        
        if has_visual_reference:
            # Check if context provides adequate description
            if not context or len(context.strip()) < 30:
                insufficient_context = True
                removal_reason = 'visual_reference_no_context'
                if verbose:
                    print(f"🚫 Visual reference without context: {question[:60]}...")
            else:
                # Context exists, but check if it's meaningful
                context_quality_indicators = [
                    "shows", "depicts", "illustrates", "contains", "displays", "represents",
                    "with", "labeled", "marked", "pointing", "indicating", "structure",
                    "experiment", "apparatus", "setup", "data", "results", "values"
                ]
                
                has_quality_context = (
                    len(context.split()) >= 10 and
                    any(indicator in context_lower for indicator in context_quality_indicators)
                )
                
                if not has_quality_context:
                    insufficient_context = True
                    removal_reason = 'insufficient_context_quality'
                    if verbose:
                        print(f"🚫 Poor quality context: {question[:60]}...")
        
        # 2. ENHANCED: Numbered sequence questions (major issue in your data)
        numbered_sequence_patterns = [
            r'\(\d+\)', r'\b\d+\s*,\s*\d+', r'\b\d+\s*and\s*\d+', 
            r'\b\d+\s*to\s*\d+', r'\bfigure\s*\(\d+\)', r'\bpetiole.*\(\d+\)'
        ]
        
        sequence_answer_patterns = [
            r'\(\d+\)\s*,\s*\(\d+\)', r'\b\d+\s*,\s*\d+\s*,\s*\d+', 
            r'\(\d+\)\s*and\s*\(\d+\)', r'\b\d+\s*→\s*\d+', r'\b\d+\s*and\s*\d+'
        ]
        
        has_numbered_question = any(re.search(pattern, question) for pattern in numbered_sequence_patterns)
        has_sequence_answer = any(re.search(pattern, answer) for pattern in sequence_answer_patterns)
        
        if (has_numbered_question or has_sequence_answer) and "arrange" in question_lower:
            # This is clearly a sequencing question - needs detailed context
            if not context or len(context.strip()) < 50:
                insufficient_context = True
                removal_reason = 'numbered_sequence_no_context'
                if verbose:
                    print(f"🚫 Numbered sequence without context: {question[:60]}...")
            else:
                # Check if context explains what each number represents
                sequence_context_indicators = [
                    "appearance", "condition", "stage", "step", "phase", "state",
                    "solution", "concentration", "potential", "shows", "represents"
                ]
                
                has_sequence_context = any(indicator in context_lower for indicator in sequence_context_indicators)
                
                if not has_sequence_context:
                    insufficient_context = True
                    removal_reason = 'ordering_sequence_no_context'
                    if verbose:
                        print(f"🚫 Sequence without explanation: {question[:60]}...")
        
        # 3. ENHANCED: Part identification questions (A, B, C, D references)
        part_identification_patterns = [
            r'\bpart [a-z]\b', r'\bstructure [a-z]\b', r'\blabelled [a-z]\b',
            r'\bwhat is [a-z]\b', r'\bidentify [a-z]\b', r'\bname [a-z]\b',
            r'\bfunction of [a-z]\b', r'\brole of [a-z]\b'
        ]
        
        single_letter_answer = len(answer.strip()) == 1 and answer.strip().isalpha()
        
        has_part_identification = any(re.search(pattern, question_lower) for pattern in part_identification_patterns)
        
        if has_part_identification or single_letter_answer:
            # Check for MCQ options in question
            mcq_option_patterns = [
                r'[a-d]\)', r'[a-d]\.', r'option [a-d]', r'choice [a-d]',
                r'\([a-d]\)', r'a\s+[^)]+\s+b\s+[^)]+\s+c\s+[^)]+\s+d\s+'
            ]
            
            has_mcq_options = any(re.search(pattern, question_lower) for pattern in mcq_option_patterns)
            
            if single_letter_answer and not has_mcq_options:
                # Single letter answer without MCQ options - needs context
                if not context or len(context.strip()) < 25:
                    insufficient_context = True
                    removal_reason = 'diagram_parts_no_context'
                    if verbose:
                        print(f"🚫 Part identification without context: {question[:60]}...")
                else:
                    # Check if context describes what the parts are
                    part_description_indicators = [
                        "pointing", "labeled", "marked", "shows", "indicates",
                        "with", "where", "represents", "structure", "part", "component"
                    ]
                    
                    has_part_description = any(indicator in context_lower for indicator in part_description_indicators)
                    
                    if not has_part_description:
                        insufficient_context = True
                        removal_reason = 'diagram_parts_no_context'
                        if verbose:
                            print(f"🚫 Parts without description: {question[:60]}...")
            elif single_letter_answer and not has_mcq_options:
                # Single letter without context or MCQ - likely poor extraction
                insufficient_context = True
                removal_reason = 'mcq_without_options'
                if verbose:
                    print(f"🚫 Single letter answer without MCQ options: {question[:40]}...")
        
        # 4. ENHANCED: Experimental setup references
        experimental_indicators = [
            "experiment", "investigation", "procedure", "method", "apparatus",
            "equipment", "setup", "test tube", "beaker", "reaction"
        ]
        
        has_experimental_ref = any(indicator in question_lower for indicator in experimental_indicators)
        
        if has_experimental_ref:
            if not context or len(context.strip()) < 40:
                insufficient_context = True
                removal_reason = 'experimental_setup_no_context'
                if verbose:
                    print(f"🚫 Experimental reference without context: {question[:60]}...")
            else:
                # Check if context describes the experimental setup
                setup_description_indicators = [
                    "contains", "placed", "added", "mixed", "heated", "cooled",
                    "temperature", "concentration", "solution", "sample", "material"
                ]
                
                has_setup_description = any(indicator in context_lower for indicator in setup_description_indicators)
                
                if not has_setup_description:
                    insufficient_context = True
                    removal_reason = 'experimental_setup_no_context'
                    if verbose:
                        print(f"🚫 Experiment without setup description: {question[:60]}...")
        
        # 5. ENHANCED: Table and data references
        data_reference_indicators = [
            "table", "data", "results", "values", "measurements", "readings",
            "concentration", "temperature", "time", "rate", "percentage"
        ]
        
        has_data_reference = any(indicator in question_lower for indicator in data_reference_indicators)
        
        if has_data_reference and ("table" in question_lower or "data" in question_lower):
            if not context or len(context.strip()) < 35:
                insufficient_context = True
                removal_reason = 'table_data_no_context'
                if verbose:
                    print(f"🚫 Table/data reference without context: {question[:60]}...")
            else:
                # Check if context provides actual data or table description
                data_context_indicators = [
                    "shows", "contains", "data", "values", "results", "measurements",
                    "temperature", "concentration", "time", "rate", "percentage", ":"
                ]
                
                has_data_context = any(indicator in context_lower for indicator in data_context_indicators)
                
                if not has_data_context:
                    insufficient_context = True
                    removal_reason = 'table_data_no_context'
                    if verbose:
                        print(f"🚫 Data reference without actual data: {question[:60]}...")
        
        # 6. ENHANCED: Location and spatial references
        location_indicators = [
            "above", "below", "left", "right", "top", "bottom", "center",
            "position", "location", "where", "region", "area", "section"
        ]
        
        has_location_ref = any(indicator in question_lower for indicator in location_indicators)
        
        if has_location_ref and any(visual in question_lower for visual in ["diagram", "figure", "image"]):
            if not context or len(context.strip()) < 30:
                insufficient_context = True
                removal_reason = 'location_reference_no_context'
                if verbose:
                    print(f"🚫 Location reference without context: {question[:60]}...")
        
        # 7. ENHANCED: Comparative questions
        comparative_indicators = [
            "compare", "difference", "similar", "contrast", "between",
            "higher than", "lower than", "more than", "less than"
        ]
        
        has_comparative = any(indicator in question_lower for indicator in comparative_indicators)
        
        if has_comparative and not context:
            # Comparative questions often need context to understand what's being compared
            insufficient_context = True
            removal_reason = 'comparative_reference_no_context'
            if verbose:
                print(f"🚫 Comparative question without context: {question[:60]}...")
        
        # 8. Very short questions (enhanced)
        word_count = len(question.split())
        if word_count < 5:
            insufficient_context = True
            removal_reason = 'too_short_question'
            if verbose:
                print(f"🚫 Too short: {question}")
        elif word_count < 8:
            # Short questions need either context or quality indicators
            quality_indicators = [
                "which", "what", "how", "why", "where", "when", "describe", "explain",
                "compare", "contrast", "identify", "name", "state", "suggest", "predict",
                "calculate", "determine", "find"
            ]
            has_quality = any(indicator in question_lower for indicator in quality_indicators)
            
            if not has_quality and not context:
                insufficient_context = True
                removal_reason = 'too_short_question'
                if verbose:
                    print(f"🚫 Short without context or quality indicators: {question}")
        
        # 9. Meaningless short answers (enhanced)
        if len(answer.strip()) <= 3:
            # Very short answers need to be meaningful
            meaningless_patterns = [
                r'^[a-d]$',  # Single letter A, B, C, D without MCQ
                r'^[xyz]$',  # Single variable letters
                r'^n/?a$',   # N/A variations
                r'^\d+$' if len(answer.strip()) == 1 else None  # Single digits (sometimes OK)
            ]
            
            meaningless_patterns = [p for p in meaningless_patterns if p is not None]
            
            is_meaningless = any(re.match(pattern, answer.strip().lower()) for pattern in meaningless_patterns)
            
            if is_meaningless:
                # Check if it's a valid MCQ answer
                mcq_in_question = any(re.search(pattern, question_lower) for pattern in [
                    r'[a-d]\)', r'option [a-d]', r'choice [a-d]'
                ])
                
                if not mcq_in_question:
                    insufficient_context = True
                    removal_reason = 'meaningless_short_answers'
                    if verbose:
                        print(f"🚫 Meaningless short answer: '{answer}' for {question[:40]}...")
        
        # 10. Fallback answers
        fallback_patterns = [
            "answer not found", "answer not understood", "refer to diagram",
            "see image", "check figure", "not provided", "unclear",
            "cannot determine", "insufficient information", "not available"
        ]
        
        if any(fallback in answer_lower for fallback in fallback_patterns):
            insufficient_context = True
            removal_reason = 'fallback_answers'
            if verbose:
                print(f"🚫 Fallback answer: {answer}")
        
        # 11. Incomplete questions
        incomplete_patterns = [
            r'\.{3,}', r'\betc\.?\b', r'\band so on\b', r'\(continued\)',
            r'\[incomplete\]', r'\bsee next page\b', r'\brefer to previous\b'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in incomplete_patterns):
            insufficient_context = True
            removal_reason = 'incomplete_questions'
            if verbose:
                print(f"🚫 Incomplete question: {question[:60]}...")
        
        # 12. Question fragments
        fragment_indicators = [
            question_lower.startswith(("and ", "or ", "but ", "also ")),
            not any(char in question for char in '.?!'),  # No ending punctuation
            question.count('(') != question.count(')'),  # Unmatched parentheses
            question.count('"') % 2 != 0,  # Unmatched quotes
        ]
        
        if any(fragment_indicators):
            insufficient_context = True
            removal_reason = 'question_fragments'
            if verbose:
                print(f"🚫 Question fragment: {question}")
        
        # If the triplet passes all filters, add it
        if not insufficient_context:
            filtered_triplets.append(triplet)
        else:
            removed_count += 1
            if removal_reason:
                removal_reasons[removal_reason] += 1
    
    # Convert null context_text to standard message
    for triplet in filtered_triplets:
        if triplet.get("context_text") is None or triplet.get("context_text") == "":
            triplet["context_text"] = "Sufficient context provided in question."
    
    # Print detailed summary
    print(f"\n📊 Enhanced Filtering Summary:")
    print(f"  ├─ Original triplets: {len(qa_triplets)}")
    print(f"  ├─ Kept triplets: {len(filtered_triplets)}")
    print(f"  └─ Removed triplets: {removed_count}")
    
    if removed_count > 0:
        print(f"\n🔍 Removal Breakdown:")
        for reason, count in removal_reasons.items():
            if count > 0:
                reason_name = reason.replace('_', ' ').title()
                print(f"  ├─ {reason_name}: {count}")
    
    # Additional statistics for v5 triplets
    with_context = sum(1 for t in filtered_triplets if t.get("context_text") and t.get("context_text") != "Sufficient context provided in question.")
    without_context = len(filtered_triplets) - with_context
    print(f"\n📋 Context Distribution:")
    print(f"  ├─ With context: {with_context}")
    print(f"  └─ Without context: {without_context}")
    
    if filtered_triplets:
        # Calculate average context length excluding the standard message
        contexts_with_actual_content = [t.get("context_text", "") for t in filtered_triplets 
                                      if t.get("context_text") and t.get("context_text") != "Sufficient context provided in question."]
        if contexts_with_actual_content:
            avg_context_length = sum(len(ctx) for ctx in contexts_with_actual_content) / len(contexts_with_actual_content)
            print(f"  └─ Average context length: {avg_context_length:.1f} characters")
        else:
            print(f"  └─ Average context length: N/A (no actual context content)")
    
    return filtered_triplets


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array of objects")
        
        return data
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in '{file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading '{file_path}': {e}")
        sys.exit(1)


def save_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save JSON data to file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Filtered data saved to '{file_path}'")
    except Exception as e:
        print(f"❌ Error saving to '{file_path}': {e}")
        sys.exit(1)


def validate_triplet_structure(data: List[Dict[str, Any]]) -> bool:
    """Validate that the data has the expected triplet structure."""
    if not data:
        print("⚠️ Warning: Empty dataset")
        return True
    
    # Check first few items for expected structure
    sample_size = min(5, len(data))
    for i, item in enumerate(data[:sample_size]):
        if not isinstance(item, dict):
            print(f"❌ Error: Item {i} is not a dictionary")
            return False
        
        required_fields = ['question_text', 'answer_text']
        optional_fields = ['context_text']
        
        for field in required_fields:
            if field not in item:
                print(f"❌ Error: Item {i} missing required field '{field}'")
                print(f"   Available fields: {list(item.keys())}")
                return False
    
    # Check for v5 structure (with context_text)
    has_context_field = any('context_text' in item for item in data[:sample_size])
    if has_context_field:
        print(f"✅ Detected v5 triplet structure (with context_text)")
    else:
        print(f"⚠️ Warning: No context_text field detected - treating as v4 structure")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced filter for low-context question-answer-context triplets (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python v2_filter_low_context.py input.json output.json    # Save filtered data to new file
  python v2_filter_low_context.py input.json               # Overwrite input file with filtered data
  python v2_filter_low_context.py input.json --quiet       # Run without verbose output
        """
    )
    
    parser.add_argument('input_file', help='Input JSON file containing triplets')
    parser.add_argument('output_file', nargs='?', help='Output JSON file (optional, defaults to input file)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output')
    parser.add_argument('--backup', '-b', action='store_true', help='Create backup of input file before overwriting')
    
    args = parser.parse_args()
    
    # Set output file
    output_file = args.output_file if args.output_file else args.input_file
    
    # Create backup if overwriting and backup requested
    if args.backup and output_file == args.input_file:
        backup_file = f"{args.input_file}.backup"
        try:
            import shutil
            shutil.copy2(args.input_file, backup_file)
            print(f"📋 Backup created: {backup_file}")
        except Exception as e:
            print(f"⚠️ Warning: Could not create backup: {e}")
    
    print(f"🔍 Loading triplet data from '{args.input_file}'...")
    
    # Load and validate data
    triplet_data = load_json_file(args.input_file)
    
    if not validate_triplet_structure(triplet_data):
        sys.exit(1)
    
    print(f"📝 Loaded {len(triplet_data)} triplets")
    
    # Apply enhanced filtering
    print(f"\n🧹 Applying enhanced context filters...")
    filtered_data = filter_low_context_triplets(triplet_data, verbose=not args.quiet)
    
    # Save results
    save_json_file(filtered_data, output_file)
    
    # Final summary
    original_count = len(triplet_data)
    filtered_count = len(filtered_data)
    removed_count = original_count - filtered_count
    retention_rate = (filtered_count / original_count * 100) if original_count > 0 else 0
    
    print(f"\n🎉 Enhanced filtering complete!")
    print(f"   Original: {original_count} triplets")
    print(f"   Retained: {filtered_count} triplets ({retention_rate:.1f}%)")
    print(f"   Removed:  {removed_count} triplets")
    
    if filtered_data:
        # Show sample of remaining data
        sample = filtered_data[0]
        print(f"\n📋 Sample triplet structure:")
        print(f"  Context: {(sample.get('context_text') or 'None')[:80]}...")
        print(f"  Question: {sample.get('question_text', '')[:80]}...")
        print(f"  Answer: {sample.get('answer_text', '')[:80]}...")


if __name__ == "__main__":
    main()