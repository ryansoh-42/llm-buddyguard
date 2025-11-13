import os
import json
import time
import re
import argparse
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from tqdm import tqdm

# Global constants
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "asia-southeast1")
MODEL_NAME = "gemini-2.5-flash"

def setup_global_variables(subject_filepath, subject):
    """Setup global variables based on CLI arguments"""
    global QUESTIONS_FOLDER, ANSWERS_FOLDER, DEFAULT_FOLDER, OUTPUT_JSON_FILE
    
    QUESTIONS_FOLDER = f"papers/{subject_filepath}/questions" # e.g: phy/pure_phy
    ANSWERS_FOLDER = f"papers/{subject_filepath}/answers"
    DEFAULT_FOLDER = f"papers/{subject_filepath}/default"  # 👈 Folder for Q&A-in-one PDFs
    OUTPUT_JSON_FILE = f"data/json/{subject}_dataset.json"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)

# --- Initialize Vertex AI ---

# --- Function 1: Enhanced Intelligent File Matching ---

def find_best_match_with_gemini(question_filename, all_answer_filenames, max_retries=2):
    """
    Enhanced version: Asks Gemini to find the best matching answer filename for a given question filename.
    Includes confidence scoring, fallback logic, and better error handling.
    """
    if not all_answer_filenames:
        return None
        
    answer_list_str = "\n".join(f"{i+1}. {fname}" for i, fname in enumerate(all_answer_filenames))
    
    enhanced_prompt = f"""
    You are an expert file matching system for educational exam papers. Your task is to find the SINGLE best answer sheet that corresponds to the given question paper.

    TARGET QUESTION PAPER:
    {question_filename}

    AVAILABLE ANSWER SHEETS:
    {answer_list_str}

    MATCHING CRITERIA (ALL MUST BE SATISFIED):
    1. EXACT SCHOOL MATCH: Same institution name (MANDATORY - consider abbreviations like "Sec" = "Secondary", "CHIJ" = "Convent of the Holy Infant Jesus")
    2. SUBJECT MATCH: Same subject (Biology, Chemistry, Physics, etc.)
    3. YEAR MATCH: Same examination year
    4. PAPER TYPE MATCH: Same paper number/type (P1, P2, P3, P4, etc.)
    5. LEVEL MATCH: Same educational level (Sec 4, 4E, 4N, etc.)
    6. EXAM TYPE MATCH: Same exam type (Prelim, Mid-Year, Final, etc.)

    CRITICAL REQUIREMENT: The SCHOOL/INSTITUTION MUST BE THE SAME. Do NOT match files from different schools, even if all other criteria match perfectly.

    COMMON PATTERNS TO RECOGNIZE:
    - Question papers often have "QP", "Question", "Qns" in filename
    - Answer sheets often have "MS", "Answer", "Ans", "Solution", "Soln" in filename
    - School abbreviations: PLMGS = Paya Lebar Methodist Girls' School, CHS = Catholic High School, CHIJ = Convent of the Holy Infant Jesus
    - Subject codes: 5087 = Combined Science Biology, 5088 = Pure Biology
    - Paper types: P1 = Paper 1, P4 = Paper 4 (practical)

    SCHOOL MATCHING EXAMPLES:
    - "CHS" matches only "CHS" or "Catholic High School"
    - "PLMGS" matches only "PLMGS" or "Paya Lebar Methodist Girls School"
    - "CHIJ" matches only "CHIJ" or "Convent of the Holy Infant Jesus"
    - Different schools should NEVER be matched together

    Return a JSON object with:
    - "best_match": The exact filename of the best match (or null if no good match)
    - "confidence": A score from 0.0 to 1.0 indicating match quality
    - "match_reason": Brief explanation of why this match was chosen
    - "alternative_matches": Array of up to 2 other possible matches (if any)

    IMPORTANT: 
    - Only return a match with confidence >= 0.8 AND school names match exactly
    - If no answer sheet is from the same school, return null for best_match
    - School mismatch automatically disqualifies a match regardless of other criteria
    """
    
    for attempt in range(max_retries + 1):
        try:
            response = model.generate_content(
                enhanced_prompt, 
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.1  # Lower temperature for more consistent matching
                }
            )
            
            result = json.loads(response.text)
            
            # Validate the response structure
            if not isinstance(result, dict):
                raise ValueError("Response is not a JSON object")
                
            best_match = result.get("best_match")
            confidence = result.get("confidence", 0.0)
            match_reason = result.get("match_reason", "No reason provided")
            
            # Additional validation
            if best_match and best_match not in all_answer_filenames:
                tqdm.write(f"⚠️ Gemini returned invalid filename: {best_match}")
                best_match = None
            # Log the matching decision
            if best_match:
                tqdm.write(f"✅ Match found: {question_filename[:30]}... → {best_match[:30]}... (confidence: {confidence:.2f})")
                tqdm.write(f"   Reason: {match_reason}")
            else:
                tqdm.write(f"❌ No suitable match for: {question_filename} (best confidence: {confidence:.2f})")
                
            return best_match
            
        except json.JSONDecodeError as e:
            tqdm.write(f"⚠️ JSON decode error on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                tqdm.write(f"❗️ Failed to parse JSON after {max_retries + 1} attempts for: {question_filename}")
                return None
                
        except Exception as e:
            tqdm.write(f"❗️ Error matching '{question_filename}' on attempt {attempt + 1}: {e}")
            if attempt < max_retries:
                time.sleep(1)
                continue
            else:
                return None
    
    return None

def find_best_match_with_fallback(question_filename, all_answer_filenames):
    """
    Enhanced matching with rule-based fallback if Gemini fails.
    """
    # First try Gemini
    gemini_match = find_best_match_with_gemini(question_filename, all_answer_filenames)
    
    if gemini_match:
        return gemini_match
    
    # Fallback to rule-based matching
    tqdm.write(f"🔄 Trying rule-based fallback for: {question_filename}")
    
    q_lower = question_filename.lower()
    
    # Extract key components from question filename
    q_tokens = set(q_lower.replace('.pdf', '').replace('_', ' ').replace('-', ' ').split())
    
    best_score = 0
    best_match = None
    
    for answer_filename in all_answer_filenames:
        a_lower = answer_filename.lower()
        a_tokens = set(a_lower.replace('.pdf', '').replace('_', ' ').replace('-', ' ').split())
        
        score = 0
        
        # School name matching (MANDATORY - must match exactly)
        school_indicators = ['plmgs', 'chij', 'tkgs', 'qtss', 'mfss', 'kcpss', 'nchs', 'chs', 'catholic high', 'raffles']
        school_match_found = False
        for school in school_indicators:
            if school in q_lower and school in a_lower:
                score += 100  # Very high weight for school matching
                school_match_found = True
                break
        
        # If no school match found, automatically disqualify this answer
        if not school_match_found:
            # Check if we can identify any school indicators in either filename
            q_has_school = any(school in q_lower for school in school_indicators)
            a_has_school = any(school in a_lower for school in school_indicators)
            
            # If both files have school indicators but they don't match, skip this answer
            if q_has_school and a_has_school:
                continue  # Skip this answer entirely
        
        # Year matching (high weight)
        years = ['2024', '2025', '2023', '2022']
        for year in years:
            if year in q_lower and year in a_lower:
                score += 25
                break
        
        # Subject matching
        subjects = ['bio', 'biology', 'chem', 'chemistry', 'physics', 'phy']
        for subject in subjects:
            if subject in q_lower and subject in a_lower:
                score += 20
                break
        
        # Paper type matching
        papers = ['p1', 'p2', 'p3', 'p4', 'paper']
        for paper in papers:
            if paper in q_lower and paper in a_lower:
                score += 15
                break
        
        # Level matching
        levels = ['4e', '4n', 'sec4', 'secondary']
        for level in levels:
            if level in q_lower and level in a_lower:
                score += 10
                break
        
        # General token overlap (only if school already matches)
        if school_match_found:
            common_tokens = q_tokens.intersection(a_tokens)
            score += len(common_tokens) * 2
        
        # Only consider matches with school match AND high score
        if score > best_score and score >= 120:  # Increased threshold (school match = 100 + other criteria)
            best_score = score
            best_match = answer_filename
    
    if best_match:
        tqdm.write(f"✅ Rule-based match: {question_filename[:30]}... → {best_match[:30]}... (score: {best_score})")
    else:
        tqdm.write(f"❌ No fallback match found for: {question_filename}")
    
    return best_match


# --- Function for Default Folder Processing ---

def check_contains_both_qa(pdf_path):
    """
    Check if a PDF contains both questions and answers/marking scheme.
    Returns True if both are present, False otherwise.
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")

        check_prompt = """
        You are a document analyzer for educational exam papers. Analyze this PDF to determine if it contains BOTH questions AND answers/marking scheme in the same document.

        Look for these indicators:
        
        QUESTION INDICATORS:
        - Question numbers (1., 2., 3., etc. or (a), (b), (c))
        - Question text asking for answers
        - Multiple choice options (A), (B), (C), (D)
        - Instructions like "Answer all questions"
        - Blank spaces or lines for answers
        
        ANSWER/MARKING SCHEME INDICATORS:
        - "Marking Scheme", "Answer Key", "Solutions", "MS"
        - Provided answers to questions
        - Point allocations (e.g., "[2 marks]", "1 point")
        - Model answers or sample responses
        - Answer explanations or workings
        - "Accept" or "Do not accept" criteria
        
        Return a JSON object with:
        - "has_questions": true/false - whether the document contains questions
        - "has_answers": true/false - whether the document contains answers/marking scheme
        - "confidence": 0.0 to 1.0 - confidence in the assessment
        - "evidence": brief description of what was found
        
        IMPORTANT: 
        - Both must be present to be useful for processing
        - A document with only questions should return has_answers: false
        - A document with only answers should return has_questions: false
        """
        
        response = model.generate_content(
            [check_prompt, pdf_part],
            generation_config={"response_mime_type": "application/json"}
        )
        
        result = json.loads(response.text)
        
        has_questions = result.get("has_questions", False)
        has_answers = result.get("has_answers", False)
        confidence = result.get("confidence", 0.0)
        evidence = result.get("evidence", "No evidence provided")
        
        contains_both = has_questions and has_answers and confidence >= 0.7
        
        filename = os.path.basename(pdf_path)
        if contains_both:
            tqdm.write(f"✅ Both Q&A found: {filename[:40]}... (confidence: {confidence:.2f})")
            tqdm.write(f"   Evidence: {evidence}")
        else:
            tqdm.write(f"❌ Incomplete document: {filename[:40]}... (Q:{has_questions}, A:{has_answers}, conf:{confidence:.2f})")
            tqdm.write(f"   Evidence: {evidence}")
        
        return contains_both
        
    except Exception as e:
        tqdm.write(f"❗️ Error checking {os.path.basename(pdf_path)}: {e}")
        return False

def extract_qa_from_combined_pdf(pdf_path):
    """
    Extract question-answer triplets from a single PDF that contains both questions and answers.
    Uses the same robust context extraction as the paired PDF function.
    """
    try:
        with open(pdf_path, "rb") as f:
            pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")

        # Enhanced prompt for combined Q&A documents
        enhanced_prompt = """
        You are an expert educational data extraction system. This PDF contains BOTH questions AND answers/marking scheme in the same document.
        
        Extract question-answer pairs with COMPREHENSIVE context from this combined PDF.
        
        Your goal: A human should be able to answer the question using ONLY the context_text provided, without seeing any diagrams or visual elements.
        
        Return a JSON array where each object has: "context_text", "question_text", "answer_text"
        
        CRITICAL CONTEXT EXTRACTION RULES:
        
        1. **context_text** - COMPREHENSIVE DESCRIPTIONS:
           
           For DIAGRAMS/FIGURES:
           - Describe EVERY labeled part with its letter/number: "The diagram shows a plant cell where A is the nucleus (large circular structure in center), B is the cytoplasm (gel-like substance filling the cell), C is the cell membrane (thin boundary around the cell), D is the chloroplast (green oval structures), and E is the cell wall (thick outer boundary)"
           - Include positioning: "A is located at the top left, B is in the center, C surrounds the entire structure"
           - Describe visual appearance: "A appears as a dark circular structure, B is a light-colored region, C shows a thick black line"
           
           For EXPERIMENTAL SETUPS:
           - Complete apparatus description: "The experiment shows three test tubes in a rack. Test tube 1 (left) contains 2ml of glucose solution + 3 drops of Benedict's reagent, appears blue. Test tube 2 (center) contains 2ml of starch solution + 2 drops of iodine solution, appears brown. Test tube 3 (right) contains 2ml of distilled water as control, appears clear."
           - Include ALL conditions: temperatures, concentrations, time periods, pH levels
           - Describe what each component looks like BEFORE and AFTER any reactions
           
           For DATA TABLES:
           - Include ALL data points: "The table shows enzyme activity data: At 10°C = 5 units/min, At 20°C = 15 units/min, At 30°C = 30 units/min, At 40°C = 45 units/min, At 50°C = 35 units/min, At 60°C = 10 units/min, At 70°C = 2 units/min"
           - Include units, headings, and any trends visible
           
           For GRAPHS/CHARTS:
           - Describe axes, scales, data points, trends: "The graph shows pH (x-axis, 0-14) vs enzyme activity rate (y-axis, 0-50 units/min). The line shows activity starts at 5 units at pH 2, rises gradually to peak at 45 units at pH 7.5, then drops sharply to 3 units at pH 12"
           
           For SEQUENCES/ORDERING:
           - Describe each numbered item completely: "Figure (1) shows a fresh Coleus petiole that is straight and firm. Figure (2) shows the same petiole after 30 minutes in high water potential solution - it appears swollen and curved upward. Figure (3) shows the petiole in medium water potential - slightly curved. Figure (4) shows the petiole in low water potential - wilted and curved downward. Figure (5) shows the petiole in very low water potential - severely wilted and drooping."
           
           **MINIMUM CONTEXT STANDARDS:**
           - Context must be self-sufficient (person can answer without seeing visual)
           - Include ALL relevant details, measurements, labels, conditions
           - Minimum 40 words for visual questions, 20 words for others
           - If insufficient detail available, mark as null rather than vague description
        
        2. **question_text**:
           - Clean question without visual references ("part A" → keep as is, but context must explain what A is)
           - Include ALL MCQ options with full text
           - Keep technical terms and specific details
           - Remove only question numbers and excess formatting
        
        3. **answer_text**:
           - Extract the corresponding answer from the marking scheme section
           - Resolve symbolic answers using context: "A" → "nucleus" (if A was described as nucleus in context)
           - Include complete explanations and reasoning from the marking scheme
           - For calculations, show full working from the answer key
           - For sequences, explain the logic: "(5), (2), (4), (3)" → "The order from lowest to highest water potential is: very low (5), low (2), medium (4), high (3) water potential, based on the degree of wilting observed"
        
        IMPORTANT FOR COMBINED DOCUMENTS:
        - Match questions with their corresponding answers from the marking scheme
        - Ensure question numbers align with answer numbers
        - Extract complete marking criteria and explanations
        - Maintain the same quality standards as separate Q&A pairs
        
        QUALITY CONTROLS:
        - Each triplet must pass the "human sufficiency test" - can a person answer using only the context?
        - Context must explain what ALL letters, numbers, and references mean
        - No vague descriptions like "the diagram shows..." without specifics
        - Include measurements, conditions, observations, and relationships
        
        EXAMPLES:
        
        GOOD Example (Combined document):
        {
          "context_text": "The diagram shows a cross-section of a leaf where A is the upper epidermis (single layer of transparent cells at the top), B is the palisade mesophyll (tightly packed column-shaped cells containing many chloroplasts, located just below the upper epidermis), C is the spongy mesophyll (loosely arranged round cells with air spaces between them, containing fewer chloroplasts), and D is the lower epidermis (single layer of cells at the bottom with small openings called stomata for gas exchange).",
          "question_text": "Which part is primarily responsible for photosynthesis?",
          "answer_text": "Part B (palisade mesophyll) is primarily responsible for photosynthesis because it contains the highest concentration of chloroplasts and receives the most direct sunlight due to its position just below the transparent upper epidermis. [2 marks: 1 mark for identifying B, 1 mark for explanation]"
        }
        """
        
        response = model.generate_content(
            [enhanced_prompt, pdf_part],
            generation_config={"response_mime_type": "application/json"}
        )
        extracted_triplets = json.loads(response.text)
        
        # Apply the same validation logic as the paired function
        validated_triplets = []
        for triplet in extracted_triplets:
            context = triplet.get("context_text", "").strip() if triplet.get("context_text") else None
            question = triplet.get("question_text", "").strip()
            answer = triplet.get("answer_text", "").strip()
            
            if not question or not answer:
                continue
            
            # Enhanced context quality validation (same as paired function)
            question_lower = question.lower()
            
            # Check if question requires visual context
            visual_indicators = [
                "diagram", "figure", "chart", "image", "graph", "table", "picture",
                "shown above", "shown below", "part a", "part b", "part c", "part d",
                "structure a", "structure b", "labeled", "arrange", "order",
                "experiment setup", "apparatus", "test tube", "beaker"
            ]
            
            needs_visual_context = any(indicator in question_lower for indicator in visual_indicators)
            
            # Check for numbered/lettered references in question or answer
            has_references = bool(
                re.search(r'\b(part |structure |tube |figure )?[a-z]\b', question_lower) or
                re.search(r'\(\d+\)', question) or
                re.search(r'\b\d+\s*,\s*\d+', answer) or
                re.search(r'\([a-z]\)', answer.lower())
            )
            
            if needs_visual_context or has_references:
                # Questions with visual elements MUST have substantial context
                if not context or len(context) < 40:
                    tqdm.write(f"⚠️ Skipping: Insufficient context for visual question: {question[:50]}...")
                    continue
                
                # Check context quality - must contain descriptive words
                context_quality_indicators = [
                    "shows", "contains", "displays", "depicts", "illustrates", "located",
                    "appears", "labeled", "where", "with", "including", "consisting",
                    "temperature", "concentration", "solution", "structure", "cell",
                    "at", "in", "on", "above", "below", "left", "right", "center"
                ]
                
                context_lower = context.lower()
                quality_score = sum(1 for indicator in context_quality_indicators if indicator in context_lower)
                
                if quality_score < 3:  # Need at least 3 quality indicators
                    tqdm.write(f"⚠️ Skipping: Poor context quality for: {question[:50]}...")
                    continue
                
                # For lettered/numbered references, context must explain what they mean
                if has_references:
                    # Check if context explains the references
                    question_refs = set(re.findall(r'\b[a-z]\b', question_lower))
                    answer_refs = set(re.findall(r'\b[a-z]\b', answer.lower()))
                    all_refs = question_refs.union(answer_refs)
                    
                    if all_refs:
                        context_explains_refs = all(ref in context_lower for ref in all_refs)
                        if not context_explains_refs:
                            tqdm.write(f"⚠️ Skipping: Context doesn't explain all references: {question[:50]}...")
                            continue
            
            # Clean up context - set to None if too short or meaningless
            if context:
                if len(context) < 15:
                    context = None
                elif context.lower().strip() in ["none", "n/a", "not applicable", "no context needed"]:
                    context = None
                    
            # Enhanced answer resolution for symbolic answers
            if len(answer.strip()) == 1 and answer.strip().isalpha() and context:
                # Try to resolve single letter answers using context
                letter = answer.strip().lower()
                context_lower = context.lower()
                
                # Look for pattern "letter is/represents/points to something"
                patterns = [
                    rf'{letter}\s+is\s+(?:the\s+)?([^,.]+)',
                    rf'{letter}\s+represents\s+(?:the\s+)?([^,.]+)',
                    rf'{letter}\s+points?\s+to\s+(?:the\s+)?([^,.]+)',
                    rf'where\s+{letter}\s+(?:is\s+)?(?:the\s+)?([^,.]+)',
                    rf'{letter}\s*[:=]\s*([^,.]+)'
                ]
                
                resolved = None
                for pattern in patterns:
                    match = re.search(pattern, context_lower)
                    if match:
                        resolved = match.group(1).strip()
                        if len(resolved) > 3:  # Meaningful resolution
                            answer = resolved.title()
                            break
                
                if not resolved:
                    # Keep original single letter if no good resolution found
                    pass
            
            validated_triplets.append({
                "context_text": context,
                "question_text": question,
                "answer_text": answer,
                "source": "combined"
            })
        
        return validated_triplets
        
    except Exception as e:
        tqdm.write(f"❗️ Error extracting from {os.path.basename(pdf_path)}: {e}")
        return []


# --- Function 3: Enhanced Q&A Extraction with Context Separation ---

def extract_qa_triplets(question_pdf_path, answer_pdf_path):
    """
    Extracts (context, question, answer) triplets from a matched set of two PDFs.
    Separates contextual information from the actual questions.
    """
    try:
        with open(question_pdf_path, "rb") as f:
            q_pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")
        with open(answer_pdf_path, "rb") as f:
            a_pdf_part = Part.from_data(data=f.read(), mime_type="application/pdf")

        # ENHANCED ROBUST CONTEXT EXTRACTION PROMPT
        enhanced_prompt = """
        You are an expert educational data extraction system. Extract question-answer pairs with COMPREHENSIVE context from these PDFs.
        
        Your goal: A human should be able to answer the question using ONLY the context_text provided, without seeing any diagrams or visual elements.
        
        Return a JSON array where each object has: "context_text", "question_text", "answer_text"
        
        CRITICAL CONTEXT EXTRACTION RULES:
        
        1. **context_text** - COMPREHENSIVE DESCRIPTIONS:
           
           For DIAGRAMS/FIGURES:
           - Describe EVERY labeled part with its letter/number: "The diagram shows a plant cell where A is the nucleus (large circular structure in center), B is the cytoplasm (gel-like substance filling the cell), C is the cell membrane (thin boundary around the cell), D is the chloroplast (green oval structures), and E is the cell wall (thick outer boundary)"
           - Include positioning: "A is located at the top left, B is in the center, C surrounds the entire structure"
           - Describe visual appearance: "A appears as a dark circular structure, B is a light-colored region, C shows a thick black line"
           
           For EXPERIMENTAL SETUPS:
           - Complete apparatus description: "The experiment shows three test tubes in a rack. Test tube 1 (left) contains 2ml of glucose solution + 3 drops of Benedict's reagent, appears blue. Test tube 2 (center) contains 2ml of starch solution + 2 drops of iodine solution, appears brown. Test tube 3 (right) contains 2ml of distilled water as control, appears clear."
           - Include ALL conditions: temperatures, concentrations, time periods, pH levels
           - Describe what each component looks like BEFORE and AFTER any reactions
           
           For DATA TABLES:
           - Include ALL data points: "The table shows enzyme activity data: At 10°C = 5 units/min, At 20°C = 15 units/min, At 30°C = 30 units/min, At 40°C = 45 units/min, At 50°C = 35 units/min, At 60°C = 10 units/min, At 70°C = 2 units/min"
           - Include units, headings, and any trends visible
           
           For GRAPHS/CHARTS:
           - Describe axes, scales, data points, trends: "The graph shows pH (x-axis, 0-14) vs enzyme activity rate (y-axis, 0-50 units/min). The line shows activity starts at 5 units at pH 2, rises gradually to peak at 45 units at pH 7.5, then drops sharply to 3 units at pH 12"
           
           For SEQUENCES/ORDERING:
           - Describe each numbered item completely: "Figure (1) shows a fresh Coleus petiole that is straight and firm. Figure (2) shows the same petiole after 30 minutes in high water potential solution - it appears swollen and curved upward. Figure (3) shows the petiole in medium water potential - slightly curved. Figure (4) shows the petiole in low water potential - wilted and curved downward. Figure (5) shows the petiole in very low water potential - severely wilted and drooping."
           
           **MINIMUM CONTEXT STANDARDS:**
           - Context must be self-sufficient (person can answer without seeing visual)
           - Include ALL relevant details, measurements, labels, conditions
           - Minimum 40 words for visual questions, 20 words for others
           - If insufficient detail available, mark as null rather than vague description
        
        2. **question_text**:
           - Clean question without visual references ("part A" → keep as is, but context must explain what A is)
           - Include ALL MCQ options with full text
           - Keep technical terms and specific details
           - Remove only question numbers and excess formatting
        
        3. **answer_text**:
           - Resolve symbolic answers using context: "A" → "nucleus" (if A was described as nucleus in context)
           - Include complete explanations and reasoning
           - For calculations, show full working
           - For sequences, explain the logic: "(5), (2), (4), (3)" → "The order from lowest to highest water potential is: very low (5), low (2), medium (4), high (3) water potential, based on the degree of wilting observed"
        
        QUALITY CONTROLS:
        - Each triplet must pass the "human sufficiency test" - can a person answer using only the context?
        - Context must explain what ALL letters, numbers, and references mean
        - No vague descriptions like "the diagram shows..." without specifics
        - Include measurements, conditions, observations, and relationships
        
        EXAMPLES:
        
        GOOD Example (Visual question):
        {
          "context_text": "The diagram shows a cross-section of a leaf where A is the upper epidermis (single layer of transparent cells at the top), B is the palisade mesophyll (tightly packed column-shaped cells containing many chloroplasts, located just below the upper epidermis), C is the spongy mesophyll (loosely arranged round cells with air spaces between them, containing fewer chloroplasts), and D is the lower epidermis (single layer of cells at the bottom with small openings called stomata for gas exchange).",
          "question_text": "Which part is primarily responsible for photosynthesis?",
          "answer_text": "Part B (palisade mesophyll) is primarily responsible for photosynthesis because it contains the highest concentration of chloroplasts and receives the most direct sunlight due to its position just below the transparent upper epidermis."
        }
        
        BAD Example (Insufficient context):
        {
          "context_text": "The diagram shows a plant cell with labeled parts A, B, C, D",
          "question_text": "What is the function of part A?",
          "answer_text": "Cell wall"
        }
        """
        
        response = model.generate_content(
            [enhanced_prompt, q_pdf_part, a_pdf_part],
            generation_config={"response_mime_type": "application/json"}
        )
        extracted_triplets = json.loads(response.text)
        
        # Enhanced post-processing and validation with robust context checking
        validated_triplets = []
        for triplet in extracted_triplets:
            context = triplet.get("context_text", "").strip() if triplet.get("context_text") else None
            question = triplet.get("question_text", "").strip()
            answer = triplet.get("answer_text", "").strip()
            
            if not question or not answer:
                continue
            
            # Enhanced context quality validation
            question_lower = question.lower()
            
            # Check if question requires visual context
            visual_indicators = [
                "diagram", "figure", "chart", "image", "graph", "table", "picture",
                "shown above", "shown below", "part a", "part b", "part c", "part d",
                "structure a", "structure b", "labeled", "arrange", "order",
                "experiment setup", "apparatus", "test tube", "beaker"
            ]
            
            needs_visual_context = any(indicator in question_lower for indicator in visual_indicators)
            
            # Check for numbered/lettered references in question or answer
            has_references = bool(
                re.search(r'\b(part |structure |tube |figure )?[a-z]\b', question_lower) or
                re.search(r'\(\d+\)', question) or
                re.search(r'\b\d+\s*,\s*\d+', answer) or
                re.search(r'\([a-z]\)', answer.lower())
            )
            
            if needs_visual_context or has_references:
                # Questions with visual elements MUST have substantial context
                if not context or len(context) < 40:
                    tqdm.write(f"⚠️ Skipping: Insufficient context for visual question: {question[:50]}...")
                    continue
                
                # Check context quality - must contain descriptive words
                context_quality_indicators = [
                    "shows", "contains", "displays", "depicts", "illustrates", "located",
                    "appears", "labeled", "where", "with", "including", "consisting",
                    "temperature", "concentration", "solution", "structure", "cell",
                    "at", "in", "on", "above", "below", "left", "right", "center"
                ]
                
                context_lower = context.lower()
                quality_score = sum(1 for indicator in context_quality_indicators if indicator in context_lower)
                
                if quality_score < 3:  # Need at least 3 quality indicators
                    tqdm.write(f"⚠️ Skipping: Poor context quality for: {question[:50]}...")
                    continue
                
                # For lettered/numbered references, context must explain what they mean
                if has_references:
                    # Check if context explains the references
                    question_refs = set(re.findall(r'\b[a-z]\b', question_lower))
                    answer_refs = set(re.findall(r'\b[a-z]\b', answer.lower()))
                    all_refs = question_refs.union(answer_refs)
                    
                    if all_refs:
                        context_explains_refs = all(ref in context_lower for ref in all_refs)
                        if not context_explains_refs:
                            tqdm.write(f"⚠️ Skipping: Context doesn't explain all references: {question[:50]}...")
                            continue
            
            # Clean up context - set to None if too short or meaningless
            if context:
                if len(context) < 15:
                    context = None
                elif context.lower().strip() in ["none", "n/a", "not applicable", "no context needed"]:
                    context = None
                    
            # Enhanced answer resolution for symbolic answers
            if len(answer.strip()) == 1 and answer.strip().isalpha() and context:
                # Try to resolve single letter answers using context
                letter = answer.strip().lower()
                context_lower = context.lower()
                
                # Look for pattern "letter is/represents/points to something"
                patterns = [
                    rf'{letter}\s+is\s+(?:the\s+)?([^,.]+)',
                    rf'{letter}\s+represents\s+(?:the\s+)?([^,.]+)',
                    rf'{letter}\s+points?\s+to\s+(?:the\s+)?([^,.]+)',
                    rf'where\s+{letter}\s+(?:is\s+)?(?:the\s+)?([^,.]+)',
                    rf'{letter}\s*[:=]\s*([^,.]+)'
                ]
                
                resolved = None
                for pattern in patterns:
                    match = re.search(pattern, context_lower)
                    if match:
                        resolved = match.group(1).strip()
                        if len(resolved) > 3:  # Meaningful resolution
                            answer = resolved.title()
                            break
                
                if not resolved:
                    # Keep original single letter if no good resolution found
                    pass
            
            validated_triplets.append({
                "context_text": context,
                "question_text": question,
                "answer_text": answer,
                "source": "paired"
            })
        
        return validated_triplets
        
    except Exception as e:
        tqdm.write(f"❗️ Error extracting from {os.path.basename(question_pdf_path)}: {e}")
        return []


# --- Function 4: Robust Context Quality Filter ---
def filter_low_context_questions(qa_triplets):
    """
    ENHANCED filtering for context-question-answer triplets with human sufficiency test.
    Goal: A human should be able to answer the question using ONLY the context provided.
    """
    filtered_triplets = []
    removed_count = 0
    removal_reasons = {
        'empty_fields': 0,
        'visual_reference_insufficient_context': 0,
        'numbered_sequence_no_context': 0,
        'part_identification_poor_context': 0,
        'experimental_setup_incomplete': 0,
        'table_data_missing': 0,
        'single_letter_unresolved': 0,
        'comparative_no_baseline': 0,
        'calculation_missing_data': 0,
        'too_short_question': 0,
        'meaningless_answers': 0,
        'fallback_answers': 0,
        'incomplete_questions': 0,
        'question_fragments': 0,
        'insufficient_context_quality': 0,
        'human_sufficiency_fail': 0
    }
    
    for triplet in qa_triplets:
        context = triplet.get("context_text", "")
        question = triplet.get("question_text", "").strip()
        answer = triplet.get("answer_text", "").strip()
        
        # Skip if either question or answer is empty
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
        
        # 1. ENHANCED: Visual reference questions - MUST have comprehensive context
        visual_element_patterns = [
            r'\bdiagram\b', r'\bfigure\b', r'\bchart\b', r'\bimage\b', r'\billustration\b',
            r'\bpicture\b', r'\bgraph\b', r'\btable\b', r'\bshown above\b', r'\bshown below\b',
            r'\bin the figure\b', r'\bfrom the diagram\b', r'\busing the chart\b',
            r'\baccording to the table\b', r'\bfrom the graph\b'
        ]
        
        has_visual_reference = any(re.search(pattern, question_lower) for pattern in visual_element_patterns)
        
        if has_visual_reference:
            if not context or len(context.strip()) < 50:  # Increased minimum length
                insufficient_context = True
                removal_reason = 'visual_reference_insufficient_context'
                tqdm.write(f"🚫 Visual reference needs detailed context: {question[:50]}...")
            else:
                # Check for comprehensive description indicators
                comprehensive_indicators = [
                    "shows", "depicts", "illustrates", "contains", "displays", "represents",
                    "labeled", "marked", "where", "located", "positioned", "appears",
                    "structure", "part", "component", "section", "region", "area"
                ]
                
                context_quality_score = sum(1 for indicator in comprehensive_indicators if indicator in context_lower)
                
                if context_quality_score < 4:  # Need multiple quality indicators
                    insufficient_context = True
                    removal_reason = 'insufficient_context_quality'
                    tqdm.write(f"🚫 Insufficient context depth: {question[:50]}...")
        
        # 2. ENHANCED: Part identification questions (A, B, C, D references)
        part_reference_patterns = [
            r'\bpart [a-z]\b', r'\bstructure [a-z]\b', r'\blabelled [a-z]\b',
            r'\bregion [a-z]\b', r'\barea [a-z]\b', r'\bsection [a-z]\b'
        ]
        
        has_part_reference = any(re.search(pattern, question_lower) for pattern in part_reference_patterns)
        single_letter_answer = len(answer.strip()) == 1 and answer.strip().isalpha()
        
        if has_part_reference or single_letter_answer:
            if not context or len(context.strip()) < 30:
                insufficient_context = True
                removal_reason = 'part_identification_poor_context'
                tqdm.write(f"🚫 Part identification needs context: {question[:50]}...")
            else:
                # Check if context explains what the parts represent
                if single_letter_answer:
                    letter = answer.strip().lower()
                    letter_explained = (
                        f"{letter} is" in context_lower or
                        f"{letter} represents" in context_lower or
                        f"{letter} points" in context_lower or
                        f"{letter}:" in context_lower or
                        f"{letter} =" in context_lower
                    )
                    
                    if not letter_explained and "option" not in question_lower and "choice" not in question_lower:
                        insufficient_context = True
                        removal_reason = 'single_letter_unresolved'
                        tqdm.write(f"🚫 Single letter '{answer}' not explained in context: {question[:40]}...")
        
        # 3. ENHANCED: Numbered sequence questions (critical for your data)
        sequence_patterns = [
            r'\(\d+\)', r'\bfigure\s*\(\d+\)', r'\bstage\s*\d+', r'\bstep\s*\d+',
            r'\barrange.*order', r'\bsequence\b', r'\bchronological\b'
        ]
        
        sequence_answer_patterns = [
            r'\(\d+\)\s*,\s*\(\d+\)', r'\b\d+\s*,\s*\d+\s*,\s*\d+', 
            r'\(\d+\)\s*and\s*\(\d+\)', r'\b\d+\s*→\s*\d+'
        ]
        
        has_sequence_question = any(re.search(pattern, question) for pattern in sequence_patterns)
        has_sequence_answer = any(re.search(pattern, answer) for pattern in sequence_answer_patterns)
        
        if has_sequence_question or has_sequence_answer:
            if not context or len(context.strip()) < 60:  # Higher threshold for sequences
                insufficient_context = True
                removal_reason = 'numbered_sequence_no_context'
                tqdm.write(f"🚫 Sequence question needs detailed context: {question[:50]}...")
            else:
                # Check if context describes each numbered element
                sequence_description_indicators = [
                    "shows", "appears", "depicts", "condition", "stage", "step",
                    "solution", "concentration", "potential", "after", "before",
                    "minutes", "hours", "temperature", "treatment"
                ]
                
                has_sequence_descriptions = sum(1 for indicator in sequence_description_indicators if indicator in context_lower) >= 3
                
                if not has_sequence_descriptions:
                    insufficient_context = True
                    removal_reason = 'numbered_sequence_no_context'
                    tqdm.write(f"🚫 Sequence lacks detailed descriptions: {question[:50]}...")
        
        # 4. ENHANCED: Experimental setup questions
        experimental_indicators = [
            "experiment", "investigation", "procedure", "method", "apparatus",
            "equipment", "setup", "test tube", "beaker", "reaction", "flask"
        ]
        
        has_experimental_ref = any(indicator in question_lower for indicator in experimental_indicators)
        
        if has_experimental_ref:
            if not context or len(context.strip()) < 50:
                insufficient_context = True
                removal_reason = 'experimental_setup_incomplete'
                tqdm.write(f"🚫 Experimental setup needs detailed context: {question[:50]}...")
            else:
                # Check for experimental detail indicators
                setup_details = [
                    "contains", "placed", "added", "mixed", "heated", "cooled",
                    "temperature", "concentration", "solution", "ml", "drops",
                    "reagent", "indicator", "control", "variable"
                ]
                
                setup_detail_score = sum(1 for detail in setup_details if detail in context_lower)
                
                if setup_detail_score < 3:  # Need substantial experimental details
                    insufficient_context = True
                    removal_reason = 'experimental_setup_incomplete'
                    tqdm.write(f"🚫 Insufficient experimental details: {question[:50]}...")
        
        # 5. ENHANCED: Table and data questions
        data_indicators = [
            "table", "data", "results", "values", "measurements", "readings",
            "rate", "percentage", "concentration", "temperature"
        ]
        
        has_data_reference = any(indicator in question_lower for indicator in data_indicators)
        
        if has_data_reference:
            if not context or len(context.strip()) < 40:
                insufficient_context = True
                removal_reason = 'table_data_missing'
                tqdm.write(f"🚫 Data question needs context: {question[:50]}...")
            else:
                # Check if context contains actual data
                data_evidence = [
                    re.search(r'\d+\.?\d*\s*(°c|celsius|units|ml|%|percent)', context_lower),
                    re.search(r'\d+\s*=\s*\d+', context_lower),
                    ":" in context and any(char.isdigit() for char in context),
                    context.count(',') >= 2  # Multiple data points
                ]
                
                has_actual_data = any(data_evidence)
                
                if not has_actual_data:
                    insufficient_context = True
                    removal_reason = 'table_data_missing'
                    tqdm.write(f"🚫 Data reference without actual data: {question[:50]}...")
        
        # 6. ENHANCED: Comparative questions
        comparative_indicators = [
            "compare", "difference", "contrast", "higher", "lower", "more", "less",
            "better", "worse", "increased", "decreased", "similar", "different"
        ]
        
        has_comparative = any(indicator in question_lower for indicator in comparative_indicators)
        
        if has_comparative:
            if not context or len(context.strip()) < 35:
                insufficient_context = True
                removal_reason = 'comparative_no_baseline'
                tqdm.write(f"🚫 Comparative question needs context: {question[:50]}...")
        
        # 7. ENHANCED: Calculation questions
        calculation_indicators = [
            "calculate", "determine", "find", "work out", "compute", "percentage",
            "rate", "concentration", "ratio", "total", "average"
        ]
        
        has_calculation = any(indicator in question_lower for indicator in calculation_indicators)
        
        if has_calculation:
            if not context or len(context.strip()) < 25:
                insufficient_context = True
                removal_reason = 'calculation_missing_data'
                tqdm.write(f"🚫 Calculation needs data context: {question[:50]}...")
            else:
                # Check for numerical data in context
                has_numbers = bool(re.search(r'\b\d+\.?\d*\b', context))
                
                if not has_numbers:
                    insufficient_context = True
                    removal_reason = 'calculation_missing_data'
                    tqdm.write(f"🚫 Calculation lacks numerical data: {question[:50]}...")
        
        # 8. ENHANCED: Human sufficiency test
        if context and question and not insufficient_context:
            # Advanced sufficiency checks
            sufficiency_failures = []
            
            # Check if question references things not explained in context
            question_nouns = re.findall(r'\b[A-Z][a-z]+\b', question)  # Proper nouns
            for noun in question_nouns:
                if noun.lower() not in context_lower and len(noun) > 3:
                    sufficiency_failures.append(f"'{noun}' not explained")
            
            # Check for unexplained technical terms
            technical_terms = [
                "osmosis", "diffusion", "enzyme", "respiration", "photosynthesis",
                "mitosis", "meiosis", "transpiration", "plasmolysis"
            ]
            
            for term in technical_terms:
                if term in question_lower and term not in context_lower:
                    # This might be OK if it's general knowledge
                    pass
            
            # If too many unexplained references, fail
            if len(sufficiency_failures) >= 2:
                insufficient_context = True
                removal_reason = 'human_sufficiency_fail'
                tqdm.write(f"🚫 Human sufficiency fail: {', '.join(sufficiency_failures[:2])}: {question[:40]}...")
        
        # 9. Standard quality checks (enhanced)
        word_count = len(question.split())
        if word_count < 4:
            insufficient_context = True
            removal_reason = 'too_short_question'
            tqdm.write(f"🚫 Too short: {question}")
        
        # 10. Meaningless answers
        meaningless_patterns = [
            r'^[a-d]$',  # Single letter without MCQ context
            r'^n/?a$',   # N/A variations
            r'^x+$',     # Just X's
            r'^\d$'      # Single digit
        ]
        
        is_meaningless = any(re.match(pattern, answer.strip().lower()) for pattern in meaningless_patterns)
        
        if is_meaningless and "option" not in question_lower and "choice" not in question_lower:
            insufficient_context = True
            removal_reason = 'meaningless_answers'
            tqdm.write(f"🚫 Meaningless answer: '{answer}' for {question[:40]}...")
        
        # 11. Fallback answers
        fallback_patterns = [
            "answer not found", "answer not understood", "refer to diagram",
            "see image", "check figure", "not provided", "unclear",
            "cannot determine", "insufficient information", "not available"
        ]
        
        if any(fallback in answer_lower for fallback in fallback_patterns):
            insufficient_context = True
            removal_reason = 'fallback_answers'
            tqdm.write(f"🚫 Fallback answer: {answer}")
        
        # If the triplet passes all enhanced filters, add it
        if not insufficient_context:
            filtered_triplets.append(triplet)
        else:
            removed_count += 1
            if removal_reason:
                removal_reasons[removal_reason] += 1
    
    # Enhanced summary with detailed breakdown
    print(f"\n📊 ENHANCED Context Quality Filtering Summary:")
    print(f"  ├─ Original triplets: {len(qa_triplets)}")
    print(f"  ├─ Kept triplets: {len(filtered_triplets)}")
    print(f"  └─ Removed triplets: {removed_count}")
    
    if removed_count > 0:
        print(f"\n🔍 Detailed Removal Breakdown:")
        for reason, count in removal_reasons.items():
            if count > 0:
                reason_name = reason.replace('_', ' ').title()
                percentage = (count / len(qa_triplets)) * 100
                print(f"  ├─ {reason_name}: {count} ({percentage:.1f}%)")
    
    # Context distribution analysis
    with_context = sum(1 for t in filtered_triplets if t.get("context_text"))
    without_context = len(filtered_triplets) - with_context
    
    print(f"\n📋 Context Quality Distribution:")
    print(f"  ├─ With substantial context: {with_context}")
    print(f"  └─ Without context (general knowledge): {without_context}")
    
    if with_context > 0:
        avg_context_length = sum(len(t.get("context_text") or "") for t in filtered_triplets if t.get("context_text")) / with_context
        print(f"  └─ Average context length: {avg_context_length:.1f} characters")
    
    # Quality assurance check
    visual_questions_remaining = sum(1 for t in filtered_triplets 
                                   if any(indicator in t.get("question_text", "").lower() 
                                         for indicator in ["diagram", "figure", "part ", "structure "])
                                   and t.get("context_text"))
    
    print(f"\n✅ Quality Assurance:")
    print(f"  └─ Visual questions with context: {visual_questions_remaining}")
    
    return filtered_triplets


# --- Main Execution Script ---
def main(subject_filepath, subject):
    # Setup global variables based on arguments
    setup_global_variables(subject_filepath, subject)
    
    # This list will hold ALL extracted Q&A triplets from all sources
    final_dataset = []

    # --- Step 1: Intelligent Matching (Questions & Answers folders) ---
    print("--- 🤖 Step 1: Intelligently Matching Questions to Answers ---")
    question_files = [f for f in os.listdir(QUESTIONS_FOLDER) if f.lower().endswith('.pdf')]
    answer_files = [f for f in os.listdir(ANSWERS_FOLDER) if f.lower().endswith('.pdf')]

    valid_pairs = []
    available_answers = list(answer_files) # Create a mutable list
    
    for q_filename in tqdm(question_files, desc="Matching files"):
        if not available_answers:
            tqdm.write("No more answer sheets available to match.")
            break
            
        # Use enhanced matching with fallback
        best_match_filename = find_best_match_with_fallback(q_filename, available_answers)
        
        if best_match_filename and best_match_filename in available_answers:
            valid_pairs.append({
                "question_path": os.path.join(QUESTIONS_FOLDER, q_filename),
                "answer_path": os.path.join(ANSWERS_FOLDER, best_match_filename)
            })
            available_answers.remove(best_match_filename)
        else:
            tqdm.write(f"❌ No match found for '{q_filename}', ignoring.")
            
        time.sleep(0.5)  # Reduced sleep time due to better error handling

    print(f"\nFound {len(valid_pairs)} valid pairs to process.")

    # --- Step 2: Process Combined Q&A Documents from Default Folder ---
    print("\n--- 📄 Step 2: Processing Combined Q&A Documents from Default Folder ---")
    
    combined_files_processed = 0
    if os.path.exists(DEFAULT_FOLDER):
        default_files = [f for f in os.listdir(DEFAULT_FOLDER) if f.lower().endswith('.pdf')]
        
        if default_files:
            print(f"Found {len(default_files)} files in default folder to check...")
            
            for filename in tqdm(default_files, desc="Processing combined documents"):
                file_path = os.path.join(DEFAULT_FOLDER, filename)
                
                # Check if document contains both questions and answers
                if check_contains_both_qa(file_path):
                    # Extract Q&A triplets from the combined document
                    extracted_triplets = extract_qa_from_combined_pdf(file_path)
                    
                    if extracted_triplets:
                        # Filter out incomplete objects
                        clean_triplets = [
                            t for t in extracted_triplets 
                            if t.get("question_text") and t.get("answer_text")
                        ]
                        final_dataset.extend(clean_triplets)
                        combined_files_processed += 1
                        tqdm.write(f"✅ Extracted {len(clean_triplets)} triplets from: {filename}")
                    else:
                        tqdm.write(f"❌ No valid triplets extracted from: {filename}")
                else:
                    tqdm.write(f"⏭️  Skipping (questions only or incomplete): {filename}")
                
                time.sleep(0.3)  # Rate limit for API calls
        else:
            print("No PDF files found in default folder.")
    else:
        print(f"Default folder not found: {DEFAULT_FOLDER}")
    
    print(f"Processed {combined_files_processed} combined Q&A documents from default folder.")

    # --- Step 3: Context-Question-Answer Extraction from PAIRS ---
    print("\n--- 📝 Step 3: Extracting Context-Question-Answer Triplets from Paired Documents ---")
    for pair in tqdm(valid_pairs, desc="Extracting Triplets"):
        extracted_triplets = extract_qa_triplets(pair["question_path"], pair["answer_path"])
        
        if extracted_triplets:
            # Filter out incomplete objects (missing question or answer)
            clean_triplets = [
                t for t in extracted_triplets 
                if t.get("question_text") and t.get("answer_text")
            ]
            final_dataset.extend(clean_triplets)
            
        time.sleep(0.2) # Rate limit

    # --- Step 4: Filter out low-quality triplets ---
    print(f"\n--- 🔍 Step 4: Filtering out low-quality triplets ---")
    print(f"Before filtering: {len(final_dataset)} triplets")
    
    # Save dataset BEFORE filtering
    pre_filter_file = f"../../data/json/before_filter_{subject}_triplets.json"
    os.makedirs(os.path.dirname(pre_filter_file), exist_ok=True)
    
    with open(pre_filter_file, 'w', encoding='utf-8') as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)
    print(f"💾 Pre-filter dataset saved to '{pre_filter_file}'")
    
    # Apply filtering
    filtered_dataset = filter_low_context_questions(final_dataset)
    
    print(f"After filtering: {len(filtered_dataset)} triplets")
    removed_count = len(final_dataset) - len(filtered_dataset)
    print(f"Removed {removed_count} triplets with insufficient quality")
    
    # Update final_dataset to the filtered version for the final save
    final_dataset = filtered_dataset

    # --- Step 5: Save Final Dataset ---
    print("\n--- 🏁 Process Complete ---")
    if final_dataset:
        print(f"Successfully extracted a total of {len(final_dataset)} complete triplets from all sources.")
        with open(OUTPUT_JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        print(f"✅ Final clean dataset saved to '{OUTPUT_JSON_FILE}'")
        
        # Print sample data structure
        if final_dataset:
            sample = final_dataset[0]
            print(f"\n📋 Sample triplet structure:")
            context_preview = sample.get('context_text', 'None')
            if context_preview and len(context_preview) > 100:
                context_preview = context_preview[:100] + "..."
            print(f"  Context: {context_preview}")
            print(f"  Question: {sample.get('question_text', '')[:100]}...")
            print(f"  Answer: {sample.get('answer_text', '')[:100]}...")
            
        # Context quality analysis
        with_context = sum(1 for t in final_dataset if t.get("context_text"))
        substantial_context = sum(1 for t in final_dataset if t.get("context_text") and len(t.get("context_text", "")) >= 50)
        
        # Calculate triplets from different sources
        triplets_from_combined = sum(1 for t in final_dataset if t.get("source") == "combined")
        triplets_from_pairs = sum(1 for t in final_dataset if t.get("source") == "paired")
        
        print(f"\n📊 Final Dataset Quality Metrics:")
        print(f"  ├─ Total triplets: {len(final_dataset)}")
        print(f"  ├─ From paired documents: {triplets_from_pairs}")
        print(f"  ├─ From combined documents: {triplets_from_combined}")
        print(f"  ├─ Combined files processed: {combined_files_processed}")
        print(f"  ├─ With context: {with_context}")
        print(f"  ├─ Substantial context (50+ chars): {substantial_context}")
        print(f"  └─ Context coverage: {(with_context/len(final_dataset)*100):.1f}%")
        
        # Recommendation for further filtering
        print(f"\n💡 RECOMMENDATION:")
        print(f"   For even higher quality, apply additional filtering using:")
        print(f"   python ../step3\ \(combine\ and\ clean\)/v2_filter_low_context.py {OUTPUT_JSON_FILE}")
        print(f"   This applies the 'Human Sufficiency Test' for maximum context quality.")
        
    else:
        print("❗️ No complete triplets were extracted. The output file was not created.")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Extract question-answer triplets from matched PDF pairs with enhanced context processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s bio/pure_bio pure_bio
  python %(prog)s chem/pure_chem pure_chem
  python %(prog)s emath/combined_math combined_math

Available subjects and their paths:
  - Biology (Pure): bio/pure_bio pure_bio
  - Chemistry (Pure): chem/pure_chem pure_chem
  - Chemistry (Combined): chem/comb_chem comb_chem
  - Math: emath/combined_math combined_math

Directory Structure Expected:
  ../../papers/{subject_filepath}/questions/  (Question PDFs)
  ../../papers/{subject_filepath}/answers/    (Answer PDFs)
  ../../papers/{subject_filepath}/default/    (Combined Q&A PDFs)

Output:
  ../../data/json/after_filter_{subject}_dataset.json
        """
    )
    
    parser.add_argument(
        'subject_filepath',
        help='Subject folder path (e.g., "bio/pure_bio", "chem/pure_chem")'
    )
    
    parser.add_argument(
        'subject',
        help='Subject name for output file (e.g., "pure_bio", "pure_chem")'
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Initialize Vertex AI
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        print(f"✅ Vertex AI initialized successfully")
        print(f"📁 Processing: {args.subject_filepath}")
        print(f"🏷️ Subject: {args.subject}")
    except Exception as e:
        print(f"❌ Error initializing Vertex AI. Have you authenticated?")
        print(f"Run 'gcloud auth application-default login' in your terminal.")
        print(f"Error details: {e}")
        exit()
    
    # Run the main processing function
    main(args.subject_filepath, args.subject)