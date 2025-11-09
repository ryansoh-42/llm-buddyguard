# src/guardrails.py
from typing import Dict, List, Tuple, Optional
from guardrails import Guard, OnFailAction
from guardrails.hub import (
    ToxicLanguage,
    DetectPII,
    ProfanityFree,
    RestrictToTopic,
    BanList,
)
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EducationalGuardrails:
    """
    Comprehensive safety and educational guardrails using Guardrails AI library.
    Uses pre-built validators from Guardrails Hub for dynamic detection.
    """
    def __init__(self, strict_mode: bool = True):
        """
        Initialize guardrails system with Guardrails AI validators.
        
        Args:
            strict_mode: If True, use stricter filtering (default: True)
        """
        self.strict_mode = strict_mode
        
        # Configure threshold based on strictness
        toxicity_threshold = 0.3 if strict_mode else 0.5
        
        # Define allowed topics for RestrictToTopic (only Physics, Chemistry, Biology)
        allowed_topics = [
            "physics", "chemistry", "biology", "science",
            "measurement", "physical quantities", "units and measurement",
            "newtonian mechanics", "kinematics", "dynamics", "turning effect of forces", "pressure", "energy",
            "thermal physics", "kinetic particle model of matter", "thermal processes", "thermal properties of matter",
            "waves", "general properties of waves", "electromagnetic spectrum", "light",
            "electricity and magnetism", "static electricity", "current of electricity", "direct current circuits", "practical electricity", "magnetism", "electromagnetism", "electromagnetic induction",
            "radioactivity",
            "matter", "structure and properties", "experimental chemistry", "the particulate nature of matter", "chemical bonding and structure",
            "chemical reactions", "chemical calculations", "acid-base chemistry", "qualitative analysis", "redox chemistry", "patterns in the periodic table", "chemical energetics", "rate of reactions",
            "organic chemistry", "maintaining air quality",
            "cells and the chemistry of life", "cell structure and organisation", "movement of substances", "biological molecules",
            "the human body - maintaing life", "nutrition in humans", "transport in humans", "respiration in humans", "excretion in humans", "homeostasis, coordination and response in humans", "infectious diseases in humans",
            "living together - plants, animals and ecosystems", "nutrition and transport in flowering plants", "organisms and their environment",
            "continuity of life", "molecular genetics", "reproduction", "inheritance"
        ]
        
        # Define invalid topics for RestrictToTopic (not allowed topics)
        invalid_topics = [
            "terrorism", "extremism", "radicalization", "terrorist propaganda", "militant groups", "domestic terrorism", 
            "violent ideologies", "extreme political views", "hate speech", "hate groups", "hate crimes", "racism", 
            "discrimination", "sexism", "homophobia", "xenophobia", "anti-Semitism", "bullying", "cyberbullying", 
            "body shaming", "child abuse", "sexual harassment", "violence", "physical assault", "rape", "domestic violence", 
            "violence against women", "child exploitation", "human trafficking", "pedophilia", "gang violence", 
            "violent protests", "murder", "assault", "self-harm", "suicide", "self-destructive behavior", "anorexia", 
            "bulimia", "eating disorders", "pro-anorexia", "dangerous challenges", "mental illness stigma", "drug abuse", 
            "substance dependency", "addiction", "drug trafficking", "drug manufacturing", "illegal activities", 
            "criminal activity", "criminal gangs", "violence against animals", "animal cruelty", "weaponry", "bomb-making", 
            "militarization", "extreme nationalism", "radical political conspiracies", "terrorist recruitment", 
            "cult behavior", "death cults", "cultism", "extreme dieting", "unsafe sexual practices", "sex work", 
            "prostitution", "illegal drug use", "pornography", "graphic content", "pornographic material", "incest", 
            "stalking", "rape culture", "date rape", "human experimentation", "human rights violations", "euthanasia", 
            "death threats", "mass shootings", "violent video games (without context)", "military violence", 
            "weaponized violence", "dangerous online behavior", "child labor", "child soldiers", 
            "dangerous political movements", "military training", "criminal recruitment", "hate speech based on religion", 
            "extreme political ideologies", "extreme conspiracy theories"
        ]

        # High-risk words to block explicitly
        self.banned_words = [
            "hijack", "bomb", "terrorism", "self-harm", "suicide"
        ]
        
        # Create input guard for user prompts
        self.input_guard = Guard().use_many(
            # Toxic language detection
            ToxicLanguage(
                threshold=toxicity_threshold,
                validation_method="sentence",
                on_fail=OnFailAction.EXCEPTION
            ),
            # PII detection
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
                             "SSN", "IP_ADDRESS", "DATE_TIME", "LOCATION"],
                on_fail=OnFailAction.EXCEPTION
            ),
            # Profanity detection
            ProfanityFree(
                on_fail=OnFailAction.EXCEPTION
            ),
            # Topic restriction - ensure content is about allowed subjects
            RestrictToTopic(
                valid_topics=allowed_topics,
                invalid_topics=invalid_topics,
                on_fail=OnFailAction.EXCEPTION
            ),
            # Explicit ban list for violent / criminal content
            BanList(
                banned_words=self.banned_words,
                on_fail=OnFailAction.EXCEPTION
            )
        )
        
        # Create output guard for model responses (without topic restriction)
        self.output_guard = Guard().use_many(
            ToxicLanguage(
                threshold=toxicity_threshold,
                validation_method="sentence",
                on_fail=OnFailAction.EXCEPTION
            ),
            DetectPII(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", 
                             "SSN", "IP_ADDRESS", "PASSPORT_NUMBER"],
                on_fail=OnFailAction.EXCEPTION
            ),
            ProfanityFree(
                on_fail=OnFailAction.EXCEPTION
            ),
            BanList(
                banned_words=self.banned_words,
                on_fail=OnFailAction.EXCEPTION
            ),
        )
        
        # Educational-specific patterns for answer-seeking detection
        self.answer_seeking_patterns = [
            r"give me the answer",
            r"what is the answer",
            r"just tell me",
            r"solve this for me",
            r"do my homework",
            r"complete this for me",
            r"what's the solution"
        ]
    
    def check_safety_with_guardrails(self, prompt: str, is_input: bool = True) -> Dict:
        """
        Use Guardrails AI validators to check prompt safety.
        
        Args:
            prompt: Text to validate
            is_input: True if validating user input, False if validating model output
            
        Returns:
            Dictionary with 'allowed', 'message', 'violations'
        """
        guard = self.input_guard if is_input else self.output_guard
        
        try:
            # Validate using Guardrails AI
            validated = guard.validate(prompt)
            
            return {
                "allowed": True,
                "message": "Prompt approved",
                "violations": [],
                "validated_text": validated if isinstance(validated, str) else prompt
            }
            
        except Exception as e:
            # Extract violation details from exception
            error_message = str(e)
            violations = []
            detected_content_list = []  # Collect all detected content from multiple violations
            
            # Parse error message to identify violation type and extract details
            error_lower = error_message.lower()
            
            if "toxic" in error_lower or "toxiclanguage" in error_lower:
                # Try to extract specific toxic content if available
                content = self._extract_detected_content(error_message, "toxic")
                if content:
                    detected_content_list.append(f"toxic: {content}")
                violations.append({
                    "category": "toxic",
                    "severity": "high",
                    "reason": "Content contains toxic, harmful, or inappropriate language",
                    "details": "The message includes language that may be offensive, harmful, or inappropriate for an educational environment."
                })
            
            if "pii" in error_lower or "detectpii" in error_lower:
                # Try to extract PII types detected
                content = self._extract_detected_content(error_message, "pii")
                if content:
                    detected_content_list.append(f"pii: {content}")
                pii_types = self._extract_pii_types(error_message)
                reason = "Content contains personally identifiable information"
                if pii_types:
                    reason += f" ({', '.join(pii_types)})"
                violations.append({
                    "category": "pii",
                    "severity": "high",
                    "reason": reason,
                    "details": "Personal information such as email addresses, phone numbers, credit cards, or other sensitive data was detected."
                })
            
            if "profanity" in error_lower or "profanityfree" in error_lower:
                content = self._extract_detected_content(error_message, "profanity")
                if content:
                    detected_content_list.append(f"profanity: {content}")
                violations.append({
                    "category": "profanity",
                    "severity": "medium",
                    "reason": "Content contains profane or inappropriate language",
                    "details": "The message includes words or phrases that are not appropriate for an educational setting."
                })
            
            if "topic" in error_lower or "restricttotopic" in error_lower:
                content = self._extract_detected_content(error_message, "topic")
                if content:
                    detected_content_list.append(f"topic: {content}")
                violations.append({
                    "category": "off_topic",
                    "severity": "medium",
                    "reason": "Content is not related to allowed educational topics (Physics, Chemistry, Biology)",
                    "details": "The question does not appear to be related to Physics, Chemistry, or Biology subjects."
                })
            
            if "banlist" in error_lower or "bannedlist" in error_lower or "banned_content" in error_lower:
                content = self._extract_detected_content(error_message, "banned")
                if content:
                    detected_content_list.append(f"banned_content: {content}")
                violations.append({
                    "category": "banned_content",
                    "severity": "high",
                    "reason": "The request contains material flagged as banned content",
                    "details": "This content is not permitted under the tutoring safety policies."
                })
            
            if not any(v.get("category") == "banned_content" for v in violations):
                banned_hits = [
                    word for word in self.banned_words
                    if re.search(rf"\b{re.escape(word)}\b", prompt, flags=re.IGNORECASE)
                ]
                if banned_hits:
                    detected_content_list.append(f"banned_content: {', '.join(banned_hits)}")
                    violations.append({
                        "category": "banned_content",
                        "severity": "high",
                        "reason": "The request contains material flagged as banned content",
                        "details": f"The following banned terms were detected: {', '.join(banned_hits)}"
                    })
            
            # If no specific violations were detected, add an unspecified violation
            if not violations:
                violations.append({
                    "category": "unspecified",
                    "severity": "medium",
                    "reason": "Content validation failed",
                    "details": error_message
                })
            
            logger.warning(f"Guardrails validation failed: {error_message}")
            
            result = {
                "allowed": False,
                "message": self._get_user_friendly_message(violations),
                "violations": violations,
                "validated_text": None
            }
            
            # Include all detected content if any was found
            if detected_content_list:
                result["detected_content"] = "; ".join(detected_content_list)
                
            return result
    
    def _get_user_friendly_message(self, violations: List[Dict]) -> str:
        """Convert technical violation messages to user-friendly messages."""
        if not violations:
            return "Content is not appropriate for educational use."
        
        # If multiple violations, prioritize by severity (high > medium)
        # Sort violations by severity (high first)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        sorted_violations = sorted(
            violations, 
            key=lambda v: severity_order.get(v.get("severity", "medium"), 1)
        )
        
        # Get primary category from most severe violation
        primary_category = sorted_violations[0].get("category", "unspecified")
        
        # If multiple violations, mention them
        if len(violations) > 1:
            categories = [v.get("category", "unspecified") for v in sorted_violations]
            unique_categories = list(dict.fromkeys(categories))  # Preserve order, remove duplicates
            
            messages = {
                "toxic": "Your message contains inappropriate language. Please rephrase your question.",
                "pii": "Please do not share personal information. Ask your question without including personal details.",
                "profanity": "Your message contains inappropriate language. Please use respectful language.",
                "off_topic": "Please ask a question related to Physics, Chemistry, or Biology.",
                "unspecified": "Your message could not be processed. Please rephrase your question."
            }
            
            primary_message = messages.get(primary_category, messages["unspecified"])
            
            # Add note about multiple issues if applicable
            if len(unique_categories) > 1:
                return f"{primary_message} (Multiple issues detected: {', '.join(unique_categories)})"
            
            return primary_message
        
        # Single violation - use standard message
        messages = {
            "toxic": "Your message contains inappropriate language. Please rephrase your question.",
            "pii": "Please do not share personal information. Ask your question without including personal details.",
            "profanity": "Your message contains inappropriate language. Please use respectful language.",
            "off_topic": "Please ask a question related to Physics, Chemistry, or Biology.",
            "unspecified": "Your message could not be processed. Please rephrase your question."
        }
        
        return messages.get(primary_category, messages["unspecified"])
    
    def _extract_detected_content(self, error_message: str, violation_type: str) -> Optional[str]:
        """
        Try to extract specific content that triggered the violation from error message.
        
        Args:
            error_message: The error message from Guardrails AI
            violation_type: Type of violation (toxic, pii, profanity, topic)
            
        Returns:
            Detected content string or None
        """
        # Try to extract quoted content or specific patterns
        
        # Look for quoted strings in error message
        quoted = re.findall(r'"([^"]+)"', error_message)
        if quoted:
            return ", ".join(quoted[:3])  # Return first 3 matches
        
        # Look for content after common patterns
        patterns = {
            "toxic": r"(?:contains|detected|found)[\s:]+([^\.]+)",
            "pii": r"(?:email|phone|ssn|credit)[\s:]+([^\.]+)",
            "profanity": r"(?:word|phrase)[\s:]+([^\.]+)",
            "topic": r"(?:topic|subject)[\s:]+([^\.]+)"
        }
        
        pattern = patterns.get(violation_type)
        if pattern:
            matches = re.findall(pattern, error_message, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return None
    
    def _extract_pii_types(self, error_message: str) -> List[str]:
        """
        Extract types of PII detected from error message.
        
        Args:
            error_message: The error message from Guardrails AI
            
        Returns:
            List of PII types detected
        """
        pii_types = []
        error_lower = error_message.lower()
        
        pii_keywords = {
            "email": "Email Address",
            "phone": "Phone Number",
            "ssn": "Social Security Number",
            "credit": "Credit Card",
            "ip": "IP Address",
            "date": "Date of Birth",
            "location": "Location"
        }
        
        for keyword, label in pii_keywords.items():
            if keyword in error_lower:
                pii_types.append(label)
        
        return pii_types
    
    def check_answer_seeking(self, prompt: str) -> Tuple[bool, str]:
        """
        Detect if student is directly asking for answers.
        
        Args:
            prompt: Student's input
            
        Returns:
            (is_answer_seeking, message)
        """
        prompt_lower = prompt.lower()
        for pattern in self.answer_seeking_patterns:
            if re.search(pattern, prompt_lower):
                return True, "I'm here to guide your learning, not provide direct answers. Let's work through this together!"
        
        return False, ""
    
    def apply_guardrails(self, prompt: str) -> Dict:
        """
        Apply all guardrails to student prompt.
        Multi-layered safety checks using Guardrails AI.
        
        Args:
            prompt: Student's input
            
        Returns:
            Dictionary with 'allowed', 'message', 'category', 'modified_prompt', 'violations'
        """
        # Layer 1: Safety checks using Guardrails AI validators
        safety_result = self.check_safety_with_guardrails(prompt, is_input=True)
        
        if not safety_result["allowed"]:
            # Get primary category from most severe violation, or default to "safety"
            primary_category = "safety"
            if safety_result.get("violations"):
                # Sort by severity to get most severe violation
                severity_order = {"high": 0, "medium": 1, "low": 2}
                sorted_violations = sorted(
                    safety_result["violations"],
                    key=lambda v: severity_order.get(v.get("severity", "medium"), 1)
                )
                primary_category = sorted_violations[0].get("category", "safety")
            
            return {
                "allowed": False,
                "message": safety_result["message"],
                "category": primary_category,
                "modified_prompt": None,
                "violations": safety_result.get("violations", [])
            }
        
        # Layer 2: Check for answer-seeking behavior (educational guardrail)
        is_answer_seeking, answer_message = self.check_answer_seeking(prompt)
        if is_answer_seeking:
            return {
                "allowed": True,  # Still allowed, but with modified approach
                "message": answer_message,
                "category": "answer_seeking",
                "modified_prompt": f"Guide the student without giving direct answer: {prompt}",
                "violations": []
            }
        
        logger.info(f"Prompt approved: {prompt[:50]}...")
        
        return {
            "allowed": True,
            "message": "Prompt approved",
            "category": "safe",
            "modified_prompt": prompt,
            "violations": []
        }
    
    def validate_response(self, response: str) -> Dict:
        """
        Validate model response using Guardrails AI and return violation information.
        Does not filter the response, only checks for violations.
        
        Args:
            response: Model's generated response
            
        Returns:
            Dictionary with 'is_safe', 'message', 'violations', 'detected_content'
        """
        try:
            # Validate response using output guard
            safety_result = self.check_safety_with_guardrails(response, is_input=False)
            
            if not safety_result["allowed"]:
                logger.warning(f"Response flagged: {safety_result['message']}")
                
                # Get primary category from most severe violation, or default to "unspecified"
                primary_category = "unspecified"
                if safety_result.get("violations"):
                    # Sort by severity to get most severe violation
                    severity_order = {"high": 0, "medium": 1, "low": 2}
                    sorted_violations = sorted(
                        safety_result["violations"],
                        key=lambda v: severity_order.get(v.get("severity", "medium"), 1)
                    )
                    primary_category = sorted_violations[0].get("category", "unspecified")
                
                return {
                    "is_safe": False,
                    "message": safety_result["message"],
                    "violations": safety_result.get("violations", []),
                    "detected_content": safety_result.get("detected_content"),
                    "category": primary_category
                }
            
            return {
                "is_safe": True,
                "message": "Response is safe",
                "violations": [],
                "detected_content": None,
                "category": "safe"
            }
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            # On error, assume safe (fail open for UX)
            return {
                "is_safe": True,
                "message": "Response validation error",
                "violations": [],
                "detected_content": None,
                "category": "validation_error"
            }
    
    def filter_response(self, response: str) -> str:
        """
        Validate model response using Guardrails AI.
        Legacy method - kept for backward compatibility.
        
        Args:
            response: Model's generated response
            
        Returns:
            Filtered response or error message
        """
        validation_result = self.validate_response(response)
        
        if not validation_result["is_safe"]:
            return "I apologize, but I cannot provide that information. Please ask an appropriate educational question."
        
        # Add educational reminder if response is very short
        if len(response) < 50:
            response += "\n\nRemember, I'm here to guide you through the problem. What's your next step?"
        
        return response