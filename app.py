# app.py
import streamlit as st
from src.models.frontier import FrontierModel
from src.models.finetuned import FineTunedModel
from src.guardrails import EducationalGuardrails
from src.evaluation import ModelEvaluator
import os

# Page config
st.set_page_config(
    page_title="LLM BuddyGuard - O-Level Tutor",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_frontier_model():
    """Load frontier model with caching - only when needed"""
    try:
        frontier = FrontierModel()
        return frontier, True
    except Exception as e:
        st.warning(f"Frontier model not loaded: {e}")
        return None, False

@st.cache_resource
def load_finetuned_model(subject):
    """Load fine-tuned model for specific subject with caching"""
    try:
        # Map UI subject names to model subject names
        subject_map = {
            "Physics": "physics",
            "Chemistry": "chemistry",
            "Biology": "biology"
        }
        model_subject = subject_map.get(subject, "physics")
        model = FineTunedModel(subject=model_subject)
        return model, True
    except Exception as e:
        st.warning(f"Fine-tuned {subject} model not loaded: {e}")
        return None, False

guardrails = EducationalGuardrails()
evaluator = ModelEvaluator()

# Sidebar
st.sidebar.title("Settings")

subject = st.sidebar.selectbox(
    "Subject:",
    ["Physics", "Chemistry", "Biology"]
)

# Set default model choice to Fine-tuned (Subject-specific)
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Fine-tuned (Subject-specific)"

# Get index for current selection
model_options = ["Fine-tuned (Subject-specific)", "Frontier (GPT-4o)", "Compare Both"]
current_index = model_options.index(st.session_state.model_choice) if st.session_state.model_choice in model_options else 0

model_choice = st.sidebar.radio(
    "Choose Model:",
    model_options,
    index=current_index
)

# Update session state
st.session_state.model_choice = model_choice

# Load models only when needed
frontier_model = None
frontier_ok = False
finetuned_model = None
finetuned_ok = False

# Load fine-tuned model if it's the default or selected
if model_choice in ["Fine-tuned (Subject-specific)", "Compare Both"]:
    finetuned_model, finetuned_ok = load_finetuned_model(subject)

# Load frontier model only if selected
if model_choice in ["Frontier (GPT-4o)", "Compare Both"]:
    frontier_model, frontier_ok = load_frontier_model()

# Safety guardrails toggle
if "safety_mode_enabled" not in st.session_state:
    st.session_state.safety_mode_enabled = True

enable_safety_mode = st.sidebar.checkbox(
    "Enable Safety Guardrails",
    value=st.session_state.safety_mode_enabled,
    help="Toggle safety checks for toxic language, PII, profanity, and topic restrictions"
)

# Update session state when toggle changes
st.session_state.safety_mode_enabled = enable_safety_mode

show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=False)

# Main interface
st.title("LLM BuddyGuard - O-Level Tutor")
st.markdown("Your AI study companion for Singapore O-Level examinations")

# Model status
col1, col2 = st.columns(2)
with col1:
    if model_choice in ["Fine-tuned (Subject-specific)", "Compare Both"]:
        status = "Ready" if finetuned_ok else "Not Loaded"
    else:
        status = "Not Selected"
    st.metric(f"Fine-tuned {subject} Model", status)
with col2:
    if model_choice in ["Frontier (GPT-4o)", "Compare Both"]:
        status = "Ready" if frontier_ok else "Not Loaded"
    else:
        status = "Not Selected"
    st.metric("Frontier Model", status)

st.divider()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "metrics" in message and show_metrics:
            with st.expander("Evaluation Metrics"):
                st.json(message["metrics"])

# Chat input
if prompt := st.chat_input("Ask your O-Level question..."):
    # Apply guardrails only if safety mode is enabled
    if enable_safety_mode:
        guardrail_result = guardrails.apply_guardrails(prompt)
        
        if not guardrail_result["allowed"]:
            # Show main error message
            st.error(guardrail_result["message"])
            
            # Show detailed violation information if available
            if guardrail_result.get("violations"):
                with st.expander("View Violation Details", expanded=True):
                    violations = guardrail_result["violations"]
                    
                    for i, violation in enumerate(violations, 1):
                        category = violation.get("category", "unknown")
                        severity = violation.get("severity", "unknown")
                        reason = violation.get("reason", "No reason provided")
                        details = violation.get("details", "")
                        
                        # Color code by severity
                        if severity == "high":
                            severity_icon = "[HIGH]"
                            severity_color = "red"
                        elif severity == "medium":
                            severity_icon = "[MEDIUM]"
                            severity_color = "orange"
                        else:
                            severity_icon = "[LOW]"
                            severity_color = "blue"
                        
                        st.markdown(f"""
                        **Violation {i}:**
                        - **Category:** `{category.replace('_', ' ').title()}`
                        - **Severity:** {severity_icon} `{severity.upper()}`
                        - **Reason:** {reason}
                        """)
                        
                        if details:
                            st.markdown(f"  - *{details}*")
                    
                    # Show category-specific guidance
                    category = violations[0].get("category", "") if violations else ""
                    if category == "toxic":
                        st.info("**Tip:** Please use respectful language when asking questions. Focus on the educational content.")
                    elif category == "pii":
                        st.info("**Tip:** Personal information like emails, phone numbers, or addresses is not needed for educational questions.")
                    elif category == "profanity":
                        st.info("**Tip:** Please rephrase your question using appropriate academic language.")
                    elif category == "off_topic":
                        st.info("**Tip:** This tutor specializes in Physics, Chemistry, and Biology. Please ask questions related to these subjects.")
                    
                    # Show what was detected (if available)
                    if guardrail_result.get("detected_content"):
                        st.markdown("**Detected Content:**")
                        st.code(guardrail_result["detected_content"], language="text")
        else:
            # Safety mode enabled and prompt approved, continue to model generation
            pass
    else:
        # Safety mode disabled - allow all prompts but show warning
        guardrail_result = {
            "allowed": True,
            "message": "Safety mode disabled",
            "category": "safety_disabled"
        }
        st.warning("Safety guardrails are currently disabled. Proceed with caution.")
    
    # Continue with model generation if allowed
    if guardrail_result.get("allowed", True):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show warning if answer-seeking detected
        if guardrail_result["message"] != "Prompt approved":
            st.warning(guardrail_result["message"])
        
        # Generate response
        with st.chat_message("assistant"):
            if model_choice == "Compare Both":
                if finetuned_ok and frontier_ok:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Fine-tuned {subject} Model**")
                        with st.spinner("Generating..."):
                            finetuned_result = finetuned_model.generate(prompt)
                            response_text = finetuned_result["response"]
                            
                            # Validate response if safety mode is enabled
                            if enable_safety_mode:
                                validation_result = guardrails.validate_response(response_text)
                                if not validation_result["is_safe"]:
                                    st.warning(f"Warning: {validation_result['message']}")
                                    if validation_result.get("violations"):
                                        with st.expander("View Response Violation Details"):
                                            for i, violation in enumerate(validation_result["violations"], 1):
                                                category = violation.get("category", "unknown")
                                                severity = violation.get("severity", "unknown")
                                                reason = violation.get("reason", "No reason provided")
                                                details = violation.get("details", "")
                                                
                                                severity_icon = "[HIGH]" if severity == "high" else "[MEDIUM]" if severity == "medium" else "[LOW]"
                                                
                                                st.markdown(f"""
                                                **Violation {i}:**
                                                - **Category:** `{category.replace('_', ' ').title()}`
                                                - **Severity:** {severity_icon} `{severity.upper()}`
                                                - **Reason:** {reason}
                                                """)
                                                
                                                if details:
                                                    st.markdown(f"  - *{details}*")
                                            
                                            if validation_result.get("detected_content"):
                                                st.markdown("**Detected Content:**")
                                                st.code(validation_result["detected_content"], language="text")
                            
                            st.markdown(response_text)
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(response_text)
                                with st.expander("Metrics"):
                                    st.json(metrics)
                    
                    with col2:
                        st.markdown("**Frontier Model (GPT-4o)**")
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for token in frontier_model.stream_generate(prompt, subject=subject):
                            full_response += token
                            response_placeholder.markdown(full_response)
                        
                        # Validate response if safety mode is enabled
                        if enable_safety_mode:
                            validation_result = guardrails.validate_response(full_response)
                            if not validation_result["is_safe"]:
                                st.warning(f"Warning: {validation_result['message']}")
                                if validation_result.get("violations"):
                                    with st.expander("View Response Violation Details"):
                                        for i, violation in enumerate(validation_result["violations"], 1):
                                            category = violation.get("category", "unknown")
                                            severity = violation.get("severity", "unknown")
                                            reason = violation.get("reason", "No reason provided")
                                            details = violation.get("details", "")
                                            
                                            severity_icon = "[HIGH]" if severity == "high" else "[MEDIUM]" if severity == "medium" else "[LOW]"
                                            
                                            st.markdown(f"""
                                            **Violation {i}:**
                                            - **Category:** `{category.replace('_', ' ').title()}`
                                            - **Severity:** {severity_icon} `{severity.upper()}`
                                            - **Reason:** {reason}
                                            """)
                                            
                                            if details:
                                                st.markdown(f"  - *{details}*")
                                        
                                        if validation_result.get("detected_content"):
                                            st.markdown("**Detected Content:**")
                                            st.code(validation_result["detected_content"], language="text")
                        
                        if show_metrics:
                            metrics = evaluator.evaluate_response(full_response)
                            with st.expander("Metrics"):
                                st.json(metrics)
                else:
                    st.error("Both models must be loaded for comparison mode")
                        
            elif model_choice == "Frontier (GPT-4o)" and frontier_ok:
                response_placeholder = st.empty()
                full_response = ""
                
                for token in frontier_model.stream_generate(prompt, subject=subject):
                    full_response += token
                    response_placeholder.markdown(full_response)
                
                # Validate response if safety mode is enabled
                if enable_safety_mode:
                    validation_result = guardrails.validate_response(full_response)
                    if not validation_result["is_safe"]:
                        st.warning(f"Warning: {validation_result['message']}")
                        if validation_result.get("violations"):
                            with st.expander("View Response Violation Details"):
                                for i, violation in enumerate(validation_result["violations"], 1):
                                    category = violation.get("category", "unknown")
                                    severity = violation.get("severity", "unknown")
                                    reason = violation.get("reason", "No reason provided")
                                    details = violation.get("details", "")
                                    
                                    severity_icon = "[HIGH]" if severity == "high" else "[MEDIUM]" if severity == "medium" else "[LOW]"
                                    
                                    st.markdown(f"""
                                    **Violation {i}:**
                                    - **Category:** `{category.replace('_', ' ').title()}`
                                    - **Severity:** {severity_icon} `{severity.upper()}`
                                    - **Reason:** {reason}
                                    """)
                                    
                                    if details:
                                        st.markdown(f"  - *{details}*")
                                
                                if validation_result.get("detected_content"):
                                    st.markdown("**Detected Content:**")
                                    st.code(validation_result["detected_content"], language="text")
                
                if show_metrics:
                    metrics = evaluator.evaluate_response(full_response)
                    with st.expander("Evaluation Metrics"):
                        st.json(metrics)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response,
                        "metrics": metrics
                    })
                else:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
            elif model_choice == "Fine-tuned (Subject-specific)" and finetuned_ok:
                with st.spinner("Thinking..."):
                    result = finetuned_model.generate(prompt)
                    response_text = result["response"]
                    
                    # Validate response if safety mode is enabled
                    if enable_safety_mode:
                        validation_result = guardrails.validate_response(response_text)
                        if not validation_result["is_safe"]:
                            st.warning(f"Warning: {validation_result['message']}")
                            if validation_result.get("violations"):
                                with st.expander("View Response Violation Details"):
                                    for i, violation in enumerate(validation_result["violations"], 1):
                                        category = violation.get("category", "unknown")
                                        severity = violation.get("severity", "unknown")
                                        reason = violation.get("reason", "No reason provided")
                                        details = violation.get("details", "")
                                        
                                        severity_icon = "[HIGH]" if severity == "high" else "[MEDIUM]" if severity == "medium" else "[LOW]"
                                        
                                        st.markdown(f"""
                                        **Violation {i}:**
                                        - **Category:** `{category.replace('_', ' ').title()}`
                                        - **Severity:** {severity_icon} `{severity.upper()}`
                                        - **Reason:** {reason}
                                        """)
                                        
                                        if details:
                                            st.markdown(f"  - *{details}*")
                                    
                                    if validation_result.get("detected_content"):
                                        st.markdown("**Detected Content:**")
                                        st.code(validation_result["detected_content"], language="text")
                    
                    st.markdown(response_text)
                    
                    if show_metrics:
                        metrics = evaluator.evaluate_response(response_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "metrics": metrics
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
            else:
                st.error("Selected model is not available")

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### How to Use
1. Choose your subject (Physics, Chemistry, or Biology)
2. Select a model (Fine-tuned or Frontier)
3. Ask your O-Level question
4. Get step-by-step guidance!

**Remember:** This tutor guides you through problems rather than giving direct answers.

**Fine-tuned Models:** Subject-specific models fine-tuned for Physics, Chemistry, and Biology.

**Compare Mode:** Compare responses from the fine-tuned model and GPT-4o side-by-side.
""")