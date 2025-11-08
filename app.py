# app.py
import streamlit as st
from typing import Dict, List
import re
from src.models.frontier import FrontierModel
from src.models.finetuned import FineTunedModel
from src.guardrails import EducationalGuardrails
from src.evaluation import ModelEvaluator

# Page config
st.set_page_config(
    page_title="LLM BuddyGuard - O-Level Tutor",
    layout="wide"
)


def render_guardrail_notifications(
    guardrail_message: str | None,
    violations: List[Dict] | None,
    detected_content: str | None
) -> None:
    """Display guardrail warnings and detailed violations inline in the chat."""

    if guardrail_message and guardrail_message not in ("Prompt approved", "Safety mode disabled"):
        st.warning(guardrail_message)

    if not violations:
        return

    severity_to_display = {
        "high": st.error,
        "medium": st.warning,
        "low": st.info
    }

    for idx, violation in enumerate(violations, 1):
        severity = violation.get("severity", "medium").lower()
        display_fn = severity_to_display.get(severity, st.warning)
        category = violation.get("category", "unknown").replace("_", " ").title()
        reason = violation.get("reason", "No reason provided")
        display_fn(f"Violation {idx}: [{severity.upper()}] {category} — {reason}")

        details = violation.get("details")
        if details:
            st.markdown(f"*{details}*")

    if detected_content:
        st.markdown("**Detected Content:**")
        st.code(detected_content, language="text")

    tips = {
        "toxic": "Please use respectful language when asking questions. Focus on the educational content.",
        "pii": "Personal information like emails, phone numbers, or addresses is not needed for educational questions.",
        "profanity": "Please rephrase your question using appropriate academic language.",
        "off_topic": "This tutor specializes in Physics, Chemistry, and Biology. Please ask questions related to these subjects."
    }

    primary_category = violations[0].get("category") if violations else None
    if primary_category in tips:
        st.info(f"**Tip:** {tips[primary_category]}")


def sanitize_response(text: str, violations: List[Dict] | None) -> str:
    """Redact sensitive details from model responses when violations are detected."""
    if not violations:
        return text

    sanitized_text = text
    categories = {violation.get("category") for violation in violations}

    if "pii" in categories:
        # Mask sequences that resemble credit card numbers or other long digit strings
        sanitized_text = re.sub(r"(?:(?<!\d)(?:\d[ -]?){12,19}(?!\d))", "[REDACTED NUMBER]", sanitized_text)
        # Mask email-like patterns
        sanitized_text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[REDACTED EMAIL]", sanitized_text)

    return sanitized_text

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
        render_guardrail_notifications(
            message.get("guardrail_message"),
            message.get("violations"),
            message.get("detected_content")
        )
        if "metrics" in message and show_metrics:
            with st.expander("Evaluation Metrics"):
                st.json(message["metrics"])

# Chat input
if prompt := st.chat_input("Ask your O-Level question..."):
    if enable_safety_mode:
        guardrail_result = guardrails.apply_guardrails(prompt)
    else:
        guardrail_result = {
            "allowed": True,
            "message": "Safety mode disabled",
            "category": "safety_disabled"
        }
        st.warning("Safety guardrails are currently disabled. Proceed with caution.")

    guardrail_message = guardrail_result.get("message") if guardrail_result else None
    guardrail_category = guardrail_result.get("category") if guardrail_result else None
    guardrail_violations = guardrail_result.get("violations") if guardrail_result else None
    detected_content = guardrail_result.get("detected_content") if guardrail_result else None

    user_message: Dict = {"role": "user", "content": prompt}

    if guardrail_category != "safety_disabled":
        if guardrail_message and guardrail_message != "Prompt approved":
            user_message["guardrail_message"] = guardrail_message
        if guardrail_violations:
            user_message["violations"] = guardrail_violations
        if detected_content:
            user_message["detected_content"] = detected_content

    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)
        render_guardrail_notifications(
            user_message.get("guardrail_message"),
            user_message.get("violations"),
            user_message.get("detected_content")
        )

    prompt_allowed = guardrail_result.get("allowed", True)

    if prompt_allowed:
        # Generate response
        with st.chat_message("assistant"):
            if model_choice == "Compare Both":
                if finetuned_ok and frontier_ok:
                    col1, col2 = st.columns(2)

                    finetuned_display_text: str | None = None
                    finetuned_guardrail_info: Dict = {}
                    finetuned_metrics: Dict | None = None
                    with col1:
                        st.markdown(f"**Fine-tuned {subject} Model**")
                        with st.spinner("Generating..."):
                            finetuned_result = finetuned_model.generate(prompt)
                            response_text = finetuned_result["response"]

                            display_text = response_text

                            if enable_safety_mode:
                                validation_result = guardrails.validate_response(response_text)
                                if not validation_result["is_safe"]:
                                    st.warning("⚠️ Guardrails detected issues with this response. Sensitive details may be redacted.")
                                    render_guardrail_notifications(
                                        validation_result.get("message"),
                                        validation_result.get("violations"),
                                        validation_result.get("detected_content")
                                    )
                                    display_text = sanitize_response(response_text, validation_result.get("violations"))
                                    finetuned_guardrail_info["guardrail_message"] = validation_result.get("message")
                                    if validation_result.get("violations"):
                                        finetuned_guardrail_info["violations"] = validation_result["violations"]
                                    if validation_result.get("detected_content"):
                                        finetuned_guardrail_info["detected_content"] = validation_result["detected_content"]

                            st.markdown(display_text)
                            finetuned_display_text = display_text
                            
                            if show_metrics:
                                finetuned_metrics = evaluator.evaluate_response(display_text)
                                with st.expander("Metrics"):
                                    st.json(finetuned_metrics)

                    frontier_display_text: str | None = None
                    frontier_guardrail_info: Dict = {}
                    frontier_metrics: Dict | None = None
                    with col2:
                        st.markdown("**Frontier Model (GPT-4o)**")
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        for token in frontier_model.stream_generate(prompt, subject=subject):
                            full_response += token
                            response_placeholder.markdown(full_response)
                        
                        display_text = full_response

                        if enable_safety_mode:
                            validation_result = guardrails.validate_response(full_response)
                            if not validation_result["is_safe"]:
                                st.warning("⚠️ Guardrails detected issues with this response. Sensitive details may be redacted.")
                                render_guardrail_notifications(
                                    validation_result.get("message"),
                                    validation_result.get("violations"),
                                    validation_result.get("detected_content")
                                )
                                display_text = sanitize_response(full_response, validation_result.get("violations"))
                                frontier_guardrail_info["guardrail_message"] = validation_result.get("message")
                                if validation_result.get("violations"):
                                    frontier_guardrail_info["violations"] = validation_result["violations"]
                                if validation_result.get("detected_content"):
                                    frontier_guardrail_info["detected_content"] = validation_result["detected_content"]

                        response_placeholder.markdown(display_text)
                        frontier_display_text = display_text
                        
                        if show_metrics:
                            frontier_metrics = evaluator.evaluate_response(display_text)
                            with st.expander("Metrics"):
                                st.json(frontier_metrics)

                    if finetuned_display_text is not None:
                        assistant_message: Dict = {
                            "role": "assistant",
                            "content": f"**Fine-tuned {subject} Model**\n\n{finetuned_display_text}"
                        }
                        if finetuned_guardrail_info:
                            assistant_message.update(finetuned_guardrail_info)
                        if show_metrics and finetuned_metrics is not None:
                            assistant_message["metrics"] = finetuned_metrics
                        st.session_state.messages.append(assistant_message)

                    if frontier_display_text is not None:
                        assistant_message: Dict = {
                            "role": "assistant",
                            "content": "**Frontier Model (GPT-4o)**\n\n" + frontier_display_text
                        }
                        if frontier_guardrail_info:
                            assistant_message.update(frontier_guardrail_info)
                        if show_metrics and frontier_metrics is not None:
                            assistant_message["metrics"] = frontier_metrics
                        st.session_state.messages.append(assistant_message)
                else:
                    st.error("Both models must be loaded for comparison mode")
                        
            elif model_choice == "Frontier (GPT-4o)" and frontier_ok:
                response_placeholder = st.empty()
                full_response = ""
                
                for token in frontier_model.stream_generate(prompt, subject=subject):
                    full_response += token
                    response_placeholder.markdown(full_response)
                
                display_text = full_response
                assistant_guardrail_info: Dict = {}

                if enable_safety_mode:
                    validation_result = guardrails.validate_response(full_response)
                    if not validation_result["is_safe"]:
                        st.warning("⚠️ Guardrails detected issues with this response. Sensitive details may be redacted.")
                        render_guardrail_notifications(
                            validation_result.get("message"),
                            validation_result.get("violations"),
                            validation_result.get("detected_content")
                        )
                        display_text = sanitize_response(full_response, validation_result.get("violations"))
                        assistant_guardrail_info["guardrail_message"] = validation_result.get("message")
                        if validation_result.get("violations"):
                            assistant_guardrail_info["violations"] = validation_result["violations"]
                        if validation_result.get("detected_content"):
                            assistant_guardrail_info["detected_content"] = validation_result["detected_content"]

                response_placeholder.markdown(display_text)

                assistant_message: Dict = {
                    "role": "assistant",
                    "content": display_text
                }

                if assistant_guardrail_info:
                    assistant_message.update(assistant_guardrail_info)

                if show_metrics:
                    metrics = evaluator.evaluate_response(display_text)
                    with st.expander("Evaluation Metrics"):
                        st.json(metrics)
                    assistant_message["metrics"] = metrics

                st.session_state.messages.append(assistant_message)
            elif model_choice == "Fine-tuned (Subject-specific)" and finetuned_ok:
                with st.spinner("Thinking..."):
                    result = finetuned_model.generate(prompt)
                    response_text = result["response"]

                    display_text = response_text
                    assistant_guardrail_info: Dict = {}

                    if enable_safety_mode:
                        validation_result = guardrails.validate_response(response_text)
                        if not validation_result["is_safe"]:
                            st.warning("⚠️ Guardrails detected issues with this response. Sensitive details may be redacted.")
                            render_guardrail_notifications(
                                validation_result.get("message"),
                                validation_result.get("violations"),
                                validation_result.get("detected_content")
                            )
                            display_text = sanitize_response(response_text, validation_result.get("violations"))
                            assistant_guardrail_info["guardrail_message"] = validation_result.get("message")
                            if validation_result.get("violations"):
                                assistant_guardrail_info["violations"] = validation_result["violations"]
                            if validation_result.get("detected_content"):
                                assistant_guardrail_info["detected_content"] = validation_result["detected_content"]

                    st.markdown(display_text)

                    assistant_message: Dict = {
                        "role": "assistant",
                        "content": display_text
                    }

                    if assistant_guardrail_info:
                        assistant_message.update(assistant_guardrail_info)

                    if show_metrics:
                        metrics = evaluator.evaluate_response(display_text)
                        assistant_message["metrics"] = metrics

                    st.session_state.messages.append(assistant_message)
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