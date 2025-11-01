# app.py
import streamlit as st
from src.models.baseline import BaselineModel
from src.models.frontier import FrontierModel
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
def load_models():
    """Load models with caching"""
    try:
        baseline = BaselineModel(model_name="meta-llama/Llama-3.2-3B-Instruct")
        baseline_loaded = True
    except Exception as e:
        st.warning(f"Baseline model not loaded: {e}")
        baseline = None
        baseline_loaded = False
    
    try:
        frontier = FrontierModel()
        frontier_loaded = True
    except Exception as e:
        st.warning(f"Frontier model not loaded: {e}")
        frontier = None
        frontier_loaded = False
    
    return baseline, frontier, baseline_loaded, frontier_loaded

baseline_model, frontier_model, baseline_ok, frontier_ok = load_models()
guardrails = EducationalGuardrails()
evaluator = ModelEvaluator()

# Sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.radio(
    "Choose Model:",
    ["Baseline (Open-weight)", "Frontier (GPT-4o)", "Compare Both"],
    disabled=not (baseline_ok or frontier_ok)
)

subject = st.sidebar.selectbox(
    "Subject:",
    ["Mathematics", "Science", "English"]
)

show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=False)

# Main interface
st.title("LLM BuddyGuard - O-Level Tutor")
st.markdown("Your AI study companion for Singapore O-Level examinations")

# Model status
col1, col2 = st.columns(2)
with col1:
    status = "Ready" if baseline_ok else "Not Loaded"
    st.metric("Baseline Model", status)
with col2:
    status = "Ready" if frontier_ok else "Not Loaded"
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
    # Apply guardrails
    guardrail_result = guardrails.apply_guardrails(prompt)
    
    if not guardrail_result["allowed"]:
        st.error(guardrail_result["message"])
    else:
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
                if baseline_ok and frontier_ok:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Baseline Model**")
                        with st.spinner("Generating..."):
                            baseline_result = baseline_model.generate(prompt, subject=subject)
                            st.markdown(baseline_result["response"])
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(baseline_result["response"])
                                with st.expander("Metrics"):
                                    st.json(metrics)
                    
                    with col2:
                        st.markdown("**Frontier Model**")
                        with st.spinner("Generating..."):
                            frontier_result = frontier_model.generate(prompt, subject=subject)
                            st.markdown(frontier_result["response"])
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(frontier_result["response"])
                                with st.expander("Metrics"):
                                    st.json(metrics)
                else:
                    st.error("Both models must be loaded for comparison mode")
                    
            elif model_choice == "Baseline (Open-weight)" and baseline_ok:
                with st.spinner("Thinking..."):
                    result = baseline_model.generate(prompt, subject=subject)
                    st.markdown(result["response"])
                    
                    if show_metrics:
                        metrics = evaluator.evaluate_response(result["response"])
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"],
                            "metrics": metrics
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"]
                        })
                        
            elif model_choice == "Frontier (GPT-4o)" and frontier_ok:
                response_placeholder = st.empty()
                full_response = ""
                
                for token in frontier_model.stream_generate(prompt, subject=subject):
                    full_response += token
                    response_placeholder.markdown(full_response)
                
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
            else:
                st.error("Selected model is not available")

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### How to Use
1. Select a model (Baseline or Frontier)
2. Choose your subject
3. Ask your O-Level question
4. Get step-by-step guidance!

**Remember:** This tutor guides you through problems rather than giving direct answers.
""")