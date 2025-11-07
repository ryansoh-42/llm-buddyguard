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
    
    # Subject-specific model mapping
    subject_models = {}
    
    # Comment out Physics model for now to save memory and test Chemistry
    # try:
    #     physics_model = BaselineModel(model_name="Fawl/is469_project_physics")
    #     subject_models["Physics"] = physics_model
    #     print("✅ Physics model (local) ready")
    # except Exception as e:
    #     print(f"❌ Physics model failed: {e}")
    #     subject_models["Physics"] = None
    subject_models["Physics"] = None
    print("ℹ️ Physics model disabled for testing")

    # Load only Chemistry model for testing
    try:
        chemistry_model = BaselineModel(model_name="Fawl/is469_project_chem")
        subject_models["Chemistry"] = chemistry_model
        print("✅ Chemistry model (local) ready")
    except Exception as e:
        print(f"❌ Chemistry model failed: {e}")
        subject_models["Chemistry"] = None

    # try:
    #     biology_model = BaselineModel(model_name="Fawl/is469_project_bio")
    #     subject_models["Biology"] = biology_model
    #     print("✅ Biology model (local) ready")
    # except Exception as e:
    #     print(f"❌ Biology model failed: {e}")
    #     subject_models["Biology"] = None
    subject_models["Biology"] = None
    print("ℹ️ Biology model disabled for testing")

    # Skip baseline model to save memory - only using fine-tuned subject models
    # Uncomment below if you need a general baseline model later:
    # try:
    #     baseline = BaselineModel(model_name="meta-llama/Llama-3.2-1B-Instruct")
    #     baseline_loaded = True
    #     print("✅ General baseline model ready")
    # except Exception as e:
    #     st.warning(f"Baseline model not loaded: {e}")
    #     baseline = None
    #     baseline_loaded = False
    
    baseline = None
    baseline_loaded = False
    print("ℹ️ Baseline model commented out to conserve memory")
    
    try:
        frontier = FrontierModel()
        frontier_loaded = True
    except Exception as e:
        st.warning(f"Frontier model not loaded: {e}")
        frontier = None
        frontier_loaded = False
    
    # Check which subject models are available
    available_subjects = [subj for subj, model in subject_models.items() if model is not None]
    subject_loaded = len(available_subjects) > 0
    
    return baseline, frontier, subject_models, baseline_loaded, frontier_loaded, subject_loaded, available_subjects

baseline_model, frontier_model, subject_models, baseline_ok, frontier_ok, subject_ok, available_subjects = load_models()
guardrails = EducationalGuardrails()
evaluator = ModelEvaluator()

# Sidebar
st.sidebar.title("Settings")

# Show available subjects and model types
st.sidebar.markdown("### Available Models")
for subj in ["Physics", "Chemistry", "Biology"]:
    if subj in available_subjects and subject_models.get(subj) != baseline_model:
        st.sidebar.success(f"✅ {subj}: Fine-tuned model")
    elif subj in available_subjects:
        st.sidebar.info(f"ℹ️ {subj}: Base model")
    else:
        st.sidebar.error(f"❌ {subj}: Not available")

model_choice = st.sidebar.radio(
    "Choose Model Type:",
    ["Subject-Specific (Recommended)", "General Baseline", "Frontier (GPT-4o)", "Compare Models"],
    disabled=not (baseline_ok or frontier_ok or subject_ok)
)

subject = st.sidebar.selectbox(
    "Subject:",
    ["Physics", "Chemistry", "Biology"],
    help="Select your subject. All subjects have fine-tuned models available."
)

show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=False)

# Main interface
st.title("LLM BuddyGuard - O-Level Tutor")
st.markdown("Your AI study companion for Singapore O-Level examinations")

# Model status
col1, col2, col3 = st.columns(3)
with col1:
    subject_model_status = "Ready" if subject in available_subjects else "Not Available"
    model_type = "Fine-tuned" if (subject in ["Physics", "Chemistry", "Biology"] and 
                                  subject_models.get(subject) != baseline_model) else "Base"
    st.metric(f"{subject} Model", subject_model_status, delta=model_type)
with col2:
    status = "Ready" if baseline_ok else "Not Loaded"
    st.metric("General Baseline", status)
with col3:
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
            
            # Helper function to get the appropriate model for the subject
            def get_subject_model():
                if subject in subject_models and subject_models[subject] is not None:
                    return subject_models[subject]
                elif baseline_ok:
                    return baseline_model
                else:
                    return None
            
            if model_choice == "Subject-Specific (Recommended)":
                # Use subject-specific model (fine-tuned if available, baseline if not)
                model_to_use = get_subject_model()
                if model_to_use:
                    model_type = "fine-tuned" if (subject in ["Physics", "Chemistry", "Biology"] and 
                                                subject_models.get(subject) != baseline_model) else "base"
                    
                    with st.spinner(f"Using {model_type} model for {subject}..."):
                        result = model_to_use.generate(prompt, subject=subject, max_new_tokens=128, temperature=0.3)
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
                else:
                    st.error(f"No model available for {subject}")
            
            elif model_choice == "General Baseline" and baseline_ok:
                with st.spinner("Using general baseline model..."):
                    result = baseline_model.generate(prompt, subject=subject, max_new_tokens=256, temperature=0.7)
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
                    
            elif model_choice == "Compare Models":
                # Compare subject-specific vs frontier
                subject_model_to_use = get_subject_model()
                
                if subject_model_to_use and frontier_ok:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        model_type = "Fine-tuned" if (subject in ["Physics", "Chemistry", "Biology"] and 
                                                    subject_models.get(subject) != baseline_model) else "Base"
                        st.markdown(f"**{model_type} {subject} Model**")
                        with st.spinner("Generating..."):
                            subject_result = subject_model_to_use.generate(prompt, subject=subject)
                            st.markdown(subject_result["response"])
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(subject_result["response"])
                                with st.expander("Metrics"):
                                    st.json(metrics)
                    
                    with col2:
                        st.markdown("**Frontier Model (GPT-4o)**")
                        with st.spinner("Generating..."):
                            frontier_result = frontier_model.generate(prompt, subject=subject)
                            st.markdown(frontier_result["response"])
                            
                            if show_metrics:
                                metrics = evaluator.evaluate_response(frontier_result["response"])
                                with st.expander("Metrics"):
                                    st.json(metrics)
                elif subject_model_to_use and baseline_ok:
                    # Compare subject-specific vs general baseline
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        model_type = "Fine-tuned" if (subject in ["Physics", "Chemistry", "Biology"] and 
                                                    subject_models.get(subject) != baseline_model) else "Subject-Optimized"
                        st.markdown(f"**{model_type} Model**")
                        with st.spinner("Generating..."):
                            subject_result = subject_model_to_use.generate(prompt, subject=subject)
                            st.markdown(subject_result["response"])
                    
                    with col2:
                        st.markdown("**General Baseline Model**")
                        with st.spinner("Generating..."):
                            baseline_result = baseline_model.generate(prompt, subject=subject)
                            st.markdown(baseline_result["response"])
                else:
                    st.error("Not enough models loaded for comparison")
            else:
                st.error("Selected model is not available")

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
### How to Use
1. **Choose your subject** (Physics, Chemistry, Biology, etc.)
2. **Select model type**:
   - **Subject-Specific**: Fine-tuned models for Physics/Chemistry/Biology
   - **General Baseline**: General purpose model
   - **Frontier**: GPT-4o for comparison
3. **Ask your O-Level question**
4. **Get step-by-step guidance!**

### Model Info
- **🔬 Physics/Chemistry/Biology**: Fine-tuned on Singapore O-Level content

**Remember:** This tutor guides you through problems rather than giving direct answers.
""")