# app.py
import streamlit as st
from src.models.finetune import FineTunedModel
from src.models.frontier import FrontierModel
from src.guardrails import EducationalGuardrails
from src.evaluation import ModelEvaluator
from src.metrics import ResponseMetrics
import requests
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
    #     physics_model = FineTunedModel(model_name="Fawl/is469_project_physics")
    #     subject_models["Physics"] = physics_model
    #     print("✅ Physics model (local) ready")
    # except Exception as e:
    #     print(f"❌ Physics model failed: {e}")
    #     subject_models["Physics"] = None
    subject_models["Physics"] = None
    print("ℹ️ Physics model disabled for testing")

    # Load only Chemistry model for testing
    try:
        chemistry_model = FineTunedModel(model_name="Fawl/is469_project_chem")
        subject_models["Chemistry"] = chemistry_model
        print("✅ Chemistry model (local) ready")
    except Exception as e:
        print(f"❌ Chemistry model failed: {e}")
        subject_models["Chemistry"] = None

    # try:
    #     biology_model = FineTunedModel(model_name="Fawl/is469_project_bio")
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
    #     baseline = FineTunedModel(model_name="meta-llama/Llama-3.2-1B-Instruct")
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
api_metrics = ResponseMetrics()  # For comprehensive API-style metrics

def display_dual_metrics(response, reference_answer=None, expected_keywords=None):
    """Display both educational and API metrics side by side."""
    
    # Get both metric types
    educational_metrics = evaluator.evaluate_response(response)
    api_metrics_result = api_metrics.compute_all_metrics(
        generated=response,
        reference=reference_answer,
        expected_keywords=expected_keywords
    )
    
    # Display in tabs
    tab1, tab2 = st.tabs(["📚 Educational Quality", "🔬 Technical Analysis"])
    
    with tab1:
        st.markdown("**Educational Metrics** - Teaching effectiveness")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("No Direct Answer", f"{educational_metrics.get('no_direct_answer', 0):.1f}")
            st.metric("Step-by-Step", f"{educational_metrics.get('step_by_step_score', 0):.2f}")
        
        with col2:
            st.metric("Tone Quality", f"{educational_metrics.get('tone_score', 0):.2f}")
            st.metric("Overall Score", f"{educational_metrics.get('overall_score', 0):.2f}")
        
        with col3:
            st.metric("Response Length", educational_metrics.get('response_length', 0))
            st.metric("Word Count", educational_metrics.get('word_count', 0))
    
    with tab2:
        st.markdown("**API Metrics** - Content accuracy & reasoning")
        
        if reference_answer:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Similarity (ROUGE)**")
                rouge = api_metrics_result.get('rouge', {})
                for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
                    if rouge_type in rouge:
                        score = rouge[rouge_type]['fmeasure']
                        st.metric(rouge_type.upper(), f"{score:.3f}")
            
            with col2:
                st.markdown("**Reasoning Order**")
                order = api_metrics_result.get('order', {})
                if order:
                    st.metric("Order Score", f"{order.get('order_score', 0):.3f}")
                    st.text(f"Status: {order.get('message', 'N/A')}")
        
        if expected_keywords:
            st.markdown("**Keyword Coverage**")
            keyword_recall = api_metrics_result.get('keyword_recall', {})
            if keyword_recall:
                st.metric("Keyword Recall", f"{keyword_recall.get('recall', 0):.3f}")
                st.write(f"✅ Found: {keyword_recall.get('matched_keywords', [])}")
                st.write(f"❌ Missing: {keyword_recall.get('missing_keywords', [])}")
        
        # Basic stats
        st.markdown("**Response Statistics**")
        st.metric("Response Length", api_metrics_result.get('response_length', 0))
        st.metric("Word Count", api_metrics_result.get('word_count', 0))

def get_api_metrics(response, reference_answer=None, expected_keywords=None):
    """Get metrics from the backend API server."""
    try:
        payload = {
            "generated": response,
            "reference": reference_answer,
            "expected_keywords": expected_keywords or []
        }
        
        response_api = requests.post("http://localhost:8000/evaluate", json=payload, timeout=10)
        if response_api.status_code == 200:
            return response_api.json()
        else:
            st.error(f"API Error: {response_api.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.warning(f"API not available: {e}")
        return None

def display_triple_metrics(response, reference_answer=None, expected_keywords=None):
    """Display educational, direct API, AND backend API metrics."""
    
    # Get all three metric types
    educational_metrics = evaluator.evaluate_response(response)
    direct_api_metrics = api_metrics.compute_all_metrics(
        generated=response,
        reference=reference_answer,
        expected_keywords=expected_keywords
    )
    backend_api_result = get_api_metrics(response, reference_answer, expected_keywords)
    
    # Display in tabs
    tab1, tab2, tab3 = st.tabs(["📚 Educational", "🔬 Direct API", "🌐 Backend API"])
    
    with tab1:
        st.markdown("**Educational Metrics** - Teaching effectiveness")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("No Direct Answer", f"{educational_metrics.get('no_direct_answer', 0):.1f}")
            st.metric("Step-by-Step", f"{educational_metrics.get('step_by_step_score', 0):.2f}")
        
        with col2:
            st.metric("Tone Quality", f"{educational_metrics.get('tone_score', 0):.2f}")
            st.metric("Overall Score", f"{educational_metrics.get('overall_score', 0):.2f}")
    
    with tab2:
        st.markdown("**Direct API Metrics** - Content accuracy (embedded)")
        
        if expected_keywords:
            keyword_recall = direct_api_metrics.get('keyword_recall', {})
            if keyword_recall:
                st.metric("Keyword Recall", f"{keyword_recall.get('recall', 0):.3f}")
                st.write(f"✅ Found: {keyword_recall.get('matched_keywords', [])}")
        
        if reference_answer:
            order = direct_api_metrics.get('order', {})
            if order:
                st.metric("Order Score", f"{order.get('order_score', 0):.3f}")
                st.text(f"Status: {order.get('message', 'N/A')}")
    
    with tab3:
        st.markdown("**Backend API Metrics** - Independent validation")
        
        if backend_api_result:
            api_metrics_data = backend_api_result.get('metrics', {})
            
            if expected_keywords:
                keyword_recall = api_metrics_data.get('keyword_recall', {})
                if keyword_recall:
                    st.metric("Keyword Recall", f"{keyword_recall.get('recall', 0):.3f}")
                    st.write(f"✅ Found: {keyword_recall.get('matched_keywords', [])}")
            
            if reference_answer:
                order = api_metrics_data.get('order', {})
                if order:
                    st.metric("Order Score", f"{order.get('order_score', 0):.3f}")
                    st.text(f"Status: {order.get('message', 'N/A')}")
            
            # Show API response info
            st.info(f"API Status: {backend_api_result.get('status', 'unknown')}")
            st.text(f"Message: {backend_api_result.get('message', 'N/A')}")
        else:
            st.error("Backend API not available")

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

# Safety guardrails toggle
if "safety_mode_enabled" not in st.session_state:
    st.session_state.safety_mode_enabled = True

enable_safety_mode = st.sidebar.checkbox(
    "Enable Safety Guardrails",
    value=st.session_state.safety_mode_enabled,
    help="Toggle safety checks for inappropriate content and off-topic questions"
)

# Update session state when toggle changes
st.session_state.safety_mode_enabled = enable_safety_mode

show_metrics = st.sidebar.checkbox("Show Evaluation Metrics", value=False)

# Main interface
st.title("LLM BuddyGuard - O-Level Tutor")
st.markdown("Your AI study companion for Singapore O-Level examinations")

# Safety status indicator
if enable_safety_mode:
    st.success("🛡️ Safety guardrails are enabled - Content filtering active")
else:
    st.warning("🔓 Safety guardrails are disabled - All content allowed")

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
    # Apply guardrails only if enabled
    if enable_safety_mode:
        guardrail_result = guardrails.apply_guardrails(prompt)
    else:
        # Bypass guardrails when disabled
        guardrail_result = {
            "allowed": True,
            "message": "Safety guardrails disabled",
            "modified_prompt": prompt
        }
        st.info("🔓 Safety guardrails are currently disabled")
    
    if not guardrail_result["allowed"]:
        st.error(guardrail_result["message"])
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show warning if answer-seeking detected (only when guardrails are enabled)
        if enable_safety_mode and guardrail_result["message"] not in ["Prompt approved", "Safety guardrails disabled"]:
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
                        result = model_to_use.generate(prompt, subject=subject, max_new_tokens=200, temperature=0.3)
                        st.markdown(result["response"])
                        
                        if show_metrics:
                            # Define chemistry-specific keywords for API metrics
                            chemistry_keywords = ["atoms", "molecules", "electrons", "bonds", "equation", "balance", "reaction", "chemical", "formula"]
                            
                            # Call backend API silently in background (no UI display)
                            backend_result = get_api_metrics(
                                response=result["response"],
                                reference_answer=None,
                                expected_keywords=chemistry_keywords if subject.lower() == "chemistry" else None
                            )
                            
                            # Log API result for debugging (not shown to user)
                            if backend_result:
                                print(f"✅ Backend API called successfully: {backend_result['status']}")
                            else:
                                print("❌ Backend API failed")
                            
                            # Display the original 2-tab metrics UI
                            with st.expander("📊 Complete Evaluation Metrics"):
                                display_dual_metrics(
                                    response=result["response"],
                                    reference_answer=None,
                                    expected_keywords=chemistry_keywords if subject.lower() == "chemistry" else None
                                )
                            
                            # Store basic metrics for session
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
                    # Define some sample expected keywords for chemistry
                    chemistry_keywords = ["atoms", "molecules", "electrons", "bonds", "equation", "balance", "reaction"]
                    
                    # Display dual metrics with enhanced info
                    with st.expander("📊 Complete Evaluation Metrics"):
                        display_dual_metrics(
                            response=full_response,
                            reference_answer=None,  # Could be enhanced with reference answers
                            expected_keywords=chemistry_keywords if subject.lower() == "chemistry" else None
                        )
                    
                    # Store basic metrics for session
                    metrics = evaluator.evaluate_response(full_response)
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
3. **Toggle safety guardrails** (on/off)
4. **Ask your O-Level question**
5. **Get step-by-step guidance!**

### Safety Features
- **🛡️ Guardrails ON**: Blocks inappropriate content and off-topic questions
- **🔓 Guardrails OFF**: Allows all questions (use with caution)

### Model Info
- **🔬 Physics/Chemistry/Biology**: Fine-tuned on Singapore O-Level content

**Remember:** This tutor guides you through problems rather than giving direct answers.
""")