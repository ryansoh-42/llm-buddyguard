"""
Metrics API Service
FastAPI service that wraps metrics.py and returns all metrics in one JSON response.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.metrics import ResponseMetrics
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="LLM Metrics API",
    description="Evaluate model responses with ROUGE, Text F1, Exact Match, and Order Scoring",
    version="1.0.0"
)

# Initialize metrics service
metrics_service = ResponseMetrics()


# Request model
class EvaluationRequest(BaseModel):
    """Request body for evaluation endpoint"""
    generated: str = Field(..., description="Model-generated response", min_length=1)
    reference: Optional[str] = Field(None, description="Reference/correct answer")
    expected_keywords: Optional[List[str]] = Field(None, description="Expected keywords for coverage evaluation")
    is_mcq: bool = Field(False, description="Whether this is a multiple choice question")

    class Config:
        schema_extra = {
            "example": {
                "generated": "First multiply the terms, then factorize, then simplify",
                "reference": "First factorize, then multiply, then simplify",
                "expected_keywords": ["multiply", "factorize", "simplify"],
                "is_mcq": False
            }
        }


# Response model
class EvaluationResponse(BaseModel):
    """Response from evaluation endpoint"""
    status: str = Field(..., description="Status of the request (success/error)")
    metrics: Dict[str, Any] = Field(..., description="Dictionary of all computed metrics")
    message: Optional[str] = Field(None, description="Additional message or error details")


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "LLM Metrics API",
        "version": "1.0.0",
        "description": "Evaluate model responses with multiple metrics",
        "endpoints": {
            "POST /evaluate": "Evaluate a response and get all metrics",
            "GET /health": "Health check endpoint",
            "GET /metrics-info": "Get information about available metrics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "metrics-api",
        "version": "1.0.0"
    }


@app.get("/metrics-info")
async def metrics_info():
    """Get information about available metrics"""
    return {
        "available_metrics": {
            "rouge": {
                "description": "ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)",
                "requires": ["generated", "reference"],
                "output": "Precision, recall, F-measure for each variant"
            },
            "text_f1": {
                "description": "Text F1 score based on word overlap (always computed when reference provided)",
                "requires": ["generated", "reference"],
                "output": "F1, precision, recall, matched/missing/extra words"
            },
            "keyword_recall": {
                "description": "Keyword coverage using simple recall metric",
                "requires": ["generated", "expected_keywords"],
                "output": "Recall score (0-1), matched/missing keywords"
            },
            "exact_match": {
                "description": "Exact match accuracy for MCQ questions",
                "requires": ["generated", "reference", "is_mcq=true"],
                "output": "Boolean match, extracted answer, accuracy"
            },
            "order": {
                "description": "Reasoning step order correctness (dynamic extraction, always computed)",
                "requires": ["generated", "reference"],
                "output": "Order score, concept sequences, edit distance"
            },
            "basic": {
                "description": "Basic text statistics",
                "requires": ["generated"],
                "output": "Response length, word count"
            }
        }
    }


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EvaluationRequest):
    """
    Evaluate a model response with all available metrics.

    Returns all applicable metrics based on provided inputs:
    - Always: response_length, word_count
    - If reference provided: rouge scores, text_f1, order score
    - If expected_keywords provided: keyword_recall
    - If is_mcq=true and reference provided: exact_match
    """
    try:
        # Compute all metrics
        metrics = metrics_service.compute_all_metrics(
            generated=request.generated,
            reference=request.reference,
            expected_keywords=request.expected_keywords,
            is_mcq=request.is_mcq
        )

        return EvaluationResponse(
            status="success",
            metrics=metrics,
            message=None
        )

    except Exception as e:
        # Log error and return error response
        raise HTTPException(
            status_code=500,
            detail=f"Error computing metrics: {str(e)}"
        )


@app.post("/evaluate/batch")
async def evaluate_batch(requests: List[EvaluationRequest]):
    """
    Evaluate multiple responses in batch.

    Returns a list of evaluation results.
    """
    try:
        results = []

        for idx, request in enumerate(requests):
            try:
                metrics = metrics_service.compute_all_metrics(
                    generated=request.generated,
                    reference=request.reference,
                    expected_keywords=request.expected_keywords,
                    is_mcq=request.is_mcq
                )

                results.append({
                    "index": idx,
                    "status": "success",
                    "metrics": metrics
                })

            except Exception as e:
                results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(e)
                })

        return {
            "status": "success",
            "total": len(requests),
            "results": results
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error in batch evaluation: {str(e)}"
        )


# Custom endpoint for quick ROUGE-only evaluation
@app.post("/evaluate/rouge")
async def evaluate_rouge_only(generated: str, reference: str):
    """Quick endpoint for ROUGE scores only"""
    try:
        rouge_scores = metrics_service.compute_rouge(generated, reference)
        return {
            "status": "success",
            "rouge": rouge_scores
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Custom endpoint for quick order scoring
@app.post("/evaluate/order")
async def evaluate_order_only(generated: str, reference: str, use_spacy: bool = True):
    """Quick endpoint for order scoring only"""
    try:
        order_result = metrics_service.compute_order_score(generated, reference, use_spacy)
        return {
            "status": "success",
            "order": order_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
