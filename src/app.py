import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Define Request and Response Models
class FraudRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "URGENT: Your GTBank account has been restricted due to BVN update. Click http://gtb-update.info to verify immediately."})

class FraudResponse(BaseModel):
    text: str
    is_fraud: bool
    confidence_score: float
    risk_level: str
    recommendation: str

# Initialize FastAPI App
app = FastAPI(
    title="NaijaFinProtect API — Multilingual Financial Fraud & Sentiment Detection",
    description="An AI-powered NLP microservice that detects phishing, loan shark harassment, and financial fraud across African code-switched languages (Pidgin, Yoruba, Igbo, Hausa, English).",
    version="1.0.0"
)

# Load Trained NLP Pipeline
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/afri_fraud_model.joblib")
try:
    model_pipeline = joblib.load(MODEL_PATH)
    print(f"--> Successfully loaded production NLP model from {MODEL_PATH}")
except Exception as e:
    print(f"--> Error loading model: {e}. Ensure Day 2 script has been run.")
    model_pipeline = None

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "NaijaFinProtect Multilingual AI Microservice",
        "model_loaded": model_pipeline is not None,
        "supported_languages": ["English", "Nigerian Pidgin", "Yoruba", "Igbo", "Hausa"]
    }

@app.post("/predict_fraud", response_model=FraudResponse)
def predict_fraud(request: FraudRequest):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="NLP Model is not loaded on the server.")
    
    # Extract text and predict probability
    input_text = request.text
    probabilities = model_pipeline.predict_proba([input_text])[0]
    
    # Probability of class 1 (Fraud / Threat)
    fraud_prob = float(probabilities[1])
    is_fraud = fraud_prob >= 0.5
    
    # Assign Risk Level & Recommendation
    if fraud_prob >= 0.75:
        risk_level = "CRITICAL / HIGH RISK"
        recommendation = "DO NOT click any links, do not share OTP/PIN, and report number immediately."
    elif fraud_prob >= 0.5:
        risk_level = "SUSPICIOUS / MEDIUM RISK"
        recommendation = "Message exhibits common phishing patterns. Verify independently with your bank."
    else:
        risk_level = "SAFE / LOW RISK"
        recommendation = "Message appears to be a genuine banking notification or customer support communication."
        
    return FraudResponse(
        text=input_text,
        is_fraud=is_fraud,
        confidence_score=round(fraud_prob * 100, 2),
        risk_level=risk_level,
        recommendation=recommendation
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)