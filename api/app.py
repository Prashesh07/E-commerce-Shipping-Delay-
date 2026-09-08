from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
try:
    from api.predict import ShipmentInput, predict_delay, MODEL_VERSION, pipeline
except ImportError:
    from predict import ShipmentInput, predict_delay, MODEL_VERSION, pipeline

app = FastAPI()


@app.get("/")
def root():
    return {"status": "ok", "model": "xgboost_shipping_delay_model"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
        "model_loaded": True if pipeline else False,
    }


@app.post("/predict")
def predict(data: ShipmentInput):
    try:
        result = predict_delay(data)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
