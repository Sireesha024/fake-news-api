from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("model/lr_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

app = FastAPI()

# Request body model
class InputData(BaseModel):
    domain: str
    title: str
    content: str

@app.get("/")
def read_root():
    return {"message": "Fake News Predictor API is running"}

@app.post("/predict")
def predict_news(data: InputData):
    try:
        # Combine text fields
        full_text = f"{data.domain} {data.title} {data.content}"

        # Transform input using the same vectorizer used in training
        input_vec = vectorizer.transform([full_text])

        # Predict using the LR model
        prediction = model.predict(input_vec)[0]

        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
