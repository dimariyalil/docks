import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lighthouse.models import CGDETRPredictor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CGDETRPredictor(
    device=device,
    feature_name="clip_slowfast"       # оставить или заменить на нужный
)

app = FastAPI(title="Lighthouse API")

class Query(BaseModel):
    video_path: str
    prompt: str

@app.post("/predict")
def predict(q: Query):
    try:
        video_feats = model.encode_video(q.video_path)
        result = model.predict(q.prompt, video_feats)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
