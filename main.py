from fastapi import FastAPI
from app.api.endpoints import router

app = FastAPI(title="Multilingual Multimodal Phishing Detection API")

app.include_router(router)