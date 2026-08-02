from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import score, score_paysim

app = FastAPI(title="VigilNet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(score.router)
app.include_router(score_paysim.router)

@app.get("/health")
def health():
    return {"status": "ok"}