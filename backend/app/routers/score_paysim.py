from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.alert import Alert
from app.services.ml_service_paysim import score_by_account, list_sample_paysim, search_paysim_accounts
from app.services.sar_service import generate_sar

router = APIRouter(prefix="/score/paysim", tags=["paysim"])

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ScorePaysimRequest(BaseModel):
    account: str

@router.get("/sample")
def get_sample():
    return list_sample_paysim(limit=25)

@router.get("/search")
def search(q: str):
    if len(q) < 3:
        return []
    return search_paysim_accounts(q)

@router.post("/")
def score(req: ScorePaysimRequest, db: Session = Depends(get_db)):
    result = score_by_account(req.account)
    if result is None:
        raise HTTPException(status_code=404, detail="Account not found in PaySim dataset")

    sar_text = None
    if result["confidence"] > 0.5:
        existing = db.query(Alert).filter(
            Alert.node_id == result["account"], Alert.status == "open"
        ).first()
        if existing:
            return {**result, "sar_narrative": existing.sar_narrative}

        sar_text = generate_sar(result["account"], result["confidence"])
        alert = Alert(
            node_id=result["account"],
            confidence=result["confidence"],
            sar_narrative=sar_text,
            status="open",
        )
        db.add(alert)
        db.commit()

    return {**result, "sar_narrative": sar_text}