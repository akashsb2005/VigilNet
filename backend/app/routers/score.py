from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from app.models.alert import Alert
from app.services.ml_service import (
    score_by_txid,
    list_sample_transactions,
    search_transactions,
    score_new_transaction,
)
from app.services.sar_service import generate_sar

router = APIRouter(prefix="/score", tags=["scoring"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class ScoreRequest(BaseModel):
    txid: int


class NewTransactionRequest(BaseModel):
    amount: float
    fan_in: int
    fan_out: int
    community_id: int = 0


@router.get("/sample")
def get_sample_transactions():
    return list_sample_transactions(limit=25)


@router.get("/search")
def search(q: str):
    if len(q) < 3:
        return []
    return search_transactions(q)


@router.get("/alerts")
def list_alerts(db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.created_at.desc()).limit(50).all()
    return [
        {
            "id": a.id, "node_id": a.node_id, "confidence": a.confidence,
            "sar_narrative": a.sar_narrative, "status": a.status,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ]


@router.post("/")
def score_transaction(req: ScoreRequest, db: Session = Depends(get_db)):
    result = score_by_txid(req.txid)
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction ID not found in dataset")

    sar_text = None
    if result["confidence"] > 0.5:
        existing = db.query(Alert).filter(
            Alert.node_id == str(result["txid"]), Alert.status == "open"
        ).first()
        if existing:
            return {**result, "sar_narrative": existing.sar_narrative}

        sar_text = generate_sar(result["txid"], result["confidence"])
        alert = Alert(
            node_id=str(result["txid"]),
            confidence=result["confidence"],
            sar_narrative=sar_text,
            status="open",
        )
        db.add(alert)
        db.commit()

    return {**result, "sar_narrative": sar_text}


@router.post("/evaluate")
def evaluate_new_transaction(req: NewTransactionRequest):
    """Score a brand-new transaction never seen during training, using engineered
    features only (no real graph neighborhood — see README limitations)."""
    features = {
        "fan_in": req.fan_in,
        "fan_out": req.fan_out,
        "pass_through_ratio": req.fan_out / (req.fan_in + 1),
        "community_id": req.community_id,
    }
    result = score_new_transaction(features)
    return result