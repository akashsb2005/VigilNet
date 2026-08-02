from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from datetime import datetime
from app.core.database import Base

class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    node_id = Column(String, index=True)
    confidence = Column(Float)
    community_id = Column(Integer)
    community_illicit_concentration = Column(Float)
    sar_narrative = Column(Text)
    status = Column(String, default="open")
    created_at = Column(DateTime, default=datetime.utcnow)