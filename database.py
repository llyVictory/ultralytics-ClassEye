import os
import numpy as np
import datetime
import binascii
from typing import List, Optional, Tuple, Dict
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, LargeBinary, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# MySQL configuration
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'root')
DB_NAME = os.getenv('DB_NAME', 'insightface_empower')

# Create SQLAlchemy engine
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 1. User Face Feature Table
class UserFaceFeature(Base):
    __tablename__ = 'face_feature_lite'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    number = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    feature_blob = Column(LargeBinary, nullable=False) # 512-dim vector
    created_at = Column(DateTime, default=datetime.datetime.now)

# 2. Identification Log Table (Updated with Threshold and all-log capability)
class IdentifyLog(Base):
    __tablename__ = 'face_identify_log'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    number = Column(String(50), nullable=True) # Can be NULL if not matched
    name = Column(String(100), nullable=True) # Can be NULL if not matched
    score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False) # Current threshold at time of log
    status = Column(String(20), default='pass') # 'pass' or 'not_pass'
    created_at = Column(DateTime, default=datetime.datetime.now)

def init_db():
    """Initialize database tables"""
    try:
        with engine.connect() as conn:
            print("[DB] Connected to MySQL successfully")
        Base.metadata.create_all(bind=engine)
        print("[DB] Database tables initialized")
    except Exception as e:
        print(f"[DB] Connection FAILED: {e}")

# --- Data Operations ---

def save_face(number: str, name: str, feature: np.ndarray):
    """Save or update face feature"""
    session = SessionLocal()
    try:
        feature_blob = feature.tobytes()
        user = session.query(UserFaceFeature).filter(UserFaceFeature.number == number).first()
        if user:
            user.name = name
            user.feature_blob = feature_blob
            user.created_at = datetime.datetime.now()
        else:
            user = UserFaceFeature(number=number, name=name, feature_blob=feature_blob)
            session.add(user)
        session.commit()
        return True
    except Exception as e:
        print(f"[DB] Save failed: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def get_all_faces() -> List[Tuple[str, str, np.ndarray]]:
    """Get all features for 1:N matching"""
    session = SessionLocal()
    try:
        users = session.query(UserFaceFeature).all()
        results = []
        for u in users:
            feature = np.frombuffer(u.feature_blob, dtype=np.float32)
            results.append((u.number, u.name, feature))
        return results
    except Exception as e:
        print(f"[DB] Query failed: {e}")
        return []
    finally:
        session.close()

def get_user_list():
    """Get registry list with feature preview"""
    session = SessionLocal()
    try:
        users = session.query(UserFaceFeature).order_by(UserFaceFeature.created_at.desc()).all()
        return [
            {
                "number": u.number,
                "name": u.name,
                "feature_preview": binascii.hexlify(u.feature_blob[:12]).decode('utf-8').upper() + "...",
                "created_at": u.created_at.strftime('%Y-%m-%d %H:%M:%S')
            } for u in users
        ]
    finally:
        session.close()

def add_identify_log(number: Optional[str], name: Optional[str], score: float, threshold: float, status: str):
    """Log identification attempt (passes or failures)"""
    session = SessionLocal()
    try:
        log = IdentifyLog(number=number, name=name, score=score, threshold=threshold, status=status)
        session.add(log)
        session.commit()
        return True
    except Exception as e:
        print(f"[DB] Log failed: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def get_history_logs(limit: int = 50):
    """Get history entries including status and threshold"""
    session = SessionLocal()
    try:
        logs = session.query(IdentifyLog).order_by(IdentifyLog.created_at.desc()).limit(limit).all()
        return [
            {
                "number": l.number if l.number else "----",
                "name": l.name if l.name else "未知人员",
                "score": round(l.score, 4),
                "threshold": l.threshold,
                "status": l.status,
                "created_at": l.created_at.strftime('%Y-%m-%d %H:%M:%S')
            } for l in logs
        ]
    finally:
        session.close()

def update_user(number: str, name: str):
    """Update user's name"""
    session = SessionLocal()
    try:
        user = session.query(UserFaceFeature).filter(UserFaceFeature.number == number).first()
        if user:
            user.name = name
            session.commit()
            return True
        return False
    except Exception as e:
        print(f"[DB] Update failed: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def delete_user(number: str):
    """Delete user and their face feature"""
    session = SessionLocal()
    try:
        user = session.query(UserFaceFeature).filter(UserFaceFeature.number == number).first()
        if user:
            session.delete(user)
            session.commit()
            return True
        return False
    except Exception as e:
        print(f"[DB] Delete failed: {e}")
        session.rollback()
        return False
    finally:
        session.close()

def is_user_exists(number: str) -> bool:
    """Check if user number already exists"""
    session = SessionLocal()
    try:
        user = session.query(UserFaceFeature).filter(UserFaceFeature.number == number).first()
        return user is not None
    finally:
        session.close()

if __name__ == "__main__":
    init_db()
