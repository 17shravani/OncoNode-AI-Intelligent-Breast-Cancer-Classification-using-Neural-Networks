from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship, declarative_base
import datetime

Base = declarative_base()

class Patient(Base):
    __tablename__ = 'patients'
    
    id = Column(String(50), primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    age = Column(Integer, nullable=False)
    gender = Column(String(10), nullable=False)
    hospital_id = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    biopsies = relationship("ClinicalBiopsy", back_populates="patient", cascade="all, delete-orphan")
    audits = relationship("AuditLog", back_populates="patient")

class ClinicalBiopsy(Base):
    __tablename__ = 'clinical_biopsies'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(String(50), ForeignKey('patients.id'), nullable=False)
    features = Column(JSON, nullable=False) # JSON dict of 30 features
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="biopsies")
    prediction = relationship("PredictionResult", uselist=False, back_populates="biopsy", cascade="all, delete-orphan")

class PredictionResult(Base):
    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    biopsy_id = Column(Integer, ForeignKey('clinical_biopsies.id'), nullable=False)
    prediction_class = Column(Integer, nullable=False) # 0 = Malignant, 1 = Benign
    malignancy_prob = Column(Float, nullable=False)
    model_used = Column(String(50), nullable=False)
    agent_insights = Column(JSON, nullable=True) # Swarm insights
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    biopsy = relationship("ClinicalBiopsy", back_populates="prediction")

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String(50), nullable=False)
    action = Column(String(100), nullable=False) # e.g. "VIEW_PATIENT", "RUN_DIAGNOSTIC"
    patient_id = Column(String(50), ForeignKey('patients.id'), nullable=True)
    ip_address = Column(String(45), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    
    patient = relationship("Patient", back_populates="audits")
