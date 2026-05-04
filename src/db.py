"""
Database setup. Postgres for structured data, ChromaDB for semantic search.
Same stack as REIPv2 so deployment is familiar.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DATABASE_URL", "sqlite:///character_bible.db")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Video(Base):
    __tablename__ = "videos"
    id = Column(Integer, primary_key=True)
    url = Column(String, unique=True)
    title = Column(String)
    character = Column(String, index=True)  # billi_maasi, coco_bhaiya, etc
    notes = Column(Text)
    transcript = Column(Text)  # full whisper transcript
    created_at = Column(DateTime, default=datetime.utcnow)


class CharacterBible(Base):
    __tablename__ = "character_bibles"
    id = Column(Integer, primary_key=True)
    character = Column(String, unique=True, index=True)
    catchphrases = Column(Text)        # JSON list
    vocabulary = Column(Text)          # JSON list
    opinions = Column(Text)            # JSON dict {topic: opinion}
    relationships = Column(Text)       # JSON list
    backstory = Column(Text)           # JSON dict
    brands_referenced = Column(Text)   # JSON list
    aesthetic = Column(Text)           # JSON dict
    last_updated = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)
    print("Database initialized.")


if __name__ == "__main__":
    init_db()
