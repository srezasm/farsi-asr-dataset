from sqlalchemy import create_engine, Column, Integer, Float, String, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from chunker import Caption
import uuid
from normalizer import ValidationStatus
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)

# Database URL and engine setup
DATABASE_URL = "sqlite:///data.db"
engine = create_engine(DATABASE_URL, echo=False)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our ORM models
Base = declarative_base()

# ORM Model for AudioChunk
class AudioChunk(Base):
    __tablename__ = 'audio_chunks'
    id = Column(Integer, primary_key=True, index=True)
    audio = Column(String, nullable=True)
    text = Column(String, nullable=False)
    source = Column(String, nullable=False)
    source_id = Column(String, nullable=False)
    start = Column(Float, nullable=False)
    end = Column(Float, nullable=False)
    invalidation = Column(SQLEnum(ValidationStatus), nullable=False)

# Utility to initialize (create) the database tables
def init_db():
    Base.metadata.create_all(bind=engine)

# Create (insert) a new chunk record into the database
def create_chunk(session: Session, audio: str, text: str, source: str,
                 source_id: str, start: float, end: float,
                 invalidation: ValidationStatus) -> AudioChunk:
    chunk = AudioChunk(
        audio=audio,
        text=text,
        source=source,
        source_id=source_id,
        start=start,
        end=end,
        invalidation=invalidation
    )
    session.add(chunk)
    session.commit()
    session.refresh(chunk)
    return chunk

def create_chunks(session: Session, source: str, source_id: str, captions: list[Caption]) -> None:
    chunks = []
    for caption in captions:
        chunk = AudioChunk(
            audio=caption.filename,
            text=caption.text,
            source=source,
            source_id=source_id,
            start=caption.start,
            end=caption.end,
            invalidation=caption.status
        )
        chunks.append(chunk)
    
    session.add_all(chunks)
    session.commit()


# Update an existing chunk record by its id
def update_chunk(session: Session, chunk_id: int, **kwargs) -> AudioChunk:
    chunk = session.query(AudioChunk).filter(AudioChunk.id == chunk_id).first()
    if not chunk:
        raise ValueError(f"Chunk with id {chunk_id} not found")
    for key, value in kwargs.items():
        if hasattr(chunk, key):
            setattr(chunk, key, value)
    session.commit()
    session.refresh(chunk)
    return chunk

# Delete a chunk record by its id
def delete_chunk(session: Session, chunk_id: int) -> None:
    chunk = session.query(AudioChunk).filter(AudioChunk.id == chunk_id).first()
    if not chunk:
        raise ValueError(f"Chunk with id {chunk_id} not found")
    session.delete(chunk)
    session.commit()

# Retrieve a chunk record by its id
def get_chunk(session: Session, chunk_id: int) -> AudioChunk:
    return session.query(AudioChunk).filter(AudioChunk.id == chunk_id).first()

# Context manager to handle session lifecycle
@contextmanager
def get_db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

# Example usage demonstrating create, update, retrieve, and delete operations
if __name__ == "__main__":
    init_db()  # Create the tables if they don't already exist

    with get_db_session() as session:
        # Create a new chunk
        new_chunk = create_chunk(
            session=session,
            audio="chunk1.wav",
            text="Sample text",
            source="youtube",
            source_id=uuid.uuid4(),
            start=0.0,
            end=5.0,
            invalidation=ValidationStatus.VALID
        )
        print("Created chunk with id:", new_chunk.id)

        # Update the chunk (e.g., change the text)
        updated_chunk = update_chunk(session, new_chunk.id, text="Updated sample text")
        print("Updated chunk text:", updated_chunk.text)

        # Retrieve the chunk
        retrieved_chunk = get_chunk(session, new_chunk.id)
        print("Retrieved chunk:", retrieved_chunk)

        # Delete the chunk
        delete_chunk(session, new_chunk.id)
        print("Deleted chunk with id:", new_chunk.id)
