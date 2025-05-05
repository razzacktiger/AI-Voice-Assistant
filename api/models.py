import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from enum import Enum
import json

import sqlalchemy as sa
from sqlalchemy import JSON
from sqlmodel import Field, Relationship, SQLModel, create_engine, Session, select, Column
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Set echo=False for production
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session

# --- Enums (Define BEFORE use in models) ---


class CallStatus(str, Enum):
    ACTIVE = "active"
    ENDED = "ended"
    ERROR = "error"


class LogLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    DEBUG = "DEBUG"  # Optional

# --- Models ---

# Pydantic models for request/response (can be SQLModels too if they map directly)


class TokenData(SQLModel):
    user_id: str | None = None

# Database Models


class UserBase(SQLModel):
    email: str = Field(index=True, unique=True)
    firebase_uid: str = Field(index=True, unique=True)
    # Add other user fields if needed (e.g., name, creation_date)


class User(UserBase, table=True):
    id: Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    # Define relationship to CallSession
    call_sessions: List["CallSession"] = Relationship(back_populates="user")


class UserRead(UserBase):
    id: uuid.UUID
    created_at: datetime

# Call Session model


class CallSessionBase(SQLModel):
    user_id: uuid.UUID = Field(foreign_key="user.id")
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc))
    end_time: Optional[datetime] = Field(default=None)
    # Now CallStatus is defined above
    status: CallStatus = Field(default=CallStatus.ACTIVE)
    # Store conversation history as JSON. Adjust if structure becomes complex.
    conversation_history: List[Dict[str, Any]] = Field(
        default=[], sa_column=Column(JSON))


class CallSession(CallSessionBase, table=True):
    id: Optional[uuid.UUID] = Field(
        default_factory=uuid.uuid4, primary_key=True)
    # Define relationship back to User
    user: Optional[User] = Relationship(back_populates="call_sessions")


class CallSessionRead(CallSessionBase):
    id: uuid.UUID


class CallSessionCreate(CallSessionBase):
    pass  # Inherits all needed fields

# Logging Models
# (LogLevel enum is defined above)


class LogEntry(SQLModel, table=True):
    # Use int PK for logs typically
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), index=True)
    session_id: Optional[uuid.UUID] = Field(
        default=None, foreign_key="callsession.id", index=True)  # Link to CallSession if available
    user_id: Optional[uuid.UUID] = Field(
        default=None, foreign_key="user.id", index=True)  # Link to User if available
    level: LogLevel = Field(default=LogLevel.INFO, index=True)
    # e.g., 'websocket', 'rag', 'tts', 'auth'
    component: Optional[str] = Field(default=None, index=True)
    # e.g., 'connection_start', 'stt_result', 'error', 'tts_request'
    event: str = Field(index=True)
    message: Optional[str] = Field(default=None)
    details: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(JSON))  # For extra structured data

    # Relationships (Optional, could add if needed for complex queries, but might impact performance)
    # call_session: Optional["CallSession"] = Relationship() # unidirectional for now
    # user: Optional["User"] = Relationship() # unidirectional for now

# Remove explicit index definition - index=True on timestamp field is sufficient

# --- Initial table creation call ---
# You might call this once manually or use Alembic migrations for production
# create_db_and_tables()
