from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone

# Using table=True makes these classes define database tables


class User(SQLModel, table=True):
    """Represents a user in the application database, linked to Firebase Auth."""
    id: Optional[int] = Field(default=None, primary_key=True)
    firebase_uid: str = Field(index=True, unique=True,
                              description="Firebase User ID")
    email: Optional[str] = Field(
        index=True, description="User email from Firebase")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    # Add other user profile fields as needed, e.g., name, preferences


class CallSession(SQLModel, table=True):
    """Represents a single voice call session."""
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", nullable=False, index=True)
    start_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    end_time: Optional[datetime] = Field(default=None)
    status: str = Field(default="started", index=True,
                        description="e.g., started, ended, error")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False)
    # Consider adding fields for call summary, transcript ID, etc.

# TODO: Add more models as needed, e.g., CallTranscript, LogEntry
