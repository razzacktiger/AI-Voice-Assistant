import os

import sqlalchemy as sa
from sqlmodel import create_engine, Session, SQLModel
from dotenv import load_dotenv

# Import ALL models needed for table creation
from . import models  # Import the module to access its classes

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set.")

# Consider connection pooling options for production
# Set echo=False for production
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables():
    """Creates database tables based on SQLModel metadata."""
    # Make sure all models are imported before calling create_all
    # This ensures their metadata is registered with SQLModel.metadata
    # Example: models.User, models.CallSession, models.LogEntry
    print("Creating database tables...")
    SQLModel.metadata.create_all(engine)
    print("Database tables created (if they didn't exist).")


def get_session():
    """FastAPI dependency to get a database session."""
    with Session(engine) as session:
        yield session

# Potential future CRUD operations could go here, e.g.:
# def get_user_by_firebase_uid(session: Session, firebase_uid: str) -> Optional[User]:
#     ...
# def create_user(session: Session, user_data: UserCreate) -> User:
#     ...
