import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlmodel import Session, select, func
from dotenv import load_dotenv
import csv
from io import StringIO
import logging

# Import modules from the api package
# Assuming TTS will be added later
from . import database, auth, websocket, models, rag

# --- Background Task Imports ---
from fastapi_utils.tasks import repeat_every
from fastapi_utils.session import FastAPISessionMaker
from .logging_utils import delete_old_logs

load_dotenv()  # Load environment variables

logger = logging.getLogger(__name__)

# --- Database Session Maker for Tasks ---
# Needs a configured sessionmaker for use outside request context
sessionmaker = FastAPISessionMaker(database.engine.url)
# ---------------------------------------

# --- App Lifecycle (for DB setup) ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for application lifespan events."""
    print("Application startup...")
    print("Dropping existing tables (if any)... DEV ONLY")
    # WARNING: This deletes all data! Only for development.
    database.SQLModel.metadata.drop_all(database.engine)
    print("Creating database tables...")
    database.create_db_and_tables()  # Create tables if they don't exist
    # Initialize periodic log deletion task on startup
    await delete_old_logs_task()
    yield
    print("Application shutdown...")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="AI Voice Assistant API",
    description="Backend API for the real-time AI voice assistant.",
    version="0.1.0",
    lifespan=lifespan  # Use lifespan for startup/shutdown events
)

# --- CORS Configuration ---
# Allow requests from your frontend origin
# TODO: Replace "*" with your specific frontend URL in production
origins = [
    "http://localhost",
    "http://localhost:3000",  # Example for local React dev server
    # Get from env or default (adjust default for prod)
    os.getenv("FRONTEND_URL", "*")
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- API Routes ---

# Include WebSocket route from the websocket module's router
app.include_router(websocket.router)


@app.get("/")
def read_root():
    """Root endpoint providing basic API info."""
    return {"message": "Welcome to the AI Voice Assistant API"}

# --- Stub Endpoints (from original api.py) ---

# Example protected route using the auth dependency


@app.get("/users/me", response_model=models.User)
async def read_users_me(current_user: models.User = Depends(auth.get_current_active_user)):
    """Returns the current authenticated user's details."""
    return current_user


class AnalyticsReport(models.SQLModel):
    total_calls: int
    average_duration_seconds: Optional[float]
    # Add more fields as needed


@app.get("/api/analytics/report", response_model=AnalyticsReport)
async def get_analytics_report(
    db: Session = Depends(database.get_session),
    current_user: models.User = Depends(
        auth.get_current_active_user)  # Protect endpoint
):
    """Stub for analytics report endpoint."""
    # TODO: Implement actual analytics logic by querying LogEntry or CallSession
    print(f"Analytics report requested by user: {current_user.email}")

    # Use SQLModel syntax for count query
    count_statement = select(func.count(models.CallSession.id)).where(
        models.CallSession.user_id == current_user.id)
    total_calls = db.exec(
        count_statement).scalar_one() or 0  # Handle None case

    # Use SQLModel syntax for session list query
    sessions_statement = select(models.CallSession).where(
        models.CallSession.user_id == current_user.id,
        models.CallSession.end_time != None
    )
    completed_sessions = db.exec(sessions_statement).all()

    durations = [
        (session.end_time - session.start_time).total_seconds()
        for session in completed_sessions
        if session.end_time  # Should always be true due to filter, but safe check
    ]
    avg_duration = sum(durations) / len(durations) if durations else None

    return AnalyticsReport(total_calls=total_calls, average_duration_seconds=avg_duration)


@app.get("/api/admin/export")
async def export_analytics_csv(
    db: Session = Depends(database.get_session),
    current_user: models.User = Depends(
        auth.get_current_active_user)  # Protect endpoint
):
    """Stub for exporting analytics data as CSV."""
    # TODO: Implement logic to query data, format as CSV, and return FileResponse
    print(f"Admin CSV export requested by user: {current_user.email}")

    # Use SQLModel syntax to query sessions
    sessions_statement = select(models.CallSession).where(
        models.CallSession.user_id == current_user.id)
    sessions = db.exec(sessions_statement).all()

    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["session_id", "user_email", "start_time",
                    "end_time", "duration_seconds"])

    for sess in sessions:
        duration = (
            sess.end_time - sess.start_time).total_seconds() if sess.end_time else "N/A"
        writer.writerow([
            sess.id,
            current_user.email,  # Assuming we only export for the current user here
            sess.start_time,
            sess.end_time or "N/A",
            duration
        ])

    output.seek(0)
    return StreamingResponse(output, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=analytics_export.csv"})

# --- Background Task Definition ---


# Run daily, wait for first run
@repeat_every(seconds=60 * 60 * 24, wait_first=True, logger=logger)
async def delete_old_logs_task() -> None:
    """Periodic task to delete old log entries."""
    logger.info("Running scheduled task: delete_old_logs_task")
    try:
        # Get a database session for the task
        with sessionmaker.context_session() as db:
            deleted_count = delete_old_logs(
                db=db, days_to_keep=30)  # Pass db session
            logger.info(
                f"Scheduled log deletion task finished. Deleted {deleted_count} entries.")
    except Exception as e:
        logger.error(
            f"Error in scheduled task delete_old_logs_task: {e}", exc_info=True)
# --------------------------------

# --- Optional: Add main block for running with uvicorn ---
# if __name__ == "__main__":
#     import uvicorn
#     print("Starting Uvicorn server...")
#     # Ensure reload is False if you're using the lifespan for DB creation
#     # or manage DB creation/migrations separately.
#     # uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False) # Set reload=False for production
