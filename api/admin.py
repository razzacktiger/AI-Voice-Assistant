# api/admin.py

import logging
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlmodel import Session, select

from .database import get_session
from .models import LogEntry, LogLevel, User  # Import LogLevel enum
from .auth import get_current_active_user  # Reuse auth dependency

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/logs", response_model=List[LogEntry])
async def get_logs(
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user),  # Secure endpoint
    # Filtering parameters
    level: Optional[LogLevel] = Query(
        None, description="Filter by log level (INFO, WARNING, ERROR, DEBUG)"),
    component: Optional[str] = Query(
        None, description="Filter by component name (e.g., websocket, rag, tts)"),
    event: Optional[str] = Query(
        None, description="Filter by event name (e.g., connection_start, stt_error)"),
    user_id: Optional[UUID] = Query(None, description="Filter by user UUID"),
    session_id: Optional[UUID] = Query(
        None, description="Filter by call session UUID"),
    start_time: Optional[datetime] = Query(
        None, description="Include logs from this time (ISO format)"),
    end_time: Optional[datetime] = Query(
        None, description="Include logs up to this time (ISO format)"),
    # Pagination parameters
    skip: int = Query(0, ge=0, description="Number of logs to skip"),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of logs to return (max 1000)")
):
    """Retrieves log entries with filtering and pagination.
    Requires admin privileges (TODO: Implement role checking).
    """
    logger.info(f"Admin user {current_user.email} requested logs with filters - Level: {level}, Component: {component}, Event: {event}, User: {user_id}, Session: {session_id}, Skip: {skip}, Limit: {limit}")

    # TODO: Add role check here - e.g., if not current_user.is_admin: raise HTTPException(403)

    try:
        statement = select(LogEntry)

        # Apply filters dynamically
        if level:
            statement = statement.where(LogEntry.level == level)
        if component:
            statement = statement.where(LogEntry.component == component)
        if event:
            statement = statement.where(LogEntry.event == event)
        if user_id:
            statement = statement.where(LogEntry.user_id == user_id)
        if session_id:
            statement = statement.where(LogEntry.session_id == session_id)
        if start_time:
            statement = statement.where(LogEntry.timestamp >= start_time)
        if end_time:
            statement = statement.where(LogEntry.timestamp <= end_time)

        # Apply ordering, pagination
        statement = statement.order_by(
            LogEntry.timestamp.desc()).offset(skip).limit(limit)

        logs_result = db.exec(statement)
        logs = logs_result.all()
        return logs

    except Exception as e:
        logger.error(
            f"Error fetching logs for admin {current_user.email}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error fetching logs.")
