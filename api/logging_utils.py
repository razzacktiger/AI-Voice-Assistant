# api/logging_utils.py

from sqlmodel import Session, select
from typing import Optional, Dict, Any
from uuid import UUID
import logging
from datetime import datetime, timedelta, timezone

from .models import LogEntry, LogLevel  # Import necessary models and enum

logger = logging.getLogger(__name__)


def add_log_entry(
    db: Session,
    level: LogLevel,
    event: str,
    user_id: Optional[UUID] = None,
    session_id: Optional[UUID] = None,
    component: Optional[str] = None,
    message: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Adds a log entry to the database.

    Args:
        db (Session): The database session.
        level (LogLevel): The severity level of the log.
        event (str): A short code or description of the event.
        user_id (Optional[UUID]): The ID of the associated user, if applicable.
        session_id (Optional[UUID]): The ID of the associated call session, if applicable.
        component (Optional[str]): The application component where the event originated.
        message (Optional[str]): A descriptive message for the log entry.
        details (Optional[Dict[str, Any]]): Additional structured data (JSON).
    """
    try:
        log_entry = LogEntry(
            user_id=user_id,
            session_id=session_id,
            level=level,
            component=component,
            event=event,
            message=message,
            details=details
            # timestamp is handled by default_factory
        )
        db.add(log_entry)
        db.commit()  # Commit immediately for logs?
        # Consider if batching commits or committing outside this function is better
        # db.flush() # Or just flush to get ID without full commit
        # db.refresh(log_entry) # Refresh if ID or defaults needed immediately
        logger.debug(f"Added log entry: {event} ({level})")
    except Exception as e:
        # Avoid crashing the main application flow if logging fails
        # Log the logging failure itself to the standard logger
        logger.error(
            f"Failed to add database log entry for event '{event}': {e}", exc_info=True)
        try:
            db.rollback()  # Rollback the specific failed log commit
        except Exception as rb_e:
            logger.error(f"Failed to rollback after logging error: {rb_e}")


def delete_old_logs(db: Session, days_to_keep: int = 30) -> int:
    """
    Deletes log entries older than the specified number of days.

    Args:
        db (Session): The database session.
        days_to_keep (int): The maximum age of logs to retain, in days.

    Returns:
        int: The number of log entries deleted.
    """
    try:
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
        logger.info(
            f"Deleting log entries older than {cutoff_date.isoformat()} ({days_to_keep} days old)...")

        # Construct the delete statement directly for efficiency
        # Note: SQLModel doesn't have direct bulk delete support like SQLAlchemy Core.
        # We need to query first and then delete, or use SQLAlchemy Core syntax.

        # Option 1: Query and Delete (might be slow for very large tables)
        statement = select(LogEntry).where(LogEntry.timestamp < cutoff_date)
        logs_to_delete = db.exec(statement).all()
        deleted_count = len(logs_to_delete)

        if not logs_to_delete:
            logger.info("No old log entries found to delete.")
            return 0

        logger.info(f"Found {deleted_count} old log entries to delete.")
        for log_entry in logs_to_delete:
            db.delete(log_entry)

        db.commit()
        logger.info(f"Successfully deleted {deleted_count} old log entries.")
        return deleted_count

        # Option 2: Use SQLAlchemy Core statement (More efficient for bulk deletes)
        # from sqlalchemy import delete
        # statement = delete(LogEntry).where(LogEntry.timestamp < cutoff_date)
        # result = db.exec(statement)
        # db.commit()
        # deleted_count = result.rowcount
        # logger.info(f"Successfully deleted {deleted_count} old log entries using bulk delete.")
        # return deleted_count

    except Exception as e:
        logger.error(f"Error deleting old log entries: {e}", exc_info=True)
        try:
            db.rollback()
        except Exception as rb_e:
            logger.error(
                f"Failed to rollback after log deletion error: {rb_e}")
        return 0  # Indicate failure or no deletion
