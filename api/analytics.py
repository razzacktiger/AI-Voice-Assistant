# api/analytics.py

import logging
import io
import csv
from typing import List, Optional
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select, func, case

from .database import get_session
# Import AnalyticsReport
from .models import CallSession, CallStatus, User, AnalyticsReport
from .auth import get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter()


def get_analytics_report(db: Session) -> AnalyticsReport:
    """Calculates analytics based on CallSession data.

    Args:
        db (Session): The database session.

    Returns:
        AnalyticsReport: The calculated analytics summary.
    """
    try:
        # Calculate total calls
        total_calls_stmt = select(func.count(CallSession.id))
        total_calls = db.exec(total_calls_stmt).one()

        if total_calls == 0:
            return AnalyticsReport(total_calls=0, total_error_calls=0)

        # Calculate total calls with ERROR status
        total_error_calls_stmt = select(func.count(CallSession.id)).where(
            CallSession.status == CallStatus.ERROR
        )
        total_error_calls = db.exec(total_error_calls_stmt).one()

        # Calculate average duration for COMPLETED calls (status != ERROR and end_time is not null)
        # Use func.avg over the difference between end_time and start_time
        # Need to handle potential NULL end_times and filter for non-error sessions
        # Use func.extract('epoch', ...) for PostgreSQL to get duration in seconds
        avg_duration_stmt = select(func.avg(func.extract('epoch', CallSession.end_time - CallSession.start_time))).\
            where(CallSession.end_time != None).\
            where(CallSession.status != CallStatus.ERROR)

        average_duration_seconds = db.exec(avg_duration_stmt).one_or_none()
        # The result might be a Decimal or float, ensure it's float or None
        if average_duration_seconds is not None:
            average_duration_seconds = float(average_duration_seconds)

        # Calculate error rate
        error_rate = (total_error_calls / total_calls) * \
            100 if total_calls > 0 else 0

        return AnalyticsReport(
            total_calls=total_calls,
            average_call_duration_seconds=average_duration_seconds,
            total_error_calls=total_error_calls,
            error_rate=error_rate
        )

    except Exception as e:
        logger.error(f"Error calculating analytics: {e}", exc_info=True)
        # Depending on desired behavior, could return empty report or raise HTTP exception
        raise HTTPException(
            status_code=500, detail="Error calculating analytics data.")


@router.get("/summary", response_model=AnalyticsReport)
async def get_summary(
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)  # Secure endpoint
):
    """Retrieves a summary report of call analytics."""
    # Note: Add role-based access control here if needed
    logger.info(f"User {current_user.email} requested analytics summary.")
    report = get_analytics_report(db)
    return report


@router.get("/export")
async def export_analytics(
    db: Session = Depends(get_session),
    current_user: User = Depends(get_current_active_user)  # Secure endpoint
):
    """Exports detailed call session data as a CSV file."""
    logger.info(f"User {current_user.email} requested analytics export.")
    try:
        # Fetch all relevant call session data
        # Could add filters later (date range, user, etc.)
        stmt = select(CallSession).order_by(CallSession.start_time)
        sessions = db.exec(stmt).all()

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header row
        writer.writerow(["session_id", "user_id", "start_time",
                        "end_time", "status", "duration_seconds"])

        # Write data rows
        for session in sessions:
            duration = None
            if session.end_time and session.start_time:
                duration = (session.end_time -
                            session.start_time).total_seconds()
            writer.writerow([
                str(session.id),
                str(session.user_id),
                session.start_time.isoformat(),
                session.end_time.isoformat() if session.end_time else None,
                session.status.value,
                duration
            ])

        output.seek(0)
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=analytics_export.csv"}
        )

    except Exception as e:
        logger.error(f"Error exporting analytics data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="Error exporting analytics data.")
