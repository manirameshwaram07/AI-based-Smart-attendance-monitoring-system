import pandas as pd

from analytics_engine import AttendanceAnalyticsEngine


def test_compute_kpis():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "department": "CS", "check_in_time": "2026-08-14 08:50:00", "status": "PRESENT"},
            {"user_id": "u1", "department": "CS", "check_in_time": "2026-08-14 09:20:00", "status": "LATE"},
            {"user_id": "u2", "department": "EE", "check_in_time": "2026-08-14 09:00:00", "status": "ABSENT"},
        ]
    )

    analytics = AttendanceAnalyticsEngine(df)
    kpis = analytics.compute_kpis()

    assert kpis["total_logs"] == 3
    assert kpis["absenteeism_count"] == 1
    assert "overall_attendance_rate" in kpis
    assert "punctuality_rate" in kpis


def test_train_absenteeism_risk_model_handles_small_data():
    df = pd.DataFrame(
        [
            {"user_id": "u1", "department": "CS", "check_in_time": "2026-08-14 08:45:00", "status": "PRESENT"},
            {"user_id": "u1", "department": "CS", "check_in_time": "2026-08-14 08:50:00", "status": "PRESENT"},
            {"user_id": "u1", "department": "CS", "check_in_time": "2026-08-14 09:10:00", "status": "LATE"},
            {"user_id": "u2", "department": "EE", "check_in_time": "2026-08-14 09:20:00", "status": "ABSENT"},
            {"user_id": "u2", "department": "EE", "check_in_time": "2026-08-14 09:21:00", "status": "ABSENT"},
        ]
    )

    analytics = AttendanceAnalyticsEngine(df)
    result = analytics.train_absenteeism_risk_model()

    assert "model_accuracy" in result or "Insufficient user data" in result.get("message", "")
