import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split


class AttendanceAnalyticsEngine:
    """Aggregates attendance data and produces executive insights."""

    def __init__(self, attendance_df):
        self.df = attendance_df.copy()
        if self.df.empty:
            self.df = pd.DataFrame(columns=["user_id", "department", "check_in_time", "status"])

        self.df["check_in_time"] = pd.to_datetime(self.df["check_in_time"], errors="coerce")
        self.df = self.df.dropna(subset=["check_in_time"]).copy()

    def compute_kpis(self):
        """Calculates macro metrics for administration dashboards."""
        total_records = len(self.df)
        if total_records == 0:
            return {}

        present_count = int((self.df["status"] == "PRESENT").sum())
        late_count = int((self.df["status"] == "LATE").sum())
        absent_count = int((self.df["status"] == "ABSENT").sum())

        attendance_rate = round((present_count + late_count) / total_records * 100, 2)
        punctuality_rate = round(
            present_count / (present_count + late_count + 1e-5) * 100,
            2,
        )

        return {
            "total_logs": total_records,
            "overall_attendance_rate": f"{attendance_rate}%",
            "punctuality_rate": f"{punctuality_rate}%",
            "absenteeism_count": absent_count,
        }

    def detect_anomalies(self):
        """Uses Isolation Forest to detect anomalous check-in behavior."""
        if self.df.empty:
            return pd.DataFrame(columns=["user_id", "check_in_time", "status"])

        working_df = self.df.copy()
        working_df["minute_of_day"] = (
            working_df["check_in_time"].dt.hour * 60 + working_df["check_in_time"].dt.minute
        )

        if len(working_df) < 2:
            return working_df[["user_id", "check_in_time", "status"]].copy()

        contamination = 0.05 if len(working_df) >= 20 else max(0.1, 1 / len(working_df))
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        working_df["anomaly_score"] = iso_forest.fit_predict(working_df[["minute_of_day"]])

        anomalies = working_df[working_df["anomaly_score"] == -1]
        return anomalies[["user_id", "check_in_time", "status"]].copy()

    def train_absenteeism_risk_model(self):
        """Trains a Random Forest with attendance summaries for absenteeism risk."""
        if self.df.empty:
            return {"message": "No attendance data available for model training."}

        user_summary = (
            self.df.groupby("user_id", dropna=False)
            .agg(
                total_sessions=("status", "count"),
                lates=("status", lambda x: (x == "LATE").sum()),
                presents=("status", lambda x: (x == "PRESENT").sum()),
                absents=("status", lambda x: (x == "ABSENT").sum()),
            )
            .reset_index()
        )

        if user_summary.empty:
            return {"message": "No user attendance patterns available."}

        user_summary["attendance_pct"] = (
            (user_summary["presents"] + user_summary["lates"]) / user_summary["total_sessions"]
        )
        user_summary["is_at_risk"] = (user_summary["attendance_pct"] < 0.75).astype(int)

        if len(user_summary) < 10:
            return {
                "message": "Insufficient user data to train predictive ML model.",
                "at_risk_user_ids": [],
            }

        X = user_summary[["total_sessions", "lates", "absents"]]
        y = user_summary["is_at_risk"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        user_summary["predicted_risk"] = clf.predict(X)
        at_risk_users = user_summary[user_summary["predicted_risk"] == 1]["user_id"].tolist()

        return {
            "model_accuracy": round(accuracy * 100, 2),
            "at_risk_user_ids": at_risk_users,
        }
