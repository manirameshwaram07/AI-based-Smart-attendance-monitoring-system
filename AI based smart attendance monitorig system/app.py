import json
import sqlite3
from datetime import datetime

import pandas as pd
from flask import Flask, jsonify, render_template, request

from analytics_engine import AttendanceAnalyticsEngine
from attendance_engine import SmartAttendanceEngine

app = Flask(__name__, template_folder="templates")
DB_PATH = "attendance.db"
engine = SmartAttendanceEngine(db_path=DB_PATH)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            department TEXT,
            role TEXT DEFAULT 'Student',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS facial_embeddings (
            embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            embedding_data TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
        )
        """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS attendance_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'PRESENT',
            confidence_score FLOAT,
            liveness_verified BOOLEAN DEFAULT TRUE,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


@app.before_request
def ensure_database_ready():
    init_db()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/register_user", methods=["POST"])
def register_user():
    """Registers a user along with their 128-dimensional facial vector."""
    init_db()
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    full_name = data.get("full_name") or data.get("name") or user_id
    department = data.get("department") or "General"
    embedding_vector = data.get("embedding")

    if not user_id:
        return jsonify({"error": "User ID is required."}), 400
    if embedding_vector is None:
        return jsonify({"error": "Embedding is required for registration."}), 400

    try:
        embedding = [float(v) for v in embedding_vector]
    except (TypeError, ValueError):
        return jsonify({"error": "Embedding must be a list of numeric values."}), 400

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            """
            INSERT INTO users (user_id, full_name, department, role)
            VALUES (?, ?, ?, 'Student')
            ON CONFLICT(user_id) DO UPDATE SET full_name = excluded.full_name,
            department = excluded.department
            """,
            (user_id, full_name, department),
        )
        c.execute("DELETE FROM facial_embeddings WHERE user_id = ?", (user_id,))
        c.execute(
            "INSERT INTO facial_embeddings (user_id, embedding_data) VALUES (?, ?)",
            (user_id, json.dumps(embedding)),
        )
        conn.commit()

    engine.load_known_faces()
    return jsonify({
        "status": "Success",
        "message": f"User {user_id} registered successfully.",
        "user_id": user_id,
    })


@app.route("/api/mark_attendance", methods=["POST"])
def mark_attendance():
    """Endpoint for streaming camera client to log verified recognitions."""
    init_db()
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    confidence = float(data.get("confidence", 1.0) or 1.0)
    liveness_verified = bool(data.get("liveness_verified", True))

    if not user_id:
        return jsonify({"error": "User ID is required."}), 400

    now = datetime.now()
    current_time = now.time()
    cutoff_time = datetime.strptime("09:15:00", "%H:%M:%S").time()
    status = "PRESENT" if current_time < cutoff_time else "LATE"

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO attendance_logs (user_id, check_in_time, status, confidence_score, liveness_verified) VALUES (?, ?, ?, ?, ?)",
            (user_id, now.strftime("%Y-%m-%d %H:%M:%S"), status, confidence, liveness_verified),
        )
        conn.commit()

    return jsonify({
        "status": "Logged",
        "user_id": user_id,
        "check_in_status": status,
        "confidence": confidence,
    })


@app.route("/api/analytics/dashboard", methods=["GET"])
def get_analytics():
    """Provides executive metrics, risk predictions, and anomaly insights."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM attendance_logs", conn)

    if df.empty:
        return jsonify({"message": "No logs available for analytics."})

    analytics = AttendanceAnalyticsEngine(df)
    return jsonify({
        "summary_kpis": analytics.compute_kpis(),
        "risk_predictions": analytics.train_absenteeism_risk_model(),
        "anomalies": analytics.detect_anomalies().to_dict(orient="records"),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
