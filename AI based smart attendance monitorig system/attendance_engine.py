import json
import os
import sqlite3
from datetime import datetime

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import face_recognition
except Exception:  # pragma: no cover - optional dependency
    face_recognition = None

try:
    from scipy.spatial import distance as dist
except Exception:  # pragma: no cover - optional dependency
    dist = None


class SmartAttendanceEngine:
    """Face recognition plus lightweight anti-spoofing checks for attendance capture."""

    def __init__(self, db_path="attendance.db", distance_threshold=0.5, ear_threshold=0.21):
        self.db_path = db_path
        self.distance_threshold = distance_threshold
        self.ear_threshold = ear_threshold
        self.known_face_encodings = []
        self.known_face_ids = []
        self._ensure_schema()
        self.load_known_faces()

    def _ensure_schema(self):
        """Creates the required tables if the SQLite DB is brand new or was reset."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
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
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS facial_embeddings (
                embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                embedding_data TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            )
            """
        )
        cursor.execute(
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

    def load_known_faces(self):
        """Loads face embeddings from the database."""
        if not os.path.exists(self.db_path):
            self.known_face_encodings = []
            self.known_face_ids = []
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, embedding_data FROM facial_embeddings")
        rows = cursor.fetchall()

        self.known_face_encodings = []
        self.known_face_ids = []

        for user_id, embedding_json in rows:
            try:
                embedding = np.asarray(json.loads(embedding_json), dtype=np.float64)
                self.known_face_encodings.append(embedding)
                self.known_face_ids.append(user_id)
            except (TypeError, ValueError):
                continue

        conn.close()
        print(f"[INFO] Loaded {len(self.known_face_ids)} registered face embeddings.")

    @staticmethod
    def calculate_ear(eye_landmarks):
        """Calculates Eye Aspect Ratio (EAR) for blink detection."""
        if len(eye_landmarks) < 6:
            return 0.0

        if dist is not None:
            A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
            B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
            C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        else:
            A = np.linalg.norm(np.asarray(eye_landmarks[1]) - np.asarray(eye_landmarks[5]))
            B = np.linalg.norm(np.asarray(eye_landmarks[2]) - np.asarray(eye_landmarks[4]))
            C = np.linalg.norm(np.asarray(eye_landmarks[0]) - np.asarray(eye_landmarks[3]))

        if C == 0:
            return 0.0
        return (A + B) / (2.0 * C)

    def process_frame(self, frame):
        """Processes a single frame and returns annotated frame plus recognized users."""
        if cv2 is None or face_recognition is None:
            return frame, []

        if not self.known_face_encodings:
            return frame, []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

        recognized_users = []

        for index, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
            landmarks = face_landmarks_list[index] if index < len(face_landmarks_list) else {}

            left_eye = landmarks.get("left_eye")
            right_eye = landmarks.get("right_eye")

            liveness_passed = False
            if left_eye and right_eye:
                left_ear = self.calculate_ear(left_eye)
                right_ear = self.calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                if avg_ear >= self.ear_threshold:
                    liveness_passed = True

            if not self.known_face_encodings:
                user_id = "Unknown"
                confidence = 0.0
            else:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings,
                    face_encoding,
                    tolerance=self.distance_threshold,
                )
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                user_id = "Unknown"
                confidence = 0.0

                if len(face_distances) > 0:
                    best_match_index = int(np.argmin(face_distances))
                    if matches[best_match_index]:
                        user_id = self.known_face_ids[best_match_index]
                        confidence = round(float(1.0 - face_distances[best_match_index]), 4)

            color = (0, 255, 0) if (user_id != "Unknown" and liveness_passed) else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            status_text = f"{user_id} ({confidence * 100:.1f}%)" if liveness_passed else f"{user_id} [SPOOF RISK]"
            cv2.putText(frame, status_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if user_id != "Unknown" and liveness_passed:
                recognized_users.append(
                    {
                        "user_id": user_id,
                        "confidence": confidence,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )

        return frame, recognized_users
