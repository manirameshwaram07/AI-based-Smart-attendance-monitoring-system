# AI-Based Smart Attendance Monitoring & Analytics System

This project implements the architecture described in the blueprint for a smart attendance system using face recognition, liveness checks, and attendance analytics.

## Features

- Face registration using a 128-dimensional embedding vector
- Anti-spoofing/liveness validation through Eye Aspect Ratio (EAR)
- SQLite-backed attendance logging
- Analytics engine for KPIs, anomaly detection, and absenteeism risk prediction
- Flask dashboard with real-time metrics

## Project structure

- `app.py` – Flask API and dashboard entry point
- `attendance_engine.py` – face recognition and liveness engine
- `analytics_engine.py` – attendance analytics and ML risk modeling
- `templates/index.html` – executive dashboard UI

## Setup

```bash
python -m pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000 in a browser.

## API endpoints

- `POST /api/register_user` – register a user with an embedding list
- `POST /api/mark_attendance` – log attendance for an identified user
- `GET /api/analytics/dashboard` – fetch KPI and ML-based risk metrics

## Notes

The computer vision dependencies are optional at runtime. If OpenCV or face_recognition are not installed, the app still runs in analytics mode and gracefully skips video processing.
