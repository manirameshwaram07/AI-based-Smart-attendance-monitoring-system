import os

from app import app


if os.path.exists('attendance.db'):
    os.remove('attendance.db')

client = app.test_client()

reg = client.post(
    '/api/register_user',
    json={
        'user_id': 'u100',
        'full_name': 'Test User',
        'department': 'CS',
        'embedding': [0.1] * 128,
    },
)
print('register_status', reg.status_code, reg.get_json())

att = client.post(
    '/api/mark_attendance',
    json={'user_id': 'u100', 'confidence': 0.97, 'liveness_verified': True},
)
print('attendance_status', att.status_code, att.get_json())

dash = client.get('/api/analytics/dashboard')
print('dashboard_status', dash.status_code)
data = dash.get_json()
print('dashboard_keys', sorted(data.keys()))
print('kpis', data.get('summary_kpis'))
