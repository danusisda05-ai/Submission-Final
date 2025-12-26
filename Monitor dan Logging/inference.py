from flask import Flask, request, jsonify
import requests
import time
from prometheus_flask_exporter import PrometheusMetrics

app = Flask(__name__)

# --- INISIALISASI PROMETHEUS ---
# Ini akan otomatis membuat endpoint /metrics di port yang sama
metrics = PrometheusMetrics(app)

# Tambahkan info static untuk identitas
metrics.info('app_info', 'Inference API Wrapper', version='1.0.0')

MLFLOW_URI = "http://127.0.0.1:5000/invocations"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        response = requests.post(MLFLOW_URI, json=data)
        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Pastikan route utama tidak kosong agar kita tahu server jalan
@app.route('/')
def index():
    return "API Wrapper is Running! Check /metrics for data."

if __name__ == '__main__':
    print("🚀 Server berjalan di http://localhost:5001")
    print("📊 Metrik tersedia di http://localhost:5001/metrics")
    app.run(host='0.0.0.0', port=5001)