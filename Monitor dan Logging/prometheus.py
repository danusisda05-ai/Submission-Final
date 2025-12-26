from flask import Flask, request, jsonify
import requests
import time
from prometheus_client import make_wsgi_app, Counter, Histogram
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__)

# --- METRICS PROMETHEUS ---
# Menghitung total request yang masuk
REQUEST_COUNT = Counter('prediction_requests_total', 'Total number of prediction requests')
# Menghitung waktu respon model (Latency)
PREDICTION_LATENCY = Histogram('prediction_duration_seconds', 'Time spent processing prediction')

MLFLOW_URI = "http://127.0.0.1:5000/invocations"

# Menambahkan endpoint /metrics untuk dibaca Prometheus
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

@app.route('/predict', methods=['POST'])
def predict():
    REQUEST_COUNT.inc() # Tambah hitungan request
    start_time = time.time()
    
    try:
        data = request.get_json()
        response = requests.post(MLFLOW_URI, json=data)
        
        # Catat durasi prediksi
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)

        return jsonify(response.json()), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 API Wrapper berjalan di http://127.0.0.1:5001")
    # Ganti 9090 menjadi 5001 di sini
    app.run(host='0.0.0.0', port=5001)