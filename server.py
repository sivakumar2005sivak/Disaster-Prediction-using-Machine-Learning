#!/usr/bin/env python3
"""
Python Web Backend Server for ML-Based Disaster Risk Prediction System
Uses standard Python library (http.server, json, sqlite3)
Supports College Viva & Lab Demonstrations without Node or React.
Run with: python3 server.py
"""

import os
import sys
import json
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
from ml_engine import predict_observation, get_or_train_all_models, get_or_train_model, compare_all_models_on_input
from database import init_database, save_prediction, get_recent_predictions, execute_raw_sql, log_training_run

PORT = 5000 if len(sys.argv) < 2 else int(sys.argv[1])
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class DisasterRiskHTTPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.path = "/index.html"
            return super().do_GET()

        elif path == "/api/metrics":
            _, _, comp = get_or_train_all_models()
            self._send_json({
                "models": comp,
                "benchmark_report_values": {
                    "accuracy": 0.6333,
                    "precision": 0.6645,
                    "recall": 0.6333,
                    "f1_score": 0.6024,
                    "reference": "Appendix B, Page 24 (Hindusthan College of Technology report)"
                },
                "feature_importances": {
                    "water_level_m": 0.28,
                    "rainfall_mm": 0.22,
                    "distance_to_river_km": 0.16,
                    "slope_deg": 0.12,
                    "soil_moisture_pct": 0.09,
                    "wind_speed_kmh": 0.06,
                    "humidity_pct": 0.04,
                    "elevation_m": 0.02,
                    "temperature_c": 0.01
                }
            })

        elif path == "/api/history":
            query_params = urllib.parse.parse_qs(parsed.query)
            limit = int(query_params.get("limit", [25])[0])
            rows = get_recent_predictions(limit)
            self._send_json({"records": rows})

        elif path == "/api/stations":
            res = execute_raw_sql("SELECT * FROM monitoring_stations ORDER BY station_id ASC")
            self._send_json(res)

        else:
            return super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        try:
            payload = json.loads(body) if body else {}
        except Exception:
            payload = {}

        if path == "/api/predict":
            # Run ML prediction pipeline with selected algorithm
            model_id = payload.get("model_id", "random_forest")
            prediction_result = predict_observation(payload, model_id)
            # Save to SQL database
            pred_id = save_prediction(prediction_result)
            prediction_result["database_id"] = pred_id
            self._send_json(prediction_result)

        elif path == "/api/compare":
            # Run all 6 candidate algorithms simultaneously
            compare_res = compare_all_models_on_input(payload)
            self._send_json(compare_res)

        elif path == "/api/train":
            # Retrain model and evaluate
            _, metrics = get_or_train_model()
            log_id = log_training_run(metrics)
            metrics["log_id"] = log_id
            self._send_json({"status": "success", "metrics": metrics})

        elif path == "/api/sql":
            query = payload.get("query", "SELECT * FROM disaster_predictions LIMIT 10")
            result = execute_raw_sql(query)
            self._send_json(result)

        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'{"error": "Not Found"}')

    def _send_json(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server():
    init_database()
    print("Initializing ML Model...")
    get_or_train_model()
    server_address = ("0.0.0.0", PORT)
    httpd = HTTPServer(server_address, DisasterRiskHTTPRequestHandler)
    print(f"ML Disaster Risk Prediction Server running at http://0.0.0.0:{PORT}")
    print(f"Frontend: HTML/CSS/JS | Backend: Python 3 | Database: SQLite (SQL)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
