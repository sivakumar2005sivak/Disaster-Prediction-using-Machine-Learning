"""
Alternative Flask Web Application
For running with: python app.py or flask run
"""
import os
import json
from ml_engine import predict_observation, get_or_train_model
from database import init_database, save_prediction, get_recent_predictions, execute_raw_sql, log_training_run

try:
    from flask import Flask, request, jsonify, send_from_directory
    app = Flask(__name__, static_folder=".", static_url_path="")

    init_database()

    @app.route("/")
    def index():
        return send_from_directory(".", "index.html")

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json() or {}
        res = predict_observation(data)
        pid = save_prediction(res)
        res["database_id"] = pid
        return jsonify(res)

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        _, m = get_or_train_model()
        return jsonify(m)

    @app.route("/api/train", methods=["POST"])
    def train():
        _, m = get_or_train_model()
        lid = log_training_run(m)
        m["log_id"] = lid
        return jsonify({"status": "success", "metrics": m})

    @app.route("/api/history", methods=["GET"])
    def history():
        limit = int(request.args.get("limit", 25))
        return jsonify({"records": get_recent_predictions(limit)})

    @app.route("/api/sql", methods=["POST"])
    def sql_exec():
        data = request.get_json() or {}
        query = data.get("query", "SELECT * FROM disaster_predictions LIMIT 10")
        return jsonify(execute_raw_sql(query))

    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=5000, debug=True)

except ImportError:
    # If Flask is not installed, fallback message
    if __name__ == "__main__":
        print("Flask is not installed in this environment. Run server.py instead: python3 server.py")
