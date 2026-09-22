#!/usr/bin/env python3
"""
SQL Database Manager for ML-Based Disaster Risk Prediction System
Uses SQLite3 (standard SQL database engine)
Handles schema initialization, prediction logging, and live SQL querying.
"""

import os
import sys
import json
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "disaster_risk.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = get_connection()
    c = conn.cursor()

    # Table 1: Predictions log
    c.execute("""
    CREATE TABLE IF NOT EXISTS disaster_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        location_name TEXT NOT NULL,
        disaster_type TEXT NOT NULL,
        rainfall_mm REAL NOT NULL,
        temperature_c REAL NOT NULL,
        humidity_pct REAL NOT NULL,
        wind_speed_kmh REAL NOT NULL,
        soil_moisture_pct REAL NOT NULL,
        water_level_m REAL NOT NULL,
        elevation_m REAL NOT NULL,
        slope_deg REAL NOT NULL,
        distance_to_river_km REAL NOT NULL,
        predicted_risk TEXT NOT NULL,
        prob_low REAL NOT NULL,
        prob_moderate REAL NOT NULL,
        prob_high REAL NOT NULL,
        prob_critical REAL NOT NULL,
        raw_score REAL NOT NULL,
        advisory_action TEXT NOT NULL
    );
    """)

    # Table 1b: Registered Users
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        display_name TEXT,
        phone TEXT,
        department TEXT,
        location TEXT,
        role TEXT,
        auth_provider TEXT DEFAULT 'password',
        created_at TEXT NOT NULL
    );
    """)

    # Safe migrations for existing SQLite databases
    for col in ["phone", "department", "location", "role"]:
        try:
            c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT;")
        except Exception:
            pass

    # Table 2: Model Training Runs
    c.execute("""
    CREATE TABLE IF NOT EXISTS model_training_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trained_at TEXT NOT NULL,
        model_name TEXT NOT NULL,
        training_samples INTEGER NOT NULL,
        testing_samples INTEGER NOT NULL,
        accuracy REAL NOT NULL,
        precision_score REAL NOT NULL,
        recall_score REAL NOT NULL,
        f1_score REAL NOT NULL,
        hyperparameters TEXT
    );
    """)

    # Table 3: Monitoring Sensor Stations
    c.execute("""
    CREATE TABLE IF NOT EXISTS monitoring_stations (
        station_id TEXT PRIMARY KEY,
        station_name TEXT NOT NULL,
        district TEXT NOT NULL,
        state TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        installed_year INTEGER NOT NULL,
        status TEXT NOT NULL
    );
    """)

    # Check if model_used column exists
    c.execute("PRAGMA table_info(disaster_predictions)")
    cols = [r[1] for r in c.fetchall()]
    if "model_used" not in cols:
        try:
            c.execute("ALTER TABLE disaster_predictions ADD COLUMN model_used TEXT DEFAULT 'Random Forest (RF)'")
        except Exception:
            pass

    # Seed initial sensor stations if empty
    c.execute("SELECT COUNT(*) FROM monitoring_stations")
    if c.fetchone()[0] == 0:
        stations = [
            ("STN-SLM-01", "Mettur Dam Hydrological Station", "Salem", "Tamil Nadu", 11.7963, 77.8016, 2021, "Active"),
            ("STN-SLM-02", "Yercaud Hill Slope Monitor", "Salem", "Tamil Nadu", 11.7753, 78.2093, 2022, "Active"),
            ("STN-SLM-03", "Thirumanimutharu River Basin Sensor", "Salem", "Tamil Nadu", 11.6643, 78.1460, 2023, "Active"),
            ("STN-ERD-04", "Bhavani River Confluence Unit", "Erode", "Tamil Nadu", 11.4500, 77.6833, 2020, "Active"),
            ("STN-DMP-05", "Dharmapuri Catchment Perimeter", "Dharmapuri", "Tamil Nadu", 12.1211, 78.1582, 2022, "Active")
        ]
        c.executemany("""
            INSERT INTO monitoring_stations (station_id, station_name, district, state, latitude, longitude, installed_year, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, stations)

    # Seed demonstration historical predictions if empty
    c.execute("SELECT COUNT(*) FROM disaster_predictions")
    if c.fetchone()[0] == 0:
        demo_records = [
            ("2026-09-18 10:30:00", "Mettur Dam Hydrological Station", "Flood", 180.0, 31.0, 88.0, 48.0, 82.0, 6.2, 120.0, 12.0, 0.6, "High", 0.010, 0.200, 0.510, 0.280, 12.96, "Early Warning Issuance & Evacuation Standby"),
            ("2026-09-17 14:15:00", "Yercaud Hill Slope Monitor", "Landslide", 220.0, 24.0, 94.0, 35.0, 92.0, 4.1, 850.0, 32.0, 1.2, "Critical", 0.000, 0.080, 0.320, 0.600, 14.82, "Immediate Emergency Evacuation & Flood/Landslide Protocol"),
            ("2026-09-16 09:00:00", "Thirumanimutharu River Basin Sensor", "Flood", 45.0, 33.0, 58.0, 18.0, 40.0, 1.8, 280.0, 5.0, 4.5, "Low", 0.780, 0.180, 0.030, 0.010, 3.42, "Routine Environmental Monitoring"),
            ("2026-09-15 16:45:00", "Bhavani River Confluence Unit", "Flood", 95.0, 29.0, 72.0, 25.0, 65.0, 3.5, 190.0, 8.0, 1.8, "Moderate", 0.120, 0.620, 0.210, 0.050, 6.88, "Heightened Watch & Preparedness Check"),
            ("2026-09-14 11:20:00", "Dharmapuri Catchment Perimeter", "Drought", 5.0, 38.0, 32.0, 14.0, 15.0, 0.4, 420.0, 3.0, 8.0, "Low", 0.920, 0.060, 0.010, 0.010, 1.15, "Routine Environmental Monitoring"),
            ("2026-09-13 18:10:00", "Mettur Dam Hydrological Station", "Flood", 160.0, 30.0, 84.0, 40.0, 78.0, 5.8, 120.0, 11.0, 0.7, "High", 0.020, 0.240, 0.540, 0.200, 11.85, "Early Warning Issuance & Evacuation Standby")
        ]
        c.executemany("""
            INSERT INTO disaster_predictions (
                created_at, location_name, disaster_type, rainfall_mm, temperature_c, humidity_pct,
                wind_speed_kmh, soil_moisture_pct, water_level_m, elevation_m, slope_deg,
                distance_to_river_km, predicted_risk, prob_low, prob_moderate, prob_high,
                prob_critical, raw_score, advisory_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, demo_records)

    # Seed baseline training run from Appendix B
    c.execute("SELECT COUNT(*) FROM model_training_runs")
    if c.fetchone()[0] == 0:
        c.execute("""
            INSERT INTO model_training_runs (
                trained_at, model_name, training_samples, testing_samples, accuracy,
                precision_score, recall_score, f1_score, hyperparameters
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            "2026-09-20 12:00:00",
            "RandomForestClassifier (n_estimators=250)",
            960,
            240,
            0.6333,
            0.6645,
            0.6333,
            0.6024,
            json.dumps({"n_estimators": 250, "random_state": 42, "class_weight": "balanced", "scaler": "StandardScaler"})
        ))

    conn.commit()
    conn.close()

def save_prediction(pred_dict):
    conn = get_connection()
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    inputs = pred_dict.get("inputs", {})
    probs = pred_dict.get("probabilities", {})
    advisory = pred_dict.get("advisory", {})

    model_used = pred_dict.get("model_used", "Random Forest (RF)")

    c.execute("""
        INSERT INTO disaster_predictions (
            created_at, location_name, disaster_type, rainfall_mm, temperature_c, humidity_pct,
            wind_speed_kmh, soil_moisture_pct, water_level_m, elevation_m, slope_deg,
            distance_to_river_km, predicted_risk, prob_low, prob_moderate, prob_high,
            prob_critical, raw_score, advisory_action, model_used
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        pred_dict.get("location_name", "Salem Station"),
        pred_dict.get("disaster_type", "Flood / Landslide"),
        float(inputs.get("rainfall_mm", 0)),
        float(inputs.get("temperature_c", 0)),
        float(inputs.get("humidity_pct", 0)),
        float(inputs.get("wind_speed_kmh", 0)),
        float(inputs.get("soil_moisture_pct", 0)),
        float(inputs.get("water_level_m", 0)),
        float(inputs.get("elevation_m", 0)),
        float(inputs.get("slope_deg", 0)),
        float(inputs.get("distance_to_river_km", 0)),
        pred_dict.get("predicted_risk", "Moderate"),
        float(probs.get("Low", 0.0)),
        float(probs.get("Moderate", 0.0)),
        float(probs.get("High", 0.0)),
        float(probs.get("Critical", 0.0)),
        float(pred_dict.get("raw_score", 0.0)),
        advisory.get("action", "Standard Monitoring"),
        model_used
    ))

    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id

def log_training_run(metrics):
    conn = get_connection()
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("""
        INSERT INTO model_training_runs (
            trained_at, model_name, training_samples, testing_samples, accuracy,
            precision_score, recall_score, f1_score, hyperparameters
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        metrics.get("model_name", "Random Forest Classifier"),
        int(metrics.get("training_samples", 960)),
        int(metrics.get("testing_samples", 240)),
        float(metrics.get("accuracy", 0.6333)),
        float(metrics.get("precision", 0.6645)),
        float(metrics.get("recall", 0.6333)),
        float(metrics.get("f1_score", 0.6024)),
        json.dumps(metrics.get("feature_importances", {}))
    ))
    new_id = c.lastrowid
    conn.commit()
    conn.close()
    return new_id

def get_recent_predictions(limit=25):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT * FROM disaster_predictions
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows

def execute_raw_sql(sql_query):
    """
    Executes a SQL query against disaster_risk.db and returns columns, rows, count, execution message.
    """
    conn = get_connection()
    c = conn.cursor()
    start_time = datetime.now()
    try:
        c.execute(sql_query)
        if sql_query.strip().upper().startswith("SELECT") or sql_query.strip().upper().startswith("PRAGMA"):
            columns = [desc[0] for desc in c.description] if c.description else []
            rows = [list(row) for row in c.fetchall()]
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000.0
            return {
                "success": True,
                "columns": columns,
                "rows": rows,
                "row_count": len(rows),
                "duration_ms": round(duration_ms, 2),
                "message": f"Query executed successfully ({len(rows)} rows returned in {round(duration_ms, 1)}ms)"
            }
        else:
            conn.commit()
            affected = c.rowcount
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000.0
            return {
                "success": True,
                "columns": ["Result"],
                "rows": [[f"Statement executed successfully ({affected} rows affected)"]],
                "row_count": affected,
                "duration_ms": round(duration_ms, 2),
                "message": f"Operation committed ({affected} rows affected in {round(duration_ms, 1)}ms)"
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "columns": [],
            "rows": [],
            "row_count": 0
        }
    finally:
        conn.close()

def register_user_record(email, password, display_name=None, phone=None, department=None, location=None, role=None, auth_provider="password"):
    import hashlib
    conn = get_connection()
    c = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pwd_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    if not display_name:
        display_name = email.split("@")[0]

    try:
        c.execute("""
            INSERT INTO users (email, password_hash, display_name, phone, department, location, role, auth_provider, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (email.lower().strip(), pwd_hash, display_name, phone or "", department or "", location or "", role or "", auth_provider, now_str))
        user_id = c.lastrowid
        conn.commit()
        return {
            "success": True, 
            "user_id": user_id, 
            "email": email, 
            "display_name": display_name,
            "phone": phone,
            "department": department,
            "location": location,
            "role": role
        }
    except sqlite3.IntegrityError:
        return {"success": False, "error": "User with this email already exists"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        conn.close()

def verify_user_login(email, password):
    import hashlib
    conn = get_connection()
    c = conn.cursor()
    pwd_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    c.execute("SELECT id, email, display_name, phone, department, location, role, auth_provider, created_at FROM users WHERE email = ? AND password_hash = ?", (email.lower().strip(), pwd_hash))
    row = c.fetchone()
    conn.close()
    if row:
        return {"success": True, "user": dict(row)}
    return {"success": False, "error": "Invalid email or password"}

if __name__ == "__main__":
    init_database()
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "query":
            q = sys.argv[2] if len(sys.argv) > 2 else "SELECT * FROM disaster_predictions LIMIT 5"
            res = execute_raw_sql(q)
            print(json.dumps(res))
        elif cmd == "history":
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
            rows = get_recent_predictions(limit)
            print(json.dumps(rows))
        else:
            print(json.dumps({"status": "initialized", "db": DB_PATH}))
    else:
        print("Database initialized at:", DB_PATH)
