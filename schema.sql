-- ==========================================================
-- ML-BASED DISASTER RISK PREDICTION SYSTEM
-- Database Schema (SQLite / SQL)
-- Hindusthan College of Technology, Salem-636309
-- Department of Artificial Intelligence and Data Science
-- Project By: Tamilarasu K, Nandakishore G R, Sivakumar S
-- Guide: Dr. R. Divya, Assistant Professor
-- ==========================================================

-- 1. Table: Disaster Risk Predictions Log
CREATE TABLE IF NOT EXISTS disaster_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location_name VARCHAR(150) NOT NULL,
    disaster_type VARCHAR(50) NOT NULL,
    rainfall_mm DECIMAL(6,2) NOT NULL,
    temperature_c DECIMAL(4,2) NOT NULL,
    humidity_pct DECIMAL(5,2) NOT NULL,
    wind_speed_kmh DECIMAL(5,2) NOT NULL,
    soil_moisture_pct DECIMAL(5,2) NOT NULL,
    water_level_m DECIMAL(5,2) NOT NULL,
    elevation_m DECIMAL(6,2) NOT NULL,
    slope_deg DECIMAL(4,2) NOT NULL,
    distance_to_river_km DECIMAL(5,2) NOT NULL,
    predicted_risk VARCHAR(20) NOT NULL,     -- Low, Moderate, High, Critical
    prob_low DECIMAL(5,4) NOT NULL,
    prob_moderate DECIMAL(5,4) NOT NULL,
    prob_high DECIMAL(5,4) NOT NULL,
    prob_critical DECIMAL(5,4) NOT NULL,
    raw_score DECIMAL(6,3) NOT NULL,
    advisory_action TEXT NOT NULL,
    model_used VARCHAR(80) DEFAULT 'Random Forest (RF)'
);

-- 2. Table: Machine Learning Model Training History
CREATE TABLE IF NOT EXISTS model_training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(100) NOT NULL,
    training_samples INT NOT NULL,
    testing_samples INT NOT NULL,
    accuracy DECIMAL(6,4) NOT NULL,
    precision_score DECIMAL(6,4) NOT NULL,
    recall_score DECIMAL(6,4) NOT NULL,
    f1_score DECIMAL(6,4) NOT NULL,
    hyperparameters TEXT
);

-- 3. Table: Environmental Monitoring Sensor Stations
CREATE TABLE IF NOT EXISTS monitoring_stations (
    station_id VARCHAR(30) PRIMARY KEY,
    station_name VARCHAR(150) NOT NULL,
    district VARCHAR(60) NOT NULL,
    state VARCHAR(60) NOT NULL,
    latitude DECIMAL(9,6) NOT NULL,
    longitude DECIMAL(9,6) NOT NULL,
    installed_year INT NOT NULL,
    status VARCHAR(20) DEFAULT 'Active'
);

-- Sample Analytical Queries for Viva / College Demo:
-- Query 1: Distribution of risk levels
-- SELECT predicted_risk, count(*) AS count, AVG(rainfall_mm) as avg_rainfall, AVG(water_level_m) as avg_water_level FROM disaster_predictions GROUP BY predicted_risk;

-- Query 2: High or Critical risk events
-- SELECT id, location_name, disaster_type, predicted_risk, rainfall_mm, water_level_m, advisory_action FROM disaster_predictions WHERE predicted_risk IN ('High', 'Critical') ORDER BY id DESC;

-- Query 3: Latest training performance metrics
-- SELECT model_name, accuracy, precision_score, recall_score, f1_score, trained_at FROM model_training_runs ORDER BY id DESC LIMIT 1;
