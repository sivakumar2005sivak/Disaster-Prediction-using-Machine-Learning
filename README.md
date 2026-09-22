# 🛡️ Remix ML-Based Disaster Risk Prediction System

An advanced, full-stack Machine Learning disaster risk assessment and early warning platform built with **Python (Scikit-Learn/Pure NumPy ML Pipeline)**, **Express & TypeScript Backend**, **Vite Frontend**, and **Firebase Firestore Cloud Persistence**.

---

## 🚀 Quick Start (Run Locally in 3 Steps)

### 📋 Prerequisites
Ensure you have the following installed on your machine:
- **Node.js**: Version 18.0.0 or higher ([Download Node.js](https://nodejs.org/))
- **Python**: Version 3.10 or higher with `pip` ([Download Python](https://www.python.org/))
- **Git** (optional, for version control)

---

### Step 1: Install Python Dependencies
Open your terminal inside the project folder:
```bash
pip install -r requirements.txt
```
> *Dependencies installed: `numpy`, `scipy`, `scikit-learn`*

---

### Step 2: Install Node.js Dependencies
```bash
npm install
```
> *Dependencies installed: `express`, `vite`, `tsx`, `firebase`, `tailwindcss`, `esbuild`*

---

### Step 3: Launch the Development Server
```bash
npm run dev
```

Once started, open your browser and navigate to:
```
http://localhost:3000
```

---

## 📦 How to Transfer the Code to Another Machine

### Option A: Direct ZIP Download from Google AI Studio
1. Open the project in **Google AI Studio Build**.
2. Click the **Settings (⚙️)** gear icon in the top right.
3. Select **"Export to ZIP"** (or **"Download Project"**).
4. Extract the ZIP file onto your target machine and run the 3 steps above!

### Option B: Push & Clone with GitHub
1. In Google AI Studio Build, click **Settings (⚙️)** -> **"Export to GitHub"**.
2. On your target computer, clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   pip install -r requirements.txt
   npm install
   npm run dev
   ```

---

## 🏗️ Architecture & How It Works

| Component | Technology | Description |
|---|---|---|
| **Frontend** | HTML5, Tailwind CSS v4, TypeScript | Responsive real-time meteorological command center with interactive parameter sliders, scenario presets, risk telemetry meters, and live sorting data logs. |
| **Backend API** | Node.js Express (`server.ts`) | Handles routing for `/api/predict`, `/api/compare`, `/api/metrics`, `/api/train`, `/api/history`, and `/api/auth/*`. Spawns child Python processes for ML execution. |
| **ML Engine** | Python (`ml_engine.py`) | Dual candidate models: **Random Forest** (n=35 bagging trees) and **XGBoost** (gradient boosted trees) with feature scaling and multi-class probability outputs. |
| **Local DB** | SQLite (`database.py`, `disaster_risk.db`) | Automatic persistent storage for user accounts, historical prediction logs, and model training metrics. |
| **Cloud DB** | Firebase Firestore (`src/firebase.ts`, `src/auth.ts`) | Cloud synchronization of officer profiles and disaster assessment outputs tied to user UIDs. |

---

## ⚙️ Available npm Scripts

- `npm run dev`: Boots the development server with live Vite middleware on port 3000 (`tsx server.ts`).
- `npm run build`: Bundles the Vite frontend into `dist/` and compiles `server.ts` to `dist/server.cjs` via `esbuild`.
- `npm start`: Runs the standalone production build using `node dist/server.cjs`.
- `npm run lint`: Runs TypeScript validation (`tsc --noEmit`).

---

## 🔑 Authentication Details
- **Email / Password**: Register on the landing page with your basic details (Name, Phone, Department, District, Role). Credentials are stored and you are automatically redirected to the Login page.
- **Google Sign-In**: Click *"Continue with Google"* to authenticate directly using Firebase Auth with zero setup required.
