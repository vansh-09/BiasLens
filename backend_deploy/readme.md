# BiasLens Backend — Setup Guide

## 📁 Folder Structure
```
backend/
  ├── main.py              ← FastAPI app (entry point)
  ├── config.py            ← Settings & thresholds
  ├── requirements.txt     ← All dependencies
  ├── .env.example         ← Copy to .env and fill keys
  ├── models/
  │   └── schemas.py       ← Request/Response data models
  ├── services/
  │   ├── metrics.py       ← 9 fairness metrics engine
  │   ├── detector.py      ← Audit orchestrator
  │   ├── mitigator.py     ← Fix suggestions
  │   ├── explainer.py     ← Gemini AI explanations
  │   └── reporter.py      ← PDF report generator
  └── utils/
      ├── file_parser.py   ← CSV/JSON/XLSX parser
      └── helpers.py       ← Utility functions
```

## 🚀 Quick Start

### Step 1 — Go into backend folder
```bash
cd backend
```

### Step 2 — Create a Python virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Setup environment variables (optional)
```bash
copy .env.example .env   # Windows
# Then open .env and add your Gemini API key (free at aistudio.google.com)
```

### Step 5 — Run the server
```bash
python main.py
# OR
uvicorn main:app --reload --port 8000
```

### Step 6 — Open API docs
Visit: http://localhost:8000/docs

---

## 📡 API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET  | `/health` | Health check |
| POST | `/api/audit` | Upload dataset → get JSON audit report |
| POST | `/api/audit/report` | Upload dataset → download PDF report |
| POST | `/api/detect-columns` | Auto-detect label + sensitive columns |
| GET  | `/api/metrics/list` | List all 9 fairness metrics |
| GET  | `/api/strategies/list` | List all mitigation strategies |

---

## 🔌 Connecting to Your HTML Frontend

In your `index.html` or `dashboard.html`, call the API like this:

```javascript
async function runAudit(file) {
  const formData = new FormData();
  formData.append('file', file);
  // Optional: specify columns manually
  // formData.append('label_column', 'hired');
  // formData.append('sensitive_attributes', 'gender,race');

  const response = await fetch('http://localhost:8000/api/audit', {
    method: 'POST',
    body: formData
  });

  const result = await response.json();
  console.log('Audit result:', result);
  // result.summary.overall_score → 58
  // result.issues → array of bias issues
  // result.metrics → array of metric results
  // result.mitigation_strategies → how to fix
}
```

---

## 🤖 Fairness Metrics Computed

1. **Disparate Impact** — 80% rule (legal threshold)
2. **Statistical Parity Difference** — Outcome rate gap
3. **Equal Opportunity Difference** — TPR gap
4. **Average Odds Difference** — Combined TPR+FPR gap
5. **Predictive Parity** — Precision gap
6. **Individual Fairness** — Similar people, similar outcomes
7. **Calibration Score** — Prediction calibration across groups
8. **Theil Index** — Inequality measure
9. **Demographic Parity Ratio** — Group rate ratio

---

## 📊 Supported File Formats
- `.csv` — Comma-separated values
- `.json` — JSON array of records
- `.xlsx` / `.xls` — Excel files
- `.parquet` — Parquet files

---

## 🔑 Getting Gemini API Key (Free)
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click "Create API Key"
4. Copy it into your `.env` file as `GEMINI_API_KEY=...`
