# House Price Prediction Model - Deployment Guide

Your XGBoost model is trained and saved as `house_price_model.pkl`. Here are **5 practical deployment options**, from simplest to production-ready.

---

## Option 1: Simple Python Script (Local Use)
**Best for:** Quick predictions on your machine.

**What you need:** `predict.py` (I'll create this)

**Steps:**
1. Run the script from the project folder:
   ```bash
   cd "C:/Users/LENOVO/OneDrive/Desktop/summer training new addion/-Bengaluru-House-Price-Prediction"
   python predict.py
   ```

2. The script loads the model and makes a test prediction.

**Pros:** Simplest, no setup.  
**Cons:** Local only, not accessible to others.

---

## Option 2: Flask Web API (Local or Cloud)
**Best for:** REST API that others can call via HTTP.

**What you need:** `app.py` (Flask server) + `requirements.txt`

**Steps:**
1. Create `app.py` (I'll create this).
2. Install dependencies:
   ```bash
   python -m pip install flask
   ```
3. Run the server:
   ```bash
   python app.py
   ```
4. The API will be available at `http://localhost:5000`

**Example request (using curl or Postman):**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_sqft": 1200,
    "bath": 2,
    "bhk": 2,
    "location": "Whitefield"
  }'
```

**Response:**
```json
{
  "predicted_price": 60.86
}
```

**Pros:** Easy to use, shareable URL, works on any machine with Python.  
**Cons:** Requires server running; not production-grade.

---

## Option 3: Streamlit Web App (Interactive UI)
**Best for:** Non-technical users, visual dashboards.

**What you need:** `streamlit_app.py` (I'll create this)

**Steps:**
1. Install Streamlit:
   ```bash
   python -m pip install streamlit
   ```
2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. Opens in your browser automatically (usually `http://localhost:8501`).

**Features:** Sliders for inputs, instant prediction, nice UI.

**Pros:** Beautiful UI, no coding needed to use.  
**Cons:** Requires Streamlit server running.

---

## Option 4: Deploy to Heroku (Free/Paid Cloud)
**Best for:** Always-on web API, public access.

**What you need:** `app.py` + `requirements.txt` + Heroku account (free tier).

**Steps:**
1. Create Heroku account at https://www.heroku.com
2. Install Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli
3. In project folder:
   ```bash
   heroku login
   heroku create my-house-price-model
   git push heroku main
   ```
4. Your API is now live at `https://my-house-price-model.herokuapp.com/predict`

**Pros:** Always on, public URL, easy deployment.  
**Cons:** Requires git/version control, small cost for production use.

---

## Option 5: AWS / Azure / Google Cloud (Production)
**Best for:** Large-scale deployments, high availability.

**Services:**
- **AWS:** Lambda + API Gateway, or EC2 + Flask
- **Azure:** App Service + Flask, or Azure ML
- **Google Cloud:** Cloud Run, or App Engine

**Pros:** Scalable, professional, best security.  
**Cons:** More complex setup, can have costs.

---

## Recommended Path

**For you right now, I recommend Option 2 or Option 3:**

| Option | Time to Deploy | Ease | Share with Others | Best For |
|--------|---|---|---|---|
| Option 1 | 5 min | ⭐⭐⭐⭐⭐ | ❌ | Personal testing |
| **Option 2 (Flask)** | **15 min** | **⭐⭐⭐⭐** | **✅ (HTTP URL)** | **Simple API** |
| **Option 3 (Streamlit)** | **10 min** | **⭐⭐⭐⭐⭐** | **✅ (Share link)** | **Quick demo** |
| Option 4 (Heroku) | 30 min | ⭐⭐⭐ | ✅ (Public URL) | Production |
| Option 5 (Cloud) | 1+ hour | ⭐⭐ | ✅ | Enterprise |

---

## Files I Can Create For You

I can create ready-to-run code for any option:

1. **`predict.py`** — Local prediction script
2. **`app.py`** — Flask REST API
3. **`streamlit_app.py`** — Streamlit web UI
4. **`requirements.txt`** — All dependencies
5. **`README_DEPLOY.md`** — Detailed setup for each platform

---

## Next Steps

**Tell me which option you prefer:**
- "Create a Flask API" → I'll make `app.py` + full instructions
- "Create a Streamlit app" → I'll make the interactive UI
- "Create a simple script" → Local predictions
- "All of the above" → I'll create all three + instructions

Once you pick, I'll create the files and give you copy-paste commands to run.

---

## Current Model Summary

- **Model Type:** XGBoost Regressor
- **Saved File:** `house_price_model.pkl`
- **Features:** `total_sqft`, `bath`, `bhk`, one-hot encoded `location` columns, `area_type`
- **Target:** House price (in millions)
- **Performance:** R² = 0.647, RMSE = 92.95
- **Test Accuracy:** ~65% of variance explained
