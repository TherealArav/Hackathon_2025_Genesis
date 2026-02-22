# AI POI Guide   
*A Streamlit App for Hackathon 2025 Genesis*

> **Your personal guide, powered by RAG, LangChain, Google APIs, and Gemini 2.5-Flash!**

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Security & API Protection](#security--api-protection)
- [Setup: Local Development](#setup-local-development)
- [Deployment: Streamlit Cloud](#deployment-streamlit-cloud)
- [Acknowledgements](#acknowledgements)

---

## Overview
This project is a **Streamlit** web application that provides users with summaries of Points of Interest (POIs) near their location, leveraging an API-driven Retrieval-Augmented Generation (RAG) workflow.

The LLM generates a audio friendly summary of POIs queried from the LangChain Document.

---

## Features
- **API-Driven RAG:** Dynamic, real-time POI data using Google Maps and Google Search.
- **Natural Summaries:** Synthesized by Gemini 2.5-Flash LLM (via LangChain).
- **Easy to Use:** Clean Streamlit UI with a simple password gate.
- **Caching:** Fast repeat searches using SQLite for caching.
- **Cloud-ready:** Deploy to Streamlit Community Cloud in minutes.

---

## How It Works

#### 1. Retrieval (R)
- Custom **LangChain BaseRetriever** calls:
  - **Google Maps Places API:** Finds POIs (e.g., "museums" nearby).
  - **Google Search API:** Fetches a description for each POI.
  - Wraps results as LangChain `Document` objects.

#### 2. Augmentation (A)
- Bundles POI details into a context string.

#### 3. Generation (G)
- Sends context to **Gemini 2.5-Flash** via `ChatGoogleGenerativeAI`.
- Model outputs a human-readable summary.

#### 4. Caching
- The previouse requests, sent by users is stored in SQLite DBMS, for caching.

---

## Security & API Protection

**Prevent API abuse and runaway costs:**

### 1. Set Google Cloud API Quotas *(Strongly Recommended)*
- Go to **Google Cloud Console** > your project > **APIs & Services** > **Enabled APIs & services**
- For each API ("Places", "Custom Search", "Vertex AI / Gemini"):
  - Under **Quotas**, set low daily and per-minute limits (e.g., 100/day, 10/minute).
- Set a **Billing Alert** under "Budgets & alerts" (e.g., alert at $1.00 spend).

### 2. Add Password Protection
- **Local:**  
  In `.streamlit/secrets.toml` add  
  ```toml
  HACKATHON_PASSWORD = "your-local-password"
  ```
- **Cloud:**  
  Add secrets (password + all API keys) through Streamlit Cloud **Advanced settings**.

*Users must enter the password to use the app.*

---

## Setup: Local Development

### 1. Clone & Install
```bash
git clone https://github.com/TherealArav/GeoNavision.git
cd GeoNavision
python -m venv venv
source venv/bin/activate  # (On Windows: venv\Scripts\activate)
pip install -r requirements.txt
```

### 2. API Keys & Secrets
1. Create a folder: `.streamlit/`
2. Inside, add a file called `secrets.toml`
3. Add your keys:
    ```toml
    HACKATHON_PASSWORD = "your-local-password"
    GOOGLE_API_KEY = "..."
    GOOGLE_MAPS_API_KEY = "..."
    GOOGLE_SEARCH_API_KEY = "..."
    GOOGLE_CSE_ID = "..."
    ```
4. _Important_: Add `.streamlit/secrets.toml` to `.gitignore`

### 3. Run the App
```bash
streamlit run app.py
```
- Open browser to the provided URL, enter your chosen password.

---

## Deployment: Streamlit Cloud

1. Push code to your public GitHub (except secrets).
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app** > Select your repo.
4. In **Advanced settings**, paste secrets (from your `secrets.toml`) into the Secrets box.
5. Click **Deploy!**

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [Google Maps API](https://developers.google.com/maps/documentation/places/web-service/overview)
- [Google Search API](https://developers.google.com/custom-search/v1/overview)
- [Gemini 2.5](https://cloud.google.com/vertex-ai/docs/generative-ai/learn/models)
- [SQLite](https://sqlite.org/)
- [Kokoro-TTS](https://github.com/thewh1teagle/kokoro-onnx.git)
- [Folium](https://github.com/python-visualization/folium.git)
- [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium.git)
---

