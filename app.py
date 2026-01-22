import os
import streamlit as st
import requests
import time
import csv
import numpy as np
import io
import base64
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from geopy.distance import great_circle
from dotenv import load_dotenv

# LangChain Imports
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from scipy.io.wavfile import write as write_wav

# Geolocation Component
from streamlit_geolocation import streamlit_geolocation

load_dotenv()

# --- 1. ROBUST RETRIEVER ---
class GoogleMapsPOIRetriever(BaseRetriever):
    user_latitude: float
    user_longitude: float
    maps_api_key: str
    search_api_key: str
    cse_id: str
    radius: int = 1500
    map_latency: float = 0.0
    search_latencies: List[float] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_pois_from_maps(self, query: str) -> List[Dict[str, Any]]:
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{self.user_latitude},{self.user_longitude}",
            "radius": self.radius,
            "keyword": query,
            "key": self.maps_api_key,
        }
        try:
            start = time.time()
            response = requests.get(url, params=params, timeout=10)
            self.map_latency = time.time() - start
            res = response.json()
            return res.get("results", []) if res.get("status") == "OK" else []
        except: return []

    def _get_search_snippet(self, poi_name: str, vicinity: str) -> str:
        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.search_api_key, "cx": self.cse_id, "q": f"{poi_name} {vicinity}", "num": 1}
        try:
            start = time.time()
            response = requests.get(url, params=params, timeout=5)
            self.search_latencies.append(time.time() - start)
            res = response.json()
            return res["items"][0].get("snippet", "No info.") if "items" in res else "No info."
        except: return "Search error."

    def _get_relevant_documents(self, query: str) -> List[Document]:
        self.map_latency = 0.0
        self.search_latencies = []
        pois = self._get_pois_from_maps(query)
        docs = []
        user_loc = (self.user_latitude, self.user_longitude)

        for poi in pois[:5]: 
            loc = poi["geometry"]["location"]
            p_lat, p_lng = loc["lat"], loc["lng"]
            dist = f"{great_circle(user_loc, (p_lat, p_lng)).km:.2f}"
            snippet = self._get_search_snippet(poi.get("name"), poi.get("vicinity"))
            docs.append(Document(
                page_content=snippet,
                metadata={
                    "poi_name": poi.get("name"),
                    "address": poi.get("vicinity"),
                    "distance_km": dist,
                    "latitude": p_lat,
                    "longitude": p_lng
                }
            ))
        return docs

# --- 2. THE RAG "BRAIN" ---
@st.cache_data(show_spinner=False)
def get_rag_response(_query, _lat, _lon, _keys):
    retriever = GoogleMapsPOIRetriever(
        user_latitude=_lat, user_longitude=_lon,
        maps_api_key=_keys["GOOGLE_MAPS_API_KEY"],
        search_api_key=_keys["GOOGLE_SEARCH_API_KEY"],
        cse_id=_keys["GOOGLE_CSE_ID"]
    )
    
    # Updated to the correct model name for this environment
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-09-2025", 
        google_api_key=_keys["GOOGLE_API_KEY"]
    )
    
    template = """You are a helpful local guide. Summarize these places based on the context:
    Context: {context}
    User Query: {question}"""
    
    prompt = PromptTemplate.from_template(template)
    docs = retriever.invoke(_query)
    
    formatted_context = "\n".join([f"{d.metadata['poi_name']} ({d.metadata['distance_km']}km): {d.page_content}" for d in docs])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"context": formatted_context, "question": _query})
    
    map_data = [{"lat": d.metadata["latitude"], "lon": d.metadata["longitude"]} for d in docs]
    return summary, map_data

# --- 3. UI IMPLEMENTATION ---
st.set_page_config(page_title="Geo-AI Guide", layout="wide")
st.title("📍 Local AI Explorer")

# Initialize State
if "user_lat" not in st.session_state: st.session_state.user_lat = 25.2048
if "user_lon" not in st.session_state: st.session_state.user_lon = 55.2708
if "auth" not in st.session_state: st.session_state.auth = False

# Sidebar Password
with st.sidebar:
    pw = st.text_input("Hackathon Password", type="password")
    if pw == os.environ.get("HACKATHON_PASSWORD"):
        st.session_state.auth = True
        st.success("Authenticated")

if st.session_state.auth:
    st.write("### 1. Set Location")
    
    # Geolocation Expander to avoid blocking the UI
    with st.expander("🛰️ Auto-detect via GPS"):
        location = streamlit_geolocation()
        if location and location.get("latitude"):
            if round(st.session_state.user_lat, 4) != round(location["latitude"], 4):
                st.session_state.user_lat = location["latitude"]
                st.session_state.user_lon = location["longitude"]
                st.success("GPS Location Synced!")
                st.rerun()

    c1, c2 = st.columns(2)
    lat_input = c1.number_input("Latitude", value=float(st.session_state.user_lat), format="%.6f")
    lon_input = c2.number_input("Longitude", value=float(st.session_state.user_lon), format="%.6f")
    
    # Update state only if manually changed
    if lat_input != st.session_state.user_lat or lon_input != st.session_state.user_lon:
        st.session_state.user_lat = lat_input
        st.session_state.user_lon = lon_input

    query = st.text_input("Search Nearby (e.g., 'Best Parks')", "Historic landmarks")

    if st.button("🔍 Run Exploration"):
        keys = {
            "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
            "GOOGLE_MAPS_API_KEY": os.environ.get("GOOGLE_MAPS_API_KEY"),
            "GOOGLE_SEARCH_API_KEY": os.environ.get("GOOGLE_SEARCH_API_KEY"),
            "GOOGLE_CSE_ID": os.environ.get("GOOGLE_CSE_ID")
        }
        
        with st.spinner("Processing RAG Pipeline..."):
            try:
                summary, map_points = get_rag_response(query, st.session_state.user_lat, st.session_state.user_lon, keys)
                st.divider()
                st.subheader("AI Guide Results")
                st.markdown(summary)
                if map_points:
                    st.map(pd.DataFrame(map_points))
            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("Please enter the authentication password in the sidebar.")