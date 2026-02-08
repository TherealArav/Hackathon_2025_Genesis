import os
import streamlit as st
import requests
import time
import csv
import numpy as np
import io
import base64
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic import Field, ConfigDict
from geopy.distance import great_circle
from dotenv import load_dotenv
from utilities import store_doc_metadata
from storage import QueryStorage

# LangChain Imports
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from scipy.io.wavfile import write as write_wav

# Geolocation & Mapping
from streamlit_geolocation import streamlit_geolocation
import folium
from streamlit_folium import st_folium

load_dotenv()

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

    def _get_pois_from_places_new(self, query: str) -> List[Dict[str, Any]]:
        """
        Uses searchText to find POIs.
        """
        # Endpoint for Text-based searching in the  API
        url = "https://places.googleapis.com/v1/places:searchText"
        
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": self.maps_api_key,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.accessibilityOptions"
        }
        
        # We use locationBias to find results near the user
        payload = {
            "textQuery": query,
            "locationBias": {
                "circle": {
                    "center": {
                        "latitude": self.user_latitude,
                        "longitude": self.user_longitude
                    },
                    "radius": float(self.radius)
                }
            },
            "maxResultCount": 10
        }

        try:
            start = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            self.map_latency = time.time() - start
            
            if response.status_code != 200:
                st.error(f"Google API Error ({response.status_code}): {response.text}")
                return []
                
            res = response.json()
            places = res.get("places", [])
            
            # Debugging: Show count in the console/logs
            return places
        except Exception as e:
            st.error(f"Request Exception: {e}")
            return []

    def _get_search_snippet(self, poi_name: str, vicinity: str) -> str:
        """
        Fetch web context using Google Custom Search
        """

        url = "https://www.googleapis.com/customsearch/v1"
        params = {"key": self.search_api_key, 
                  "cx": self.cse_id, 
                  "q": f"{poi_name} {vicinity}", "num": 1
                  }
        try:
            start = time.time()
            response = requests.get(url, params=params, timeout=5)
            self.search_latencies.append(time.time() - start)
            res = response.json()
            if "items" in res and len(res["items"]) > 0:
                return res["items"][0].get("snippet", "No web info found.")
            
            return "No additional web context found."
        
        except Exception as e:
            return f"Web search error: {str(e)}"

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        Process results into LangChain documents
        """
        self.map_latency = 0.0
        self.search_latencies = []
        
        places = self._get_pois_from_places_new(query)
        docs = []
        user_loc = (self.user_latitude, self.user_longitude)

        for poi in places: 
            name = poi.get("displayName", {}).get("text", "Unknown Place")
            address = poi.get("formattedAddress", "No address available")
            loc = poi.get("location", {})
            p_lat, p_lng = loc.get("latitude"), loc.get("longitude")
            
            acc = poi.get("accessibilityOptions", {})
            wheelchair_entrance = acc.get("wheelchairAccessibleEntrance", "Unknown")
            wheelchair_restroom = acc.get("wheelchairAccessibleRestroom", "Unknown")
            
            if p_lat and p_lng:
                dist = f"{great_circle(user_loc, (p_lat, p_lng)).km:.2f}"
                snippet = self._get_search_snippet(name, address)
                
                # Combine search snippet with accessibility facts for the LLM
                content = f"Name: {name}. Distance: {dist}km. {snippet} Accessibility Info - Entrance: {wheelchair_entrance}, Restroom: {wheelchair_restroom}."
                
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "poi_name": name,
                        "address": address,
                        "distance_km": dist,
                        "latitude": p_lat,
                        "longitude": p_lng,
                        "wheelchair": wheelchair_entrance
                    }
                ))
        return docs

# Utility Functions
def get_directions_url(dest_lat, dest_lon):
    """
    Generate Google Maps directions link
    """
    return f"https://www.google.com/maps/dir/?api=1&destination={dest_lat},{dest_lon}"

def apply_custom_css() -> None:
    """
    Apply custom CSS styles to app
    """
    custom_css = """
        <style>
        /* --- 1. GLOBAL SCROLLBAR REMOVAL --- */
        /* Hide scrollbars globally for Chrome, Safari and Opera */
        ::-webkit-scrollbar {
            display: none;
        }
        /* Hide scrollbars globally for IE, Edge and Firefox */
        * {
            -ms-overflow-style: none;  /* IE and Edge */
            scrollbar-width: none;  /* Firefox */
        }

        /* --- 2. THEME & CONTAINER STYLING --- */
        .main { background-color: #fcfcfc; }
        
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            padding: 10px 24px;
            margin-top: 12px;
            background-color: #FF4B4B;
            color: white;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }

        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# @st.cache_data(show_spinner=False)
def get_rag_response(_query, _lat, _lon, _keys):
    """
    Execute RAG chain and cache results
    """
    retriever = GoogleMapsPOIRetriever(
        user_latitude=_lat, user_longitude=_lon,
        maps_api_key=_keys["GOOGLE_MAPS_API_KEY"],
        search_api_key=_keys["GOOGLE_SEARCH_API_KEY"],
        cse_id=_keys["GOOGLE_CSE_ID"]
    )
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-09-2025", 
        google_api_key=_keys["GOOGLE_API_KEY"]
    )
    
    template = """You are a helpful accessibility consultant. 
    Summarize these places based on the context provided. 
    If a place is listed, identify it clearly. *If no relevant info exists, say no.*
    Generate the summary in a tabular format with columns: Place Name, Distance (km), Accessibility Features, Additional Information.
    
    Context: {context}
    User Query: {question}"""
    
    prompt = PromptTemplate.from_template(template)
    docs = retriever.invoke(_query)
    
    if not docs:
        return "I couldn't find any specific places matching your search in this area.", []
    
    formatted_context = "\n".join([f"{d.metadata['poi_name']} ({d.metadata['distance_km']}km): {d.page_content}" for d in docs])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"context": formatted_context, "question": _query})
    
    return summary, docs

# UI IMPLEMENTATION 
st.set_page_config(page_title="Accessibility Guide", layout="wide")
apply_custom_css()
st.title("Local Accessibility Explorer")

# Manage session states
if "user_lat" not in st.session_state: st.session_state.user_lat = 25.2048
if "user_lon" not in st.session_state: st.session_state.user_lon = 55.2708
if "auth" not in st.session_state: st.session_state.auth = False
if "summary" not in st.session_state: st.session_state.summary = ""
if "docs" not in st.session_state: st.session_state.docs = []

# Manage authentification
with st.sidebar:
    pw = st.text_input("Password", type="password")
    if pw == os.environ.get("HACKATHON_PASSWORD"):
        st.success("Authenticated")
        st.session_state.auth = True

# Main Application
if st.session_state.auth:
    st.write("### 1. Set Location")
    
    # with st.expander("Auto-detect via GPS"):
    #     # Get user geolocation
    #     if st.button("Sync GPS Location"):
    #         location = streamlit_geolocation()
    #         if location and location.get("latitude") and location.get("longitude"):
    #             if round(st.session_state.user_lat, 4) != round(location["latitude"], 4) and round(st.session_state.user_lon, 4) != round(location["longitude"], 4):
    #                 st.session_state.user_lat = location["latitude"]
    #                 st.session_state.user_lon = location["longitude"]
    #                 st.success("GPS Location Synced!")
    #                 st.rerun()
           
    c1, c2 = st.columns(2)
    lat_input = c1.number_input("Latitude", value=float(st.session_state.user_lat), format="%.6f")
    lon_input = c2.number_input("Longitude", value=float(st.session_state.user_lon), format="%.6f")
    
    if lat_input != st.session_state.user_lat or lon_input != st.session_state.user_lon:
        st.session_state.user_lat = lat_input
        st.session_state.user_lon = lon_input

    query = st.text_input("Search Nearby", "Super Markets")

    if st.button("Run Exploration"):
        st.session_state.summary = ""
        st.session_state.docs = []

        # Cache System 
        connection = QueryStorage()
        cahce_result = connection.find_nearby_query(query_text=query, lat=st.session_state.user_lat, lon=st.session_state.user_lon)
        if cahce_result:
            st.session_state.summary = cahce_result.get("summary", "Unable to generate summary from cache.")
            table_data: list[dict[str,Any]] = cahce_result.get("table_data", [])
            reconstruct_docs = []
            for record in table_data:
                reconstruct_docs.append(Document(
                    page_content="",
                    metadata={
                        "poi_name": record.get("poi_name"),
                        "address": record.get("address"),
                        "distance_km": record.get("distance_km"),
                        "latitude": record.get("latitude"),
                        "longitude": record.get("longitude"),
                        "wheelchair": record.get("wheelchair")
                    }
                ))
            st.session_state.docs = reconstruct_docs
            st.success("Loaded results from cache!")
        else:
            keys = {
                "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
                "GOOGLE_MAPS_API_KEY": os.environ.get("GOOGLE_MAPS_API_KEY"),
                "GOOGLE_SEARCH_API_KEY": os.environ.get("GOOGLE_SEARCH_API_KEY"),
                "GOOGLE_CSE_ID": os.environ.get("GOOGLE_CSE_ID")
            }
            
            try:    
                st.session_state.summary, st.session_state.docs = get_rag_response(query, st.session_state.user_lat, st.session_state.user_lon, keys)
                
                # Save to cache
                df_to_cache = store_doc_metadata(st.session_state.docs)
                connection.save_query_result(query_text=query, lat=st.session_state.user_lat, lon=st.session_state.user_lon, df=df_to_cache, summary=st.session_state.summary)
            except Exception as e:
                st.error(f"Error: {e}")

        st.divider()
        st.subheader("AI Guide Results")
        st.markdown(st.session_state.summary)
        st.divider()

        # Map Visualization
        if st.session_state.docs:
            st.subheader("Interactive Map")
            m = folium.Map(location=[st.session_state.user_lat, st.session_state.user_lon], zoom_start=15)

            # Define user location marker
            folium.Marker(
                [st.session_state.user_lat, st.session_state.user_lon],
                popup="Current Position",
                icon=folium.Icon(color="blue", icon="user", prefix="fa")
            ).add_to(m)

            for d in st.session_state.docs:
                maps_link = get_directions_url(d.metadata['latitude'], d.metadata['longitude'])
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <b>{d.metadata['poi_name']}</b><br>
                    Distance: {d.metadata['distance_km']} km<br>
                    Accessibility: {d.metadata['wheelchair']}<br>
                    <a href='{maps_link}' target='_blank'>Get Directions</a>
                </div>
                """
                folium.Marker(
                    [d.metadata['latitude'], d.metadata['longitude']],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(color="orange", icon="location-dot", prefix="fa")
                ).add_to(m)
            
            # Display the map at the center
            st_folium(m,use_container_width= True,height=600,returned_objects=[])


else:
    st.info("Please enter the password in the sidebar.")