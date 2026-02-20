import os
import streamlit as st
import requests
import time
import pandas as pd
from typing import List, Dict, Any
from pydantic import ConfigDict
from geopy.distance import great_circle
from dotenv import load_dotenv
from utilities import utilities
from storage import QueryStorage
from tts_system import KokoroTTS

# LangChain Imports
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

class GoogleMapsPOIRetriever(BaseRetriever):
    user_latitude: float
    user_longitude: float
    maps_api_key: str
    search_api_key: str
    cse_id: str
    radius: int = 1500
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
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            if response.status_code != 200:
                st.error(f"Google API Error ({response.status_code}): {response.text}")
                return []
                
            res = response.json()
            places: dict[str,Any] = res.get("places", [])
            
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
            response = requests.get(url, params=params, timeout=5)
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
        self.map_latency: float = 0.0
        self.search_latencies: list[float] = []
        
        places: dict[str,Any] = self._get_pois_from_places_new(query)
        docs: list[Document] = []
        user_loc: tuple[float,float] = (self.user_latitude, self.user_longitude)

        for poi in places: 
            name: str = poi.get("displayName", {}).get("text", "Unknown Place")
            address: str = poi.get("formattedAddress", "No address available")
            loc: str = poi.get("location", {})
            p_lat, p_lng = loc.get("latitude"), loc.get("longitude")
            
            acc:dict[dict[str,Any]] = poi.get("accessibilityOptions", {})
            wheelchair_entrance: bool = acc.get("wheelchairAccessibleEntrance", "Unknown")
            wheelchair_restroom: bool = acc.get("wheelchairAccessibleRestroom", "Unknown")
            
            if p_lat and p_lng:
                dist = f"{great_circle(user_loc, (p_lat, p_lng)).km:.2f}"
                snippet = self._get_search_snippet(name, address)
                
                # Combine search snippet with accessibility facts for the LLM
                content = f"Name: {name}. Distance: {dist}km. {snippet} Accessibility Info - Entrance: {wheelchair_entrance}, Restroom: {wheelchair_restroom}."
                
                # Compile Contents and Metadata into the Document
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "poi_name": name,
                        "address": address,
                        "distance_km": dist,
                        "latitude": p_lat,
                        "longitude": p_lng,
                        "wheelchair": (wheelchair_entrance,wheelchair_restroom)
                    }
                ))
        return docs


# Utility Functions
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
            transition: all 0.2s ease;
        }
        </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def apply_df_styles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies conditional formatting and structural styling to the POI DataFrame.
    
    Refinements made:
    1. Uses a more efficient dictionary-based color mapping.
    2. Simplifies logic for "Unknown", "Yes/True", and "No/False".
    3. Optimized for display in st.dataframe or st.table.
    """

    # Check for empty DataFrame to avoid errors when applying styles
    if df.empty:
        return df

    def get_status_style(val):
        val_str = str(val)
        
        # Dictionary mapping for quick style lookups
        styles = {
            "Unknown":'color: #FFFF00;', # Yellow
            "Yes":    'color: ##CCFF00;', # Green
            "True":    'color: #66FF00 ;', # Green
            "No":      'color: #FF0000;', # Red
            "False":   'color: #FF0000;'  # Red
        }
        
        # Search for keywords in the cell value
        for key, style in styles.items():
            if key in val_str:
                return style
        return ''

    # Apply the status coloring
    styled_df = df.style.map(get_status_style, subset=["Wheelchair Accessibility"])

    return styled_df

def is_rate_limit() -> bool:
    """
    Checks if the user is clicking too fast.
    Returns True if limited, False if okay to proceed.
    """

    COOLDOWN_TIME: int = 5 # seconds
    current_time = time.time()
    last_time: time = st.session_state.get("last_click_time")

    if current_time - last_time < COOLDOWN_TIME:
        time_left = int(COOLDOWN_TIME - (current_time - st.session_state.last_click_time))
        st.warning(f"Please wait {time_left} seconds before clicking again.")
        return True
    
    st.session_state.last_click_time = current_time
    return False

def clear_results() -> None:
    """
    Clear summary, docs and cache from session state
    """
    st.session_state.summary = ""
    st.session_state.docs = []
    st.session_state.cache = {}

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
        model="gemini-2.5-flash", 
        google_api_key=_keys["GOOGLE_API_KEY"]
    )
    
    audio_prompt_template = """
ROLE:
You are an expert Audio Description Specialist and Accessibility Consultant. Your goal is to convert technical point-of-interest data into a warm, helpful, and natural-sounding audio briefing for a user with visual or mobility impairments.

TASK:
Write a spoken-word script (under 150 words) summarizing the accessibility of the nearby area based on the provided data.

CONSTRAINTS FOR TTS CLARITY:
1. NO MARKDOWN: Do not use asterisks, bolding, bullet points, or tables.
2. PHONETIC FRIENDLY: Avoid complex abbreviations. Write "kilometers" instead of "km".
3. NATURAL FLOW: Use transitional phrases like "If you're looking for," "Alternatively," and "Just a short distance away."
4. HIERARCHY: Mention the closest highly accessible location first.
5. NO SYMBOLS: Do not use symbols like "|" or "/". Use words like "and" or "or".

CONTENT FOCUS:
- Confirm wheelchair entrance accessibility clearly.
- Mention restroom availability if known.
- Provide a brief "vibe" or context if the web snippet includes it.

USER QUERY: {question}

CONTEXT DATA:
{context}

AUDIO SCRIPT:
"""
    
    prompt = PromptTemplate.from_template(audio_prompt_template)
    docs = retriever.invoke(_query)
    
    if not docs:
        return "I couldn't find any specific places matching your search in this area.", []
    
    formatted_context = "\n".join([f"{d.metadata['poi_name']} ({d.metadata['distance_km']}km): {d.page_content}" for d in docs])
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"context": formatted_context, "question": _query})
    
    return summary, docs

@st.cache_resource
def initialize_tts() -> KokoroTTS:
    """Cache the TTS model to avoid reinitialization on every audio play."""
    return KokoroTTS()

# UI IMPLEMENTATION 
st.set_page_config(page_title="Accessibility Guide", layout="wide")
apply_custom_css()
st.title("Local Accessibility Explorer")


# Manage session states
if "user_lat" not in st.session_state: st.session_state.user_lat = 25.1018
if "user_lon" not in st.session_state: st.session_state.user_lon = 55.1628
if "auth" not in st.session_state: st.session_state.auth = False
if "summary" not in st.session_state: st.session_state.summary = ""
if "docs" not in st.session_state: st.session_state.docs = []
if "last_click_time" not in st.session_state: st.session_state.last_click_time = 0
if "cache" not in st.session_state: st.session_state.cache = {}


# Manage authentification
with st.sidebar:
    pw = st.text_input("Password", type="password")
    if pw == os.environ.get("HACKATHON_PASSWORD"):
        st.success("Authenticated")
        st.session_state.auth = True

# Main Application
if st.session_state.auth:
    
    # Input Section for Cordinates and Query
    c1, c2 = st.columns(2)
    lat_input: float = c1.number_input("Latitude", value = 25.1018, format="%.6f")
    lon_input: float = c2.number_input("Longitude",  value =  55.1628, format="%.6f")
    
    # Check if cordinates have changed, if so reset summary, docs and cache to prevent confusion with old data. 
    #  also allows users to quickly change location and run a new search without needing to clear the cache manually each time.
    if lat_input != st.session_state.user_lat or lon_input != st.session_state.user_lon:
        st.session_state.user_lat = lat_input
        st.session_state.user_lon = lon_input

        clear_results()
   
    query: str = st.text_input("Search Nearby", "Super Markets")

    c3,c4,c5 = st.columns(3)
    if c3.button("Run Exploration"): 

        if not utilities.check_user_cords(st.session_state.user_lat, st.session_state.user_lon ):
            st.error("Invalid cordinates. Latitude must be between -90 and 90, Longitude must be between -180 and 180.")
            st.stop()
        
        if not utilities.check_user_query(query):
            st.error("Invalid query. Please enter a valid search term (alphanumeric and spaces only, max 100 characters)")
            st.stop()
        
        if  is_rate_limit():
            st.stop()
        
        clear_results()

        # Cache System -- Connect to local storage and check for nearby cached results before running system.
        connection = QueryStorage()
        st.session_state.cache = connection.find_nearby_query(query_text=query, lat=st.session_state.user_lat, lon=st.session_state.user_lon)
        if st.session_state.cache:
            st.session_state.summary = st.session_state.cache.get("summary", "Unable to generate summary from cache.")
            table_data: list[dict[str,Any]] = st.session_state.cache.get("table_data", [])
            reconstruct_docs: list[Document] = []
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
        # If no cache, run the RAG chain and save results
        # Implementing a simple rate limit to prevent spamming the API while testing

            keys: dict[str,str] = {
                "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
                "GOOGLE_MAPS_API_KEY": os.environ.get("GOOGLE_MAPS_API_KEY"),
                "GOOGLE_SEARCH_API_KEY": os.environ.get("GOOGLE_SEARCH_API_KEY"),
                "GOOGLE_CSE_ID": os.environ.get("GOOGLE_CSE_ID")
            }
            
            try:    
                st.session_state.summary, st.session_state.docs = get_rag_response(query, st.session_state.user_lat, st.session_state.user_lon, keys)
                
                # Save to cache
                df_to_cache = utilities.store_doc_metadata(st.session_state.docs)
                connection.save_query_result(query_text=query, lat=st.session_state.user_lat, lon=st.session_state.user_lon, df=df_to_cache, summary=st.session_state.summary)
            except Exception as e:
                st.error(f"Error: {e}")


    if c4.button("Clear Cache") and st.session_state.cache:
        connection = QueryStorage()
        connection._delete_query_result(query_text=st.session_state.cache.get("query"), lat=st.session_state.cache.get("lat"), lon=st.session_state.cache.get("lon"))
        st.session_state.cache = {}
        st.success("Cache cleared for this query and location.")
    elif st.session_state.cache:
        st.info("Cache exists for this query and location. Click 'Clear Cache' to remove it.")
    else:
        st.info("No cache found for this query and location.")  

    c5_button = c5.button("Play Audio Summary") 

    # Apply styles to DataFrame and display results
    poi_df: pd.DataFrame = utilities.create_df_table(st.session_state.docs)
    poi_df = apply_df_styles(poi_df)

    with st.container(border=True):
        st.subheader("AI Guide Results")
        st.dataframe(poi_df, hide_index=True)
        st.space("small")
        st.subheader("AI Summary")
        st.markdown(st.session_state.summary)
        st.space("small")
        if c5_button:
            tts: KokoroTTS = initialize_tts()
            audio_bytes = tts.generate_audio(st.session_state.summary)
            st.audio(audio_bytes, format="audio/wav")

else:
    st.info("Please enter the password in the sidebar.")