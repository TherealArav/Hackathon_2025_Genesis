import os
import streamlit as st
import requests
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Dict, Any
import numpy as np
from scipy.io.wavfile import write as write_wav
import io
import base64

# --- 1. CUSTOM LANGCHAIN RETRIEVER FOR POIS ---
# [Retriever class remains the same as before]
class GoogleMapsPOIRetriever(BaseRetriever):
    """
    Custom LangChain Retriever that:
    4. Returns these snippets as LangChain `Document` objects.
    """
    latitude: float
    longitude: float
    maps_api_key: str
    search_api_key: str
    cse_id: str
    radius: int = 1500  # 1.5km radius
    
    def _get_pois_from_maps(self, query: str) -> List[Dict[str, Any]]:
        """
        REAL FUNCTION: Calls the Google Maps API.
        """
        # We use st.write for logging in Streamlit
        st.write(f"--- [Maps API] Searching for '{query}' near ({self.latitude}, {self.longitude}) ---")
        
        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{self.latitude},{self.longitude}",
            "radius": self.radius,
            "keyword": query,
            "key": self.maps_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()
            
            if data.get("status") == "OK":
                return data.get("results", [])
            else:
                st.error(f"Maps API Error: {data.get('status')} - {data.get('error_message', '')}")
                return []
        except requests.exceptions.RequestException as e:
            st.error(f"HTTP Request failed: {e}")
            return []

    def _get_search_snippet(self, poi_name: str, vicinity: str) -> str:
        """
        NEW FUNCTION: Uses direct 'requests' to call the Google Search API.
        This bypasses the LangChain wrapper.
        """
        st.write(f"--- [Direct Search API] Searching for snippet: '{poi_name} {vicinity}' ---")
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.search_api_key,
                "cx": self.cse_id,
                "q": f"{poi_name} {vicinity}",
                "num": 1
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Check if 'items' exists and has at least one result
            if "items" in data and len(data["items"]) > 0:
                snippet = data["items"][0].get("snippet", "No description found.")
                st.write(f"--- [Direct Search API] Found: {snippet} ---")
                return snippet
            else:
                st.write(f"--- [Direct Search API] No items found for query. ---")
                return "No description found."
        except requests.exceptions.RequestException as e:
            st.error(f"HTTP Request to Search API failed: {e}")
            return "Error fetching description."
        except Exception as e:
            st.error(f"Error parsing Search API response: {e}")
            return "Error fetching description."

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """
        The main method for the retriever.
        """
        # 1. Retrieve POIs from Google Maps
        pois = self._get_pois_from_maps(query)
        
        documents = []
        for poi in pois:
            poi_name = poi.get("name", "Unknown Place")
            vicinity = poi.get("vicinity", "")
            
            # 2. Augment: For each POI, get a descriptive snippet
            snippet = self._get_search_snippet(poi_name, vicinity)
            
            # 3. Create a LangChain Document
            doc = Document(
                page_content=snippet,
                metadata={
                    "poi_name": poi_name,
                    "address": vicinity,
                    "source": "google_maps_and_direct_search"
                }
            )
            documents.append(doc)
            
        return documents

# --- 2. HELPER FUNCTION ---
def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Formats the retrieved documents into a clean string for the prompt.
    """
    if not docs:
        return "No information found for that query."
        
    return "\n\n".join(
        f"**{doc.metadata.get('poi_name', 'Unknown Place')}**\nAddress: {doc.metadata.get('address', 'N/A')}\nSummary: {doc.page_content}"
        for doc in docs
    )

# --- 3. CACHED RAG CHAIN FUNCTION ---
# [get_rag_response function remains the same as before]
@st.cache_data
def get_rag_response(_gemini_key, _maps_key, _search_key, _cse_id, latitude, longitude, query):
    """
    This function initializes all components and runs the RAG chain.
    """
    try:
        # 1. Instantiate LLM (The "G" in RAG)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=_gemini_key
        )
        
        # 2. Instantiate Retriever (The "R" in RAG)
        # We NO LONGER create the SearchWrapper. We pass the keys directly.
        retriever = GoogleMapsPOIRetriever(
            latitude=latitude,
            longitude=longitude,
            maps_api_key=_maps_key,
            search_api_key=_search_key,
            cse_id=_cse_id
        )
        
        # 3. Define Prompt Template (The "A" in RAG)
        template = """
        You are a friendly and engaging AI tour guide.
        Your task is to provide a short, exciting summary of Points of Interest (POIs) near the user, based *only* on the context provided.
        
        Do not make up any information. If the context is empty, say so.
        
        CONTEXT ABOUT NEARBY PLACES:
        {context}
        
        YOUR TASK:
        Write a brief, one-paragraph summary for the user, highlighting the places from the context.
        Start with a friendly greeting!
        """
        prompt = PromptTemplate.from_template(template)
        
        # 4. Build the RAG Chain using LCEL
        setup_and_retrieval = RunnableParallel(
            context=(
                RunnablePassthrough()
                | retriever
                # The 'format_docs_for_prompt' helper is now implicitly called inside the chain
                | format_docs_for_prompt
            )
        )
        
        rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

        # 5. Invoke the Chain and return the result
        return rag_chain.invoke(query)

    except Exception as e:
        st.error(f"An error occurred while running the RAG chain: {e}")
        st.error("Please ensure all API keys are correct and all APIs are enabled in your Google Cloud project.")
        return None

# --- 4. NEW: CACHED TTS FUNCTION ---
@st.cache_data
def get_tts_audio(text: str, _gemini_key: str) -> bytes | None:
    """
    Calls the Gemini 2.5 Flash TTS model to generate audio.
    Returns the audio data as WAV bytes.
    """
    st.write(f"--- [Gemini TTS] Generating audio for: '{text[:30]}...' ---")
    
    # The API returns raw 16-bit PCM data at 24000 Hz
    SAMPLE_RATE = 24000
    
    # Use the v1beta endpoint for TTS
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-tts:generateContent?key={_gemini_key}"
    
    payload = {
        "contents": [{
            "parts": [{ "text": text }]
        }],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": { "voiceName": "Kore" } # A friendly, firm voice
                }
            }
        },
        "model": "gemini-2.5-flash-preview-tts"
    }

    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        result = response.json()
        
        # Extract the base64 encoded audio data
        part = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0]
        if "inlineData" not in part:
            st.error(f"TTS API Error: No 'inlineData' in response. {result}")
            return None

        audio_data_base64 = part["inlineData"]["data"]
        
        # Decode the base64 string to raw PCM bytes
        pcm_data = base64.b64decode(audio_data_base64)
        
        # Convert the raw bytes to a NumPy array of 16-bit integers
        pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
        
        # Create an in-memory WAV file
        wav_buffer = io.BytesIO()
        write_wav(wav_buffer, SAMPLE_RATE, pcm_array)
        wav_buffer.seek(0)
        
        st.write("--- [Gemini TTS] Audio generated successfully. ---")
        return wav_buffer.read()

    except requests.exceptions.RequestException as e:
        st.error(f"HTTP Request to TTS API failed: {e}")
        return None
    except Exception as e:
        st.error(f"Error processing TTS audio: {e}")
        return None

# --- 5. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("🤖 AI POI Tour Guide")

# --- 5. HACKATHON-READY SECURITY ---

def get_api_key(key_name: str) -> str:
    """Fetch key from st.secrets or os.environ."""
    if hasattr(st, 'secrets') and key_name in st.secrets:
        return st.secrets[key_name]
    return os.environ.get(key_name, "")

HACKATHON_PASSWORD = get_api_key("HACKATHON_PASSWORD")

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "summary" not in st.session_state:
    st.session_state.summary = "" # Initialize summary in session state

if HACKATHON_PASSWORD:
    # If a password is set, show login
    password_guess = st.sidebar.text_input("Enter Judge/Guest Password", type="password")
    if st.sidebar.button("Login"):
        if password_guess == HACKATHON_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.sidebar.error("Incorrect password.")
else:
    # If no password is set, just run the app (good for local dev)
    st.session_state.authenticated = True

# --- 6. MAIN APPLICATION (Only if authenticated) ---

if st.session_state.authenticated:
    st.sidebar.success("Authenticated!")
    st.subheader("Your RAG-powered guide to nearby places")

    # --- API Key Management (Sidebar) ---
    st.sidebar.header("API Key Configuration (Optional Override)")
    st.sidebar.markdown(
        "Keys can be set in `st.secrets` for deployment. "
        "You can override them here for local testing."
    )

    # Use session_state to hold keys, initialized from st.secrets or env
    if "GOOGLE_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")
    if "GOOGLE_MAPS_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_MAPS_API_KEY = get_api_key("GOOGLE_MAPS_API_KEY")
    if "GOOGLE_SEARCH_API_KEY" not in st.session_state:
        st.session_state.GOOGLE_SEARCH_API_KEY = get_api_key("GOOGLE_SEARCH_API_KEY")
    if "GOOGLE_CSE_ID" not in st.session_state:
        st.session_state.GOOGLE_CSE_ID = get_api_key("GOOGLE_CSE_ID")

    st.session_state.GOOGLE_API_KEY = st.sidebar.text_input(
        "Gemini API Key", 
        value=st.session_state.GOOGLE_API_KEY, 
        type="password"
    )
    st.session_state.GOOGLE_MAPS_API_KEY = st.sidebar.text_input(
        "Google Maps API Key", 
        value=st.session_state.GOOGLE_MAPS_API_KEY, 
        type="password"
    )
    st.session_state.GOOGLE_SEARCH_API_KEY = st.sidebar.text_input(
        "Google Search API Key", 
        value=st.session_state.GOOGLE_SEARCH_API_KEY, 
        type="password"
    )
    st.session_state.GOOGLE_CSE_ID = st.sidebar.text_input(
        "Google Search CSE ID", 
        value=st.session_state.GOOGLE_CSE_ID, 
        type="password"
    )

    # Check if all keys are provided
    all_keys_provided = all([
        st.session_state.GOOGLE_API_KEY,
        st.session_state.GOOGLE_MAPS_API_KEY,
        st.session_state.GOOGLE_SEARCH_API_KEY,
        st.session_state.GOOGLE_CSE_ID
    ])

    if all_keys_provided:
        st.sidebar.success("All API keys are configured.")
    else:
        st.sidebar.error("Please provide all 4 API keys/IDs.")

    # --- Main Application Area ---
    st.markdown("Enter your coordinates manually, or use the defaults for Dubai.")

    # --- RE-ADDED: Manual Latitude and Longitude Input ---
    default_location = {"latitude": 25.2048, "longitude": 55.2708}

    col1, col2 = st.columns(2)
    with col1:
        latitude = st.number_input(
            "Latitude",
            value=default_location["latitude"],
            format="%.4f"
        )
    with col2:
        longitude = st.number_input(
            "Longitude",
            value=default_location["longitude"],
            format="%.4f"
        )
    
    st.success(f"Using location: ({latitude}, {longitude})")
    query = st.text_input("What are you looking for?", "museums")

    if st.button("Explore Nearby!"):
        if not all_keys_provided:
            st.error("Please configure all API keys in the sidebar first.")
        else:
            with st.spinner("Finding POIs, augmenting with data, and generating a summary..."):
                # Call the cached function
                response = get_rag_response(
                    st.session_state.GOOGLE_API_KEY,
                    st.session_state.GOOGLE_MAPS_API_KEY,
                    st.session_state.GOOGLE_SEARCH_API_KEY,
                    st.session_state.GOOGLE_CSE_ID,
                    latitude,
                    longitude,
                    query
                )
                
                if response:
                    st.markdown("### Your AI Tour Guide Summary")
                    st.markdown(response)
                    st.session_state.summary = response # Store summary for TTS
                    
                    if st.button("Clear Cache"):
                        st.cache_data.clear()
                        st.success("Cache cleared!")
                        st.session_state.summary = "" # Clear summary on cache clear
                else:
                    st.session_state.summary = "" # Clear summary on error

    # --- NEW: TTS Playback Section ---
    if st.session_state.summary:
        st.divider()
        if st.button("Click to Listen to Summary"):
            with st.spinner("Generating audio..."):
                audio_bytes = get_tts_audio(
                    st.session_state.summary, 
                    st.session_state.GOOGLE_API_KEY
                )
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.error("Sorry, I could not generate the audio for that summary.")
else:
    st.warning("Please enter the password in the sidebar to use the app.")

 