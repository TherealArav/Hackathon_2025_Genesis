import os
import streamlit as st
import requests
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file

load_dotenv()

# --- 1. CUSTOM LANGCHAIN RETRIEVER FOR POIS ---
# [Retriever class remains the same as before]
class GoogleMapsPOIRetriever(BaseRetriever):
    """
    Custom LangChain Retriever that:
    1. Takes a user's location (lat, long) and a query (e.g., "restaurants").
    2. Fetches POIs from the Google Maps Places API (Nearby Search).
    3. For each POI, fetches a descriptive snippet from Google Search.
    4. Returns these snippets as LangChain `Document` objects.
    """
    latitude: float
    longitude: float
    maps_api_key: str
    radius: int = 1500  # 1.5km radius
    search_wrapper: GoogleSearchAPIWrapper
    
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
        Uses the GoogleSearchAPIWrapper to get a snippet for a POI.
        """
        st.write(f"--- [Search API] Searching for snippet: '{poi_name} {vicinity}' ---")
        try:
            results = self.search_wrapper.results(f"{poi_name} {vicinity}", num_results=1)
            if results and results[0].get("snippet"):
                return results[0]["snippet"]
            return "No description found."
        except Exception as e:
            st.error(f"Error during Google Search: {e}")
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
                    "source": "google_maps_and_search"
                }
            )
            documents.append(doc)
            
        return documents

# --- 2. HELPER FUNCTION ---
# [Helper function remains the same]
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
# [RAG function remains the same]
@st.cache_data
def get_rag_response(_gemini_key, _maps_key, _search_key, _cse_id, latitude, longitude, query):
    """
    This function initializes all components and runs the RAG chain.
    The underscore prefix on keys tells Streamlit to not hash the keys themselves,
    but to treat them as sensitive and just check if they've changed.
    """
    try:
        # 1. Instantiate LLM (The "G" in RAG)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            google_api_key=_gemini_key
        )
        
        # 2. Instantiate Search Wrapper
        search_wrapper = GoogleSearchAPIWrapper(
            google_api_key=_search_key,
            google_cse_id=_cse_id
        )
        
        # 3. Instantiate Retriever (The "R" in RAG)
        retriever = GoogleMapsPOIRetriever(
            latitude=latitude,
            longitude=longitude,
            maps_api_key=_maps_key,
            search_wrapper=search_wrapper
        )
        
        # 4. Define Prompt Template (The "A" in RAG)
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
        
        # 5. Build the RAG Chain using LCEL
        setup_and_retrieval = RunnableParallel(
            context=(
                RunnablePassthrough()
                | retriever
                | format_docs_for_prompt
            )
        )
        
        rag_chain = setup_and_retrieval | prompt | llm | StrOutputParser()

        # 6. Invoke the Chain and return the result
        return rag_chain.invoke(query)

    except Exception as e:
        st.error(f"An error occurred while running the RAG chain: {e}")
        st.error("Please ensure all API keys are correct and all APIs are enabled in your Google Cloud project.")
        return None

# --- 4. STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("POI Explorer with RAG and Gemini")

# --- 5. HACKATHON-READY SECURITY ---

# We will check for keys in st.secrets first (for deployment)
# and fall back to os.environ (for local testing with .env files)
# For a hackathon, we can also add a simple password.

def get_api_key(key_name: str) -> str:
    """Fetch key from st.secrets or os.environ."""
    if hasattr(st, 'secrets') and key_name in st.secrets:
        return st.secrets[key_name]
    return os.environ.get(key_name, "")

# Simple password protection.
# For deployment, set this password in your st.secrets
# [deployment]
# HACKATHON_PASSWORD = "your_strong_password"
HACKATHON_PASSWORD = get_api_key("HACKATHON_PASSWORD")

# Initialize session state for login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

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
    # This section is now optional if keys are set in st.secrets
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
    st.markdown("Enter your location and what you're looking for:")

    col1, col2 = st.columns(2)
    with col1:
        # Example: Paris
        latitude = st.number_input("Your Latitude", value=48.8566, format="%.4f")
    with col2:
        longitude = st.number_input("Your Longitude", value=2.3522, format="%.4f")

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
                    
                    if st.button("Clear Cache"):
                        st.cache_data.clear()
                        st.success("Cache cleared!")
else:
    st.warning("Please enter the password in the sidebar to use the app.")