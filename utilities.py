import pandas as pd
import io
from typing import Any

from langchain_core.documents import Document

def store_doc_metadata(docs: list[Document] = []) -> pd.DataFrame:
    """
    Parses a LangChain document to extract the metadata found.
    ---
    Logic:
    1. Parse metadata, from LangChain Document for app functionality and caching.
    5. Store data in a DataFrame for easy manipulation and retrieval. 
    """

    # Extract documents for data if available
    meta_data = []
    for doc in docs:
        
        try:
            meta: dict = doc.metadata
            # Only include metadata if it has valid latitude and longitude values
            if "latitude" not in meta or "longitude" not in meta:
                meta["latitude"] = 0.0
                meta["longitude"] = 0.0
                print(f"DEBUG: Document metadata missing lat/lon, defaulting to 0.0: {meta}\n")
            meta_data.append(meta)
        except Exception as e:
            print(f"Error extracting metadata from document: {e}")
            meta_data.append({})\
    
    meta_df = pd.DataFrame(meta_data) if meta_data else pd.DataFrame()
        
    return meta_df


def check_user_query(query: str) -> bool:
    """
    Validates the user query to ensure it is safe and well-formed.
    ---
    Logic:
    1. Check for empty queries or queries that are too long.
    2. Ensure the query contains only alphanumeric characters and spaces to prevent SQL injection.
    3. Return True if the query is valid, False otherwise.
    """
    if not query or len(query) > 100 or not query.strip() or query is pd.NA:
        return False
    if not all(c.isalnum() or c.isspace() for c in query):
        return False
    return True

def check_user_cords(lat: float, lon: float) -> bool:
    if not isinstance(lat, float) or not isinstance(lon, float):
        return False
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return False
    
    return True


def debug_session_state(lat:float, lon:float, s_lat:float, s_lon:float) -> None:
        print(f"DEBUG: Upd ated user location to ({lat}, {lon})")
        print(f"DEBUG: Updated session location to ({s_lat}, {s_lon})")
        print(f"{"-" * 50}")
    



