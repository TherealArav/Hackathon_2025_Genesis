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
            if not meta.latitude or  not meta.longitude:
                meta.latitude = 0.0
                meta.longitude = 0.0
                print(f"DEBUG: Document metadata missing lat/lon, defaulting to 0.0: {meta}\n")
            meta_data.append(meta)
        except Exception as e:
            print(f"Error extracting metadata from document: {e}")
            meta_data.append({})\
    
    meta_df = pd.DataFrame(meta_data) if meta_data else pd.DataFrame()
        
    return meta_df


