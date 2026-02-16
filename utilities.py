import pandas as pd
from typing import Any
import io

from langchain_core.documents import Document

class utilities:
    """
    Utility class for handling various operations
    """

    @staticmethod
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
                meta: dict = doc.metadata.copy() # Use copy to avoid modifying original metadata
                # Only include metadata if it has valid latitude and longitude values
                if "latitude" not in meta or "longitude" not in meta:
                    meta["latitude"] = 0.0
                    meta["longitude"] = 0.0
                meta_data.append(meta)
            except Exception as e:
                meta_data.append({})
        
        meta_df = pd.DataFrame(meta_data) if meta_data else pd.DataFrame()
            
        return meta_df

    @staticmethod
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

    @staticmethod
    def create_df_table(docs: list[Document] = []) -> pd.DataFrame:
        """
        Docstring for create_df_table
        
        :param docs: Document object containing the table data in the metadata
        :type docs: list[Document]
        :return: DataFrame created from the table data in the document's metadata
        :rtype: DataFrame
        """
       # Extract documents for data if available
        meta_data = []
        for doc in docs:
            try:
                # Rearnge metadata for better presentation in the UI, and to ensure only relevant information is displayed.
                meta: dict = doc.metadata.copy() # Use copy to avoid modifying original metadata
                
                meta.pop("latitude",0.0)
                meta.pop("longitude",0.0)

                poi: str = meta.pop("poi_name","Unknown POI")
                addr: str = meta.pop("address","Unknown Address")
                meta["Point of Interest"] = f"{poi} - {addr}"
                
                wheelchair_acc: tuple = meta.pop("wheelchair", ("Unknown", "Unknown"))
                if isinstance(wheelchair_acc, (tuple,list)) and len(wheelchair_acc) == 2:
                    wheelchair_entrance, wheelchair_restroom = wheelchair_acc
                else:
                    wheelchair_entrance, wheelchair_restroom = ("Unknown", "Unknown")
                meta["Wheelchair Accessibility"] = f"Entrance: {wheelchair_entrance}, Restroom: {wheelchair_restroom}"

                distance: float = float(meta.pop("distance_km", "Unknown"))
                meta["Distance (km)"] = {distance}

                meta_data.append(meta)
                
            except Exception as e:
                print(f"Error extracting metadata from document: {e}")
                meta_data.append({
                    "Point of Interest": "Error processing record",
                    "Wheelchair Accessibility": "Unknown",
                    "Distance (km)": "Unknown"
                })
                
        return pd.DataFrame(meta_data) if meta_data else pd.DataFrame()

    



