import streamlit as st


st.header("Welcome to GeoNavision")
st.markdown("GeoNavision is an AI powered tool, to find Points of Interest (POIs) near your location. The application was developed, to help people with visually disabilities to locate POIs in new and unknown locations. ")
st.markdown("Traditional mapping application are overwhelming for visually challenged people, especially when it comes to active navigation.  To address this, we created a dynamic Retrial Augmented Generation pipeline with the LangChain framework, to retrieve relevant POIs from a users query by retrieving meta data from Google Maps and Google Search. This approach acts as an addon to existing technologies, which can be further tuned to solve problems. ")
st.markdown("This project initially begin as a vibe coded project, for am Ai hackathon. However, as vibe coded projects are often a one trick pony, we felt to further develop a more robust system, to achieve its full potential from our limited knowledge.")
st.markdown("The app presents the information in a tabular manner, consisting of 10 POIs retrieved by the Map")
