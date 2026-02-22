import streamlit as st


article_typography = """
<style>
    /* Import both Raleway and Vidaloka */
    @import url('https://fonts.googleapis.com/css2?family=Raleway:ital,wght@0,100..900;1,100..900&family=Vidaloka&display=swap');

    /* 2. Set Raleway as the standard text for the main page area */
    section[data-testid="stMain"] {
        font-family: 'Raleway', sans-serif !important;

        -webkit-font-smoothing: antialiased !important;
        -moz-osx-font-smoothing: grayscale !important;
        text-rendering: optimizeLegibility !important;
    }
    
    /* Ensure paragraphs, lists, and standard text elements inherit Raleway */
    section[data-testid="stMain"] p, 
    section[data-testid="stMain"] li, 
    section[data-testid="stMain"] span {
        font-family: 'Raleway', sans-serif !important;

        font-size: 1.15rem !important;      /* Slightly larger base text for articles */
        line-height: 1.7 !important;        /* Generous vertical breathing room */
        letter-spacing: 0.01em !important;  /* Very subtle horizontal spacing */
        margin-bottom: 1.5rem !important;   /* Space between paragraphs */
    }

    /* 3. Set Vidaloka strictly for the titles and headers */
    section[data-testid="stMain"] h1, 
    section[data-testid="stMain"] h2, 
    section[data-testid="stMain"] h3,
    section[data-testid="stMain"] h4 {

        font-family: 'Vidaloka', serif !important;
    
        line-height: 1.2 !important;        /* Tighter line height for multi-line titles */
        letter-spacing: -0.01em !important; /* Pulls characters slightly closer for a premium feel */
        margin-top: 2rem !important;        /* Extra space above headers to separate sections */
        margin-bottom: 1rem !important;     /* Space below headers before text starts */
        text-align: center;
    }

    /* Hide scrollbars globally for Chrome, Safari and Opera */
    ::-webkit-scrollbar {
        display: none;
    }
    /* Hide scrollbars globally for IE, Edge and Firefox */
    * {
        -ms-overflow-style: none;  /* IE and Edge */
        scrollbar-width: none;  /* Firefox */
    }
</style>
"""

st.markdown(article_typography, unsafe_allow_html=True)

st.header("Welcome to GeoNavision")
st.markdown("GeoNavision is an AI powered tool, to find Points of Interest (POIs) near your location. The application was developed, to help people with visually disabilities to locate POIs in new and unknown locations. ")
st.markdown("Traditional mapping application are overwhelming for visually challenged people, especially when it comes to active navigation.  To address this, we created a dynamic Retrial Augmented Generation pipeline with the LangChain framework, to retrieve relevant POIs from a users query by retrieving meta data from Google Maps and Google Search. This approach acts as an addon to existing technologies, which can be further tuned to solve problems. ")
st.markdown("This project initially begin as a vibe coded project, for am Ai hackathon. However, as vibe coded projects are often a one trick pony, we felt to further develop a more robust system, to achieve its full potential from our limited knowledge.")
st.markdown("The app presents the information in a tabular manner, consisting of 10 POIs retrieved from the Maps and Search APIs. The table includes name of POI, Wheelchair accessibility and shortest distance from users location. The LLM generates an audio friendly summary below the contents of the table. The user can generate human like text-to-speech (TTS), powered by an open source model (Kokoro-TTS).  The results are cached into a SQLite database, which stores user inputs and results. ")
st.markdown("The information retrieved, is displayed on an interactive map, which the user can access. The map highlights the POis, which the user can click for further information. An embedded link is created to take the user onto Google Maps, if they wish for directions to the place.")
st.markdown("In conclusion, we aim towards creating system, which enable users with visual disabilities to navigate the world. New technologies like smart glasses are opening the gates for building smarter and more accommodable navigation systems  for users. Lets all come together, to build an accessible world for everyone.  ")






