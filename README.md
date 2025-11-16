Hackathon Project: AI POI Tour Guide (Streamlit App)

This project is a Streamlit application that uses a RAG (Retrieval-Augmented Generation) system to provide users with summaries of Points of Interest (POIs) near their location.

It uses LangChain to structure the logic, Google Maps API to find POIs, Google Search API to get descriptions, and Gemini 2.5-Flash to generate the final summary.

How It Works: A RAG-Based System

This isn't a typical RAG that uses a static vector database. This is a dynamic, API-driven RAG, which is perfect for a hackathon.

Retrieval (The "R"): We built a custom LangChain BaseRetriever called GoogleMapsPOIRetriever. When you "query" this retriever (e.g., with "museums"):

It first calls the Google Maps Places API (Nearby Search) to find POIs (like "Louvre Museum") near the user-provided latitude/longitude.

It then "augments" this by calling the Google Search API for each POI to get a short descriptive snippet.

It bundles all this information into a list of LangChain Document objects.

Augmentation (The "A"): The list of Document objects is formatted into a clean string. This string becomes the {context} that we "stuff" into our prompt.

Generation (The "G"): The final, context-filled prompt is sent to the Gemini 2.5-Flash model (ChatGoogleGenerativeAI). The LLM's job is to synthesize this context into a friendly, human-readable summary, following the instructions in the prompt.

Caching: The entire RAG chain function is wrapped in Streamlit's @st.cache_data. This means if you search for the same query at the same location, the app will return the result instantly from its cache instead of re-calling all the APIs.

🔒 Securing Your App for Deployment (Hackathon Guide)

Exposing an app that uses paid APIs is risky. Here are the two most important steps to protect yourself from abuse and high bills.

1. (Most Important) Set Google Cloud API Quotas

This is your best safety net. It tells Google "No matter what, do not charge me for more than X requests."

Go to the Google Cloud Console.

Select the project you used to create your API keys.

In the navigation menu, go to "APIs & Services" > "Enabled APIs & services".

You will see a list of APIs you enabled. Click on one, for example, "Places API".

Go to the "Quotas" tab.

You will see a list of quotas like "Nearby Search requests per minute". Click the pencil (edit) icon.

Set a low limit. For a hackathon, you could set "Requests per day" to 100 and "Requests per minute" to 10. This is more than enough for judging but prevents a bad actor from running up a bill.

Repeat this process for your other APIs:

"Custom Search API": Set quotas (e.g., 100 queries per day).

"Vertex AI" / "Gemini": Find the "Quotas" page for the Gemini models (e.g., "Generative Language API") and set a low daily limit (e.g., 100 requests per day).

Set a Billing Alert: Go to "Billing" > "Budgets & alerts" and create a budget. Set an alert to email you if your project bill exceeds $1.00.

2. Add a Simple Password (Code Included)

The updated app.py now includes a simple password check. This prevents casual users or bots from accessing the app at all.

To use this:

For Local Development: Create a file at .streamlit/secrets.toml and add:

HACKATHON_PASSWORD = "your-local-password"
# ...add your API keys here too




For Streamlit Cloud Deployment: When you deploy your app, go to "Advanced settings..." and add your secrets. This is where you will set the password for the judges:

# Secrets for Streamlit Cloud
GOOGLE_API_KEY = "your_gemini_key_here"
GOOGLE_MAPS_API_KEY = "your_maps_key_here"
GOOGLE_SEARCH_API_KEY = "your_search_key_here"
GOOGLE_CSE_ID = "your_cse_id_here"

HACKATHON_PASSWORD = "a-secure-password-for-the-judges"




Now, only people with this password can use your app.

🚀 How to Run This Project

1. Install Dependencies

Create a Python virtual environment and install the required packages.

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt




2. Set Up Local Secrets

For local development, Streamlit uses a secrets.toml file.

Create a folder named .streamlit in your project's root directory.

Inside that folder, create a file named secrets.toml.

Copy the contents of the secrets.toml file I generated for you, and fill in your actual API keys and a test password.

IMPORTANT: Add .streamlit/secrets.toml to your .gitignore file so you never commit your keys to GitHub.

3. Run the Streamlit Application

streamlit run app.py



Your browser will open. You will be prompted to enter the password you set in your secrets.toml file.

4. Deploy to Streamlit Cloud

Push your code (including app.py, requirements.txt, README.md but NOT .streamlit/secrets.toml) to a public GitHub repository.

Go to Streamlit Community Cloud and link your GitHub account.

Click "New app", select your repository, and before deploying, click "Advanced settings...".

Copy and paste the contents of your secrets.toml file (with your real keys) into the "Secrets" text box.

Click "Deploy!"
