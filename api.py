# --- Imports ---
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Any
import requests
import urllib.parse 

# --- App Initialization ---
app = FastAPI(
    title="Career Recommendation API",
    description="An API to recommend job titles and provide real job listings from Adzuna.",
    version="1.0.0"
)

# --- Adzuna API Configuration ---
ADZUNA_APP_ID = "93821b80"
ADZUNA_APP_KEY = "1eb0d4057307502eed480bf9258aacb4"

# Base URL is now dynamic, we'll add the country code later
ADZUNA_BASE_URL = "https://api.adzuna.com/v1/api/jobs"
# --- !!!!!!!!!!!!!!!!!!!!! ---


# --- Pydantic Data Models ---
class UserRequest(BaseModel):
    ExperienceLevel: str
    YearsOfExperience: float
    master_text: str
    location_code: str  # <-- NEW: 'in', 'us', 'gb', etc.

class Recommendation(BaseModel):
    title: str
    score: float

class JobListing(BaseModel):
    title: str
    company: str
    location: str
    link: str

class JobListingGroup(BaseModel):
    recommended_title: str
    job_listings: List[JobListing]

class ApiResponse(BaseModel):
    recommendations: List[Recommendation]
    job_listings: List[JobListingGroup]


# --- Global Model Variable ---
model_pipeline = None

# --- Startup Event Handler ---
@app.on_event("startup")
def load_model():
    """
    Load the ML pipeline from disk when the server starts.
    """
    global model_pipeline
    model_file = 'career_model.pkl'
    try:
        model_pipeline = joblib.load(model_file)
        print(f"--- Model '{model_file}' loaded successfully. ---")
    except FileNotFoundError:
        print(f"--- ERROR: Model file '{model_file}' not found. ---")
    except Exception as e:
        print(f"--- ERROR: Failed to load model. {e} ---")


# --- Helper Function to Call Adzuna ---
# --- !!!!!!! UPDATED FUNCTION !!!!!!! ---
def get_job_listings(job_title: str, experience_level: str, location_code: str) -> List[JobListing]:
    """
    Calls the Adzuna API to get real job listings for a specific country.
    """
    search_query = f"{experience_level} {job_title}"
    print(f"Calling Adzuna for: {search_query} in {location_code}")

    # Build the dynamic URL for the specific country
    search_url = f"{ADZUNA_BASE_URL}/{location_code}/search/1"

    params = {
        'app_id': ADZUNA_APP_ID,
        'app_key': ADZUNA_APP_KEY,
        'what': search_query,        
        'results_per_page': 5      
    }

    try:
        response = requests.get(search_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            job_data = data.get('results', []) 
            
            parsed_jobs = []
            for item in job_data:
                parsed_jobs.append(JobListing(
                    title=item.get('title', 'No Title'),
                    company=item.get('company', {}).get('display_name', 'No Company'),
                    location=item.get('location', {}).get('display_name', 'No Location'),
                    link=item.get('redirect_url', '#')
                ))
            print(f"Found {len(parsed_jobs)} jobs from Adzuna for {job_title}")
            return parsed_jobs
        else:
            print(f"Adzuna API Error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        print(f"Exception during Adzuna call: {e}")
        return []
# --- !!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---


# --- API Endpoints ---
@app.get("/")
def get_root():
    return {"status": "ok", "message": "Career Recommendation API is running!"}


@app.post("/recommend", response_model=ApiResponse)
def post_recommend(request: UserRequest):
    """
    Main prediction endpoint.
    """
    if model_pipeline is None:
        return {"error": "Model not loaded. Please check server logs."}

    # 1. Convert request to DataFrame (ML model doesn't use location)
    input_data = {
        'YearsOfExperience': [request.YearsOfExperience],
        'ExperienceLevel': [request.ExperienceLevel],
        'master_text': [request.master_text]
    }
    input_df = pd.DataFrame.from_dict(input_data)

    # 2. Make ML prediction
    try:
        probabilities = model_pipeline.predict_proba(input_df)[0]
    except Exception as e:
        return {"error": f"Error during prediction: {e}"}

    # 3. Process ML results
    class_names = model_pipeline.classes_
    scores = list(zip(class_names, probabilities))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_3_scores = scores[:3]
    top_3_recommendations = [
        Recommendation(title=title, score=round(score, 4)) for title, score in top_3_scores
    ]

    # 4. Get Job Listings for all 3 (now with location)
    all_job_listings_groups = []
    
    for rec in top_3_recommendations:
        # --- !!!!!!! UPDATED CALL !!!!!!! ---
        # We now pass the location_code from the user request
        jobs = get_job_listings(rec.title, request.ExperienceLevel, request.location_code)
        
        all_job_listings_groups.append(JobListingGroup(
            recommended_title=rec.title,
            job_listings=jobs
        ))

    # 5. Return the final response
    return ApiResponse(
        recommendations=top_3_recommendations,
        job_listings=all_job_listings_groups
    )

# --- Run the App ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)