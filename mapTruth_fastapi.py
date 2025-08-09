import os
import re
import requests
import json
import time
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Configuration
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Initialize Ollama LLM
llm = OllamaLLM(model="gemma2:2b", base_url="http://localhost:11434",temperature=0.2)

# Initialize FastAPI app
app = FastAPI(
    title="MapTruth AI",
    description="Analyze Google Maps places and reviews for authenticity",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class URLRequest(BaseModel):
    url: str

class PlaceDetails(BaseModel):
    place_name: Optional[str]
    address: Optional[str]
    rating: Optional[float]
    total_reviews: Optional[int]

class ReviewAnalysis(BaseModel):
    reviewer: str
    sentiment: str
    specificity: str
    authenticity_score: int
    category: str
    recommendation: str
    summary: str
    original_review: str

class AnalysisResponse(BaseModel):
    place_details: PlaceDetails
    reviews_analysis: List[ReviewAnalysis]
    success: bool
    message: str

class ErrorResponse(BaseModel):
    success: bool
    error: str

# Core functions
def get_place_id_from_url(url: str, api_key: str) -> Optional[str]:
    """Extract place_id from Google Maps URL"""
    print(f"Processing URL: {url}")
    
    # Follow redirects for short URLs
    if any(shortener in url for shortener in ["goo.gl", "maps.app.goo.gl"]):
        try:
            response = requests.get(url, allow_redirects=True, timeout=10)
            url = response.url
            print(f"Redirected URL: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Redirect error: {e}")
            return None

    # If URL has place_id param, use it directly
    pid_match = re.search(r'place_id=([^&]+)', url)
    if pid_match:
        return pid_match.group(1)

    # Try finding Place ID using the full place name from the URL
    name_match = re.search(r'/place/([^/@]+)', url)
    if name_match:
        place_name = name_match.group(1).replace('+', ' ')
        find_url = (
            f"https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
            f"?input={place_name}&inputtype=textquery&fields=place_id&key={api_key}"
        )
        resp = requests.get(find_url)
        data = resp.json()
        if data.get("status") == "OK" and data["candidates"]:
            return data["candidates"][0]["place_id"]

    print("Could not extract Place ID.")
    return None

def fetch_place_details(place_id: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Fetch place details from Google Places API"""
    fields = [
        "name", "formatted_address", "rating", "user_ratings_total",
        "price_level", "opening_hours", "website", "formatted_phone_number",
        "photo", "reviews"
    ]

    fields_str = ",".join(fields)
    details_url = f"https://maps.googleapis.com/maps/api/place/details/json?place_id={place_id}&fields={fields_str}&key={api_key}"

    try:
        response = requests.get(details_url)
        response.raise_for_status()
        details_data = response.json()

        if details_data.get('status') == 'OK':
            print("Successfully fetched place details.")
            return details_data.get('result', {})
        else:
            print("Places Details API failed to return results.")
            print(f"API Response: {details_data}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error calling Places Details API: {e}")
        return None

def analyze_review(review_text: str, reviewer_name: str = "Anonymous") -> str:
    """Analyze review using Ollama"""
    prompt = f"""
You are a review analysis expert
Analyze the following Google Maps review and determine:

1. sentiment → one of: "positive", "negative", or "neutral"
2. specificity → one of: "high", "medium", or "low"
3. authenticity_score → integer from 1 to 5, where:
   a. Sentiment 
   b. specificity 
Or
   + Sentiment 
   + Specificity   
4. category → exactly one of: "Fake" or "Not Fake"
5. recommendation → exactly one of: "Go" or "Avoid"
6. summary → a short one-sentence reason for your decision

Reviewer: "{reviewer_name}"
Review: "{review_text}"

Rules:
- Output must be **valid JSON only**.
- Do not include extra commentary, explanations, or text outside JSON.
- Always pick exactly one category ("Fake" or "Not Fake").
- Always pick exactly one recommendation ("Go" or "Avoid").

Expected JSON format:
{{
    "reviewer": "{reviewer_name}",
    "sentiment": "positive|negative|neutral",
    "specificity": "high|medium|low",
    "authenticity_score": 1-5,
    "category": "Fake" or "Not Fake",
    "recommendation": "Go" or "Avoid",
    "summary": "brief explanation"
}}
"""
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error analyzing review with Ollama: {e}")
        return json.dumps({"error": str(e)})

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MapTruth AI - Google Maps Review Analyzer", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "ollama_connected": True}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_place(request: URLRequest):
    """Analyze a Google Maps place from URL"""
    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Google Maps API key not configured"
        )
    
    try:
        # Extract place ID
        place_id = get_place_id_from_url(request.url, GOOGLE_MAPS_API_KEY)
        if not place_id:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract Place ID from URL"
            )

        # Fetch place details
        details = fetch_place_details(place_id, GOOGLE_MAPS_API_KEY)
        if not details:
            raise HTTPException(
                status_code=400, 
                detail="Could not fetch place details"
            )

        # Prepare response data
        place_details = PlaceDetails(
            place_name=details.get('name'),
            address=details.get('formatted_address'),
            rating=details.get('rating'),
            total_reviews=details.get('user_ratings_total')
        )

        reviews_analysis = []
        reviews_data = details.get('reviews', [])

        if reviews_data:
            for i, review in enumerate(reviews_data, 1):
                review_text = review.get('text', '')
                if not review_text.strip():
                    continue
                    
                reviewer_name = review.get('author_name', 'Anonymous')
                analysis = analyze_review(review_text, reviewer_name)
                
                try:
                    parsed_analysis = json.loads(analysis)
                except:
                    parsed_analysis = {
                        "reviewer": reviewer_name,
                        "sentiment": "unknown",
                        "specificity": "unknown", 
                        "authenticity_score": 5,
                        "category": "Unknown",
                        "recommendation": "Unknown",
                        "summary": "Analysis failed",
                        "raw_output": analysis
                    }

                review_analysis = ReviewAnalysis(
                    original_review=review_text,
                    reviewer=parsed_analysis.get('reviewer', reviewer_name),
                    sentiment=parsed_analysis.get('sentiment', 'unknown'),
                    specificity=parsed_analysis.get('specificity', 'unknown'),
                    authenticity_score=parsed_analysis.get('authenticity_score', 5),
                    category=parsed_analysis.get('category', 'Unknown'),
                    recommendation=parsed_analysis.get('recommendation', 'Unknown'),
                    summary=parsed_analysis.get('summary', 'No analysis available')
                )
                reviews_analysis.append(review_analysis)

        return AnalysisResponse(
            place_details=place_details,
            reviews_analysis=reviews_analysis,
            success=True,
            message=f"Successfully analyzed {len(reviews_analysis)} reviews"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/analyze-text")
async def analyze_review_text(review_text: str, reviewer_name: str = "Anonymous"):
    """Analyze a single review text"""
    try:
        analysis = analyze_review(review_text, reviewer_name)
        try:
            parsed_analysis = json.loads(analysis)
            return {"success": True, "analysis": parsed_analysis}
        except:
            return {"success": False, "raw_output": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mapTruth_fastapi:app", host="0.0.0.0", port=8000, reload=True)