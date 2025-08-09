import os
import re
import requests
import json
import time
from dotenv import load_dotenv
from langchain_community.llms import Ollama

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# ✅ Initialize Ollama LLM
llm = Ollama(model="gemma2:2b", base_url="http://localhost:11434")

# --- Geocoding and Place Details Functions ---

def get_place_id_from_url(url, api_key):
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




def fetch_place_details(place_id, api_key):
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

# --- Ollama Review Analysis Function ---

def analyze_review(review_text, reviewer_name="Anonymous"):
    prompt = f"""
    Analyze the following review text for sentiment, specificity, and potential signs of being fake. 
 Provide a score from 1-10 for authenticity, where 1 is highly suspicious and 10 is very authentic.
 
 Reviewer: "{reviewer_name}"
 Review: "{review_text}"
 
 Instructions: Your analysis should be based on linguistic patterns, specificity of details, and sentiment coherence.
 Please provide your analysis in the following JSON format:
 
 {{
     "reviewer": "{reviewer_name}",
     "sentiment": "[positive|negative|neutral]",
     "specificity": "[high|medium|low]",
     "authenticity_score": [0-5],
     "category": "**Fake**" or "**Not Fake**", 
     "recommendation": "**Go**" or "**Avoid**",
     "summary": "[brief explanation]"
 }}
    """
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        print(f"Error analyzing review with Ollama: {e}")
        return json.dumps({"error": str(e)})

# --- Main Logic ---

if __name__ == "__main__":
    if not GOOGLE_MAPS_API_KEY:
        print("❌ Please set your GOOGLE_MAPS_API_KEY in the .env file.")
    else:
        url = input("Paste the Google Maps URL: ").strip()
        try:
            place_id = get_place_id_from_url(url, GOOGLE_MAPS_API_KEY)

            if not place_id:
                raise ValueError("❌ Could not find a valid Place ID.")

            details = fetch_place_details(place_id, GOOGLE_MAPS_API_KEY)

            if not details:
                raise ValueError("❌ Could not fetch place details.")

            output = {
                "place_name": details.get('name'),
                "address": details.get('formatted_address'),
                "rating": details.get('rating'),
                "total_reviews": details.get('user_ratings_total'),
                "reviews_analysis": []
            }

            reviews_data = details.get('reviews', [])
            if not reviews_data:
                print("\n⚠️ No reviews available for analysis.")
            else:
                print("\n--- 🔍 Review Analysis ---")
                for i, review in enumerate(reviews_data, 1):
                    review_text = review.get('text', '')
                    if not review_text.strip():
                        continue
                    print(f"Analyzing Review {i}...")
                    analysis = analyze_review(review_text)
                    try:
                        parsed_analysis = json.loads(analysis)
                    except:
                        parsed_analysis = {"raw_output": analysis}

                    output['reviews_analysis'].append({
                        "original_review": review_text,
                        **parsed_analysis
                    })

            print("\n✅ Final Structured JSON Output:\n")
            print(json.dumps(output, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"❌ Error: {e}")
