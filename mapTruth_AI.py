from dotenv import load_dotenv
import os
import googlemaps
import re
import requests
from langchain.llms import Ollama

# Load environment variables from .env
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Initialize Ollama LLM
llm = Ollama(model="gemma2:2b", base_url="http://localhost:11434")

# Initialize Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def expand_short_url(short_url):
    try:
        response = requests.get(short_url, allow_redirects=True)
        final_url = response.url
        print("Expanded URL:", final_url)
        return final_url
    except Exception as e:
        print("Error expanding URL:", e)
        return None

# extract the place_id from the Google Maps URL
def extract_place_id(google_maps_url):
    print(f"Processing URL: {google_maps_url}")

    # Check if it's a short URL (like goo.gl or maps.app.goo.gl)
    if any(shortener in google_maps_url for shortener in ['goo.gl', 'maps.app.goo.gl']):
        print("Detected short URL, expanding...")
        expanded = expand_short_url(google_maps_url)
        if expanded:
            google_maps_url = expanded
        else:
            raise ValueError("Failed to expand short URL")

    # Try different patterns for Google Maps URLs
    patterns = [
        r"place_id=([^&]+)",
        r"maps\.google\.com/maps/place/[^/]+/([^/?]+)",
        r"maps\.google\.com/maps/place/[^/]+/@([^/?]+)",
        r"maps\.google\.com/maps/place/[^/]+/data=([^/?]+)",
        r"maps\.google\.com/maps/place/[^/]+/@[^/]+/([^/?]+)",
        r"maps\.google\.com/maps/place/[^/]+/@[^/]+/[^/]+/([^/?]+)",
        r"maps\.google\.com/maps/place/[^/]+/@[^/]+/[^/]+/[^/]+/([^/?]+)",
        r"maps\.google\.com/maps/place/[^/]+/@[^/]+/[^/]+/[^/]+/[^/]+/([^/?]+)"
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, google_maps_url)
        if match:
            place_id = match.group(1)
            print(f"Found place_id using pattern {i+1}: {place_id}")
            return place_id

    potential_place_id = re.search(r'([A-Za-z0-9_-]{20,})', google_maps_url)
    if potential_place_id:
        print(f"Found potential place_id: {potential_place_id.group(1)}")
        return potential_place_id.group(1)

    print("Could not extract place_id. Please ensure the URL contains a place_id parameter.")
    print("Example URL format: https://maps.google.com/?place_id=ChIJN1t_tDeuEmsRUsoyG83frY4")
    print("Or share URL: https://maps.google.com/maps/place/[PlaceName]/@[coordinates]/[place_id]")
    raise ValueError("Place ID not found in URL")

# fetch place details from Google Maps API
def fetch_place_details(place_id):
    fields = ["name", "formatted_address", "rating", "user_ratings_total", "price_level", "opening_hours", "website", "formatted_phone_number", "photos", "reviews"]
    place = gmaps.place(place_id=place_id, fields=fields)
    return place.get('result', {})

# summarize location using Ollama
def summarize_place(details):
    prompt = f"""
    You are a helpful assistant that summarizes location details.

    Name: {details.get('name', 'N/A')}
    Address: {details.get('formatted_address', 'N/A')}
    Rating: {details.get('rating', 'N/A')}
    Total Ratings: {details.get('user_ratings_total', 'N/A')}
    Price Level: {details.get('price_level', 'N/A')}
    Opening Hours: {details.get('opening_hours', 'N/A')}
    Website: {details.get('website', 'N/A')}
    Phone Number: {details.get('formatted_phone_number', 'N/A')}
    Photos: {details.get('photos', 'N/A')}

    Please provide a concise summary of this location.
    """

    response = llm.invoke(prompt)
    return response

# extract reviews into an array
def extract_reviews(details):
    reviews = details.get('reviews', [])
    return [review['text'] for review in reviews if review.get('text')]

# main function
if __name__ == "__main__":
    url = input("Paste the Google Maps URL: ").strip()
    try:
        place_id = extract_place_id(url)
        # print(f"\n\U0001F4DD Place ID: {place_id}")

        details = fetch_place_details(place_id)

        print("\n\U0001F4CD Place Summary:")
        print(summarize_place(details))

        print("\n\U0001F4DD Reviews:")
        reviews = extract_reviews(details)
        if reviews:
            for i, review in enumerate(reviews, 1):
                print(f"{i}. {review}\n")
        else:
            print("No reviews available for this location.")

    except Exception as e:
        print("❌ Error:", e)
