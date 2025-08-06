from dotenv import load_dotenv
import os
import google.generativeai as genai
import googlemaps
import re
import requests

# Load environment variables from .env
load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize clients
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def get_place_id_from_short_url(short_url, api_key):
    try:
        # Step 1: Follow the redirect to get the full URL
        response = requests.get(short_url, allow_redirects=True)
        final_url = response.url
        print("Redirected URL:", final_url)

        # Step 2: Extract coordinates from the final URL using regex
        match = re.search(r'@(-?\d+\.\d+),(-?\d+\.\d+)', final_url)
        if not match:
            print("Coordinates not found in URL.")
            return None

        lat, lng = match.groups()
        print(f"Extracted Coordinates: {lat}, {lng}")

        # Step 3: Use Geocoding API to get place_id
        geo_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}"
        print(f"Calling Geocoding API: {geo_url}")

        geo_response = requests.get(geo_url)
        print(f"Geocoding API Status Code: {geo_response.status_code}")

        if geo_response.status_code != 200:
            print(f"Geocoding API Error: {geo_response.text}")
            return None

        geo_data = geo_response.json()
        print(f"Geocoding API Response: {geo_data}")

        if 'results' in geo_data and geo_data['results']:
            place_id = geo_data['results'][0]['place_id']
            print("Place ID:", place_id)
            return place_id
        else:
            print("Place ID not found in geocoding response.")
            print(f"Geocoding response: {geo_data}")
            return None

    except Exception as e:
        print("Error:", e)
        return None

# extract the place_id from the Google Maps URL
def extract_place_id(google_maps_url):
    print(f"Processing URL: {google_maps_url}")

    # Check if it's a short URL (like goo.gl or maps.app.goo.gl)
    if any(shortener in google_maps_url for shortener in ['goo.gl', 'maps.app.goo.gl', 'maps.google.com/maps']):
        print("Detected short URL, following redirects...")
        place_id = get_place_id_from_short_url(google_maps_url, GOOGLE_MAPS_API_KEY)
        if place_id:
            return place_id

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
    fields = ["name", "formatted_address", "rating", "user_ratings_total", "price_level", "opening_hours", "website", "phone_number", "photos", "reviews"]
    place = gmaps.place(place_id=place_id, fields=fields)
    return place.get('result', {})

# summarize location using Gemini AI
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
    Phone Number: {details.get('phone_number', 'N/A')}
    Photos: {details.get('photos', 'N/A')}"""

    response = model.generate_content(prompt)
    return response.text

# extract reviews into an array
def extract_reviews(details):
    reviews = details.get('reviews', [])
    return [review['text'] for review in reviews if review.get('text')]

# main function
if __name__ == "__main__":
    url = input("Paste the Google Maps URL: ").strip()
    try:
        place_id = extract_place_id(url)
        print(f"\n\U0001F4DD Place ID: {place_id}")

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
