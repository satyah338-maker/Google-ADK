import logging
import asyncio
import datetime
import os
import requests

from dotenv import load_dotenv

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.sessions import InMemorySessionService, Session
from google.adk.runners import Runner
from google.genai import types

from zoneinfo import ZoneInfo
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from timezonefinder import TimezoneFinder


import warnings
warnings.filterwarnings('ignore')


logging.basicConfig(level=logging.ERROR)


# ====================================================
# Configurations
# ====================================================
load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
WEATHER_BASE_URL = os.getenv("WEATHER_BASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
APP_NAME = "weather_agent_v1"
USER_ID = "user_1"
SESSION_ID = "session_001"

geolocater = Nominatim(user_agent="weather_time_agent")
geocode = RateLimiter(geolocater.geocode, min_delay_seconds=1)
timezone_finder = TimezoneFinder()


# ====================================================
# Setup Runner and Session Service
# ====================================================
session_service = InMemorySessionService()


async def init_session(app_name: str, user_id: str, session_id: str) -> Session:
    session = await session_service.create_session(
        app_name=app_name, user_id=user_id,
        session_id=session_id
    )

    return session

session = asyncio.run(init_session(APP_NAME, USER_ID, SESSION_ID))


# ====================================================
# Tools
# ====================================================
def get_city_location(city: str):
    """Resolve the city name to coordinates."""
    try:
        location = geocode(city, timeout=10)

        if location is None:
            return None
        return location
    except Exception:
        return None


def get_current_time(city: str) -> dict:
    """Get the current local time for any recognized city."""
    location = get_city_location(city)

    if location is None:
        return {
            "status": "error",
            "error_message": f"Could not find the city '{city}'. Try including the country name."
        }

    tz_identifier = timezone_finder.timezone_at(
        lng=location.longitude, lat=location.latitude
    )

    if not tz_identifier:
        return {
            "status": "error",
            "error_message": f"Could not determine the time zone for the {city}"
        }

    now = datetime.datetime.now(ZoneInfo(tz_identifier))

    return {
        "status": "success",
        "city": location.address,
        "timezone": tz_identifier,
        "report": f"The current time in {location.address} is {now.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')}"
    }


def get_weather(city: str) -> dict:
    """Return current weather information for a city."""

    if not city or not city.strip():
        return {
            "status": "error",
            "error_message": "City name must not be empty."
        }

    if not WEATHER_API_KEY or not WEATHER_BASE_URL:
        return {
            "status": "error",
            "error_message": "Weather API configuration is missing."
        }

    try:
        response = requests.get(
            WEATHER_BASE_URL,
            params={
                "key": WEATHER_API_KEY,
                "q": city.strip()
            },
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        return {
            "status": "success",
            "data": data
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "error_message": "The weather service timed out."
        }

    except requests.exceptions.HTTPError:
        return {
            "status": "error",
            "error_message": "The weather service returned an HTTP error."
        }

    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "error_message": "Could not connect to the weather service."
        }

    except ValueError:
        return {
            "status": "error",
            "error_message": "The weather service returned invalid JSON."
        }


# ====================================================
# Agents
# ====================================================
current_time_agent = Agent(
    name="current_time_agent",
    model=LiteLlm(model="groq/llama-3.1-8b-instant"),
    description="Provides current time information for specific cities.",
    instruction="You are a helpful current time assistant.\
    When the user asks for the current time in a specific city,\
    use the get_current_time tool to find the information.\
    If the tool returns an error, inform the user politely.\
    If the tool is successful, present the current time report clearly.",
    tools=[get_current_time]
)

weather_agent = Agent(
    name="weather_agent",
    model="gemini-3.7-flash",
    description="Provide weather information for the specific cities.",
    instruction="You are helpful weather assistant.\
    When the user asks for the weather in a specific city, \
    use the get_weather tool to find the information. \
    If the tool returns an error, inform the user politely. \
    If the tool is successful, present the weather report clearly.",
    tools=[get_weather]
)

runner = Runner(
    agent=weather_agent,
    app_name=APP_NAME,
    session_service=session_service
)


root_agent = Agent(
    name="live_api_data",
    model="gemini-3-flash-preview",
    instruction="""
    You are a helpful weather and world-clock assistant.

    You can use these tools:
    - get_current_time: Find the current local time in a city.
    - get_weather: Retrieve current weather information for a city.

    Guidelines:
    1. Use the appropriate tool for real-time questions.
    2. Never invent weather conditions or current times.
    3. Explain tool errors clearly.
    4. Present weather information in a readable format.
    5. Include the city and time zone when reporting local time.
    """,
    tools=[get_current_time, get_weather]
)



async def call_agent_async(query: str, runner: Runner, user_id: str, session_id: str):
    content = types.Content(role="user", parts=[])