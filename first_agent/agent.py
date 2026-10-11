
import datetime
from zoneinfo import ZoneInfo

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from timezonefinder import TimezoneFinder
from google.adk.agents import Agent


# Initialize reusable services
geolocator = Nominatim(user_agent="weather_time_agent")
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1
)
timezone_finder = TimezoneFinder()


def get_city_location(city: str):
    """Resolve any city name to coordinates."""
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
            "error_message": (
                f"Could not find the city '{city}'. "
                "Try including the country name."
            ),
        }

    tz_identifier = timezone_finder.timezone_at(
        lng=location.longitude,
        lat=location.latitude,
    )

    if not tz_identifier:
        return {
            "status": "error",
            "error_message": (
                f"Could not determine the time zone for {city}."
            ),
        }

    now = datetime.datetime.now(ZoneInfo(tz_identifier))

    return {
        "status": "success",
        "city": location.address,
        "timezone": tz_identifier,
        "report": (
            f"The current time in {location.address} is "
            f"{now.strftime('%Y-%m-%d %H:%M:%S %Z (%z)')}."
        ),
    }


def get_weather(city: str) -> dict:
    """Return a sample weather report for Mumbai only."""
    if city.strip().casefold() == "mumbai":
        return {
            "status": "success",
            "report": (
                "Sample weather for Mumbai: sunny, 30°C (86°F). "
                "This is sample data, not a live weather report."
            ),
        }

    return {
        "status": "error",
        "error_message": (
            "Live weather lookup is not configured yet."
        ),
    }


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-3.7-flash",
    description=(
        "An agent that provides local time for cities worldwide "
        "and sample weather for Mumbai."
    ),
    instruction=(
        "Answer questions about city time and weather. "
        "Use get_current_time for time questions and get_weather "
        "for weather questions. Accept city names worldwide. "
        "If a city name is ambiguous, ask the user which country "
        "or region they mean. Never invent live weather data."
    ),
    tools=[get_current_time, get_weather],
)
