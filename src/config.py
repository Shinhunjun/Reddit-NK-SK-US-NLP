"""
Configuration settings for Reddit US-NK Analysis Project
"""

# Search Keywords for Korean Peninsula Security Topics
QUERY_TERMS = [
    # North Korea
    "north korea",
    "kim jong un",
    "pyongyang",
    "dprk",

    # South Korea & Alliance
    "south korea",
    "us korea alliance",
    "us rok alliance",
    "us south korea",

    # Korean Peninsula Security
    "korean peninsula",
    "korea missile",
    "korea nuclear",
    "korean war",

    # Specific Events
    "korea summit",
    "camp david korea",
    "korea military",
]

# Target Subreddits (ordered by relevance)
SUBREDDITS = [
    "worldnews",        # Major international news
    "geopolitics",      # Geopolitical analysis
    "politics",         # US politics perspective
    "northkorea",       # Dedicated NK discussion
    "korea",            # Korea-related discussions
    "AskAnAmerican",    # American perspectives
    "news",             # General news
]

# Time Period Configuration
DATE_CONFIG = {
    "arctic_shift": {
        "start": "2022-01-01",
        "end": "2023-12-31",
    },
    "praw": {
        # PRAW will fetch recent data (last ~1000 posts per query)
        "start": "2024-01-01",
        "end": "2025-12-01",
    }
}

# Key Events for Analysis (date: event description)
KEY_EVENTS = {
    "2022-03-24": "North Korea ICBM test (Hwasong-17)",
    "2022-10-04": "North Korea missile over Japan",
    "2022-11-18": "North Korea Hwasong-17 ICBM test",
    "2023-02-18": "North Korea ICBM test",
    "2023-04-13": "North Korea Hwasong-18 solid-fuel ICBM",
    "2023-07-12": "North Korea Hwasong-18 test",
    "2023-08-18": "Camp David Summit (US-ROK-Japan)",
    "2023-11-21": "North Korea satellite launch",
    "2024-01-14": "North Korea cruise missile test",
    "2024-05-27": "North Korea satellite launch failure",
    "2024-07-01": "North Korea ballistic missile test",
}

# API Configuration
ARCTIC_SHIFT_BASE_URL = "https://arctic-shift.photon-reddit.com/api"
ARCTIC_SHIFT_POSTS_ENDPOINT = f"{ARCTIC_SHIFT_BASE_URL}/posts/search"
ARCTIC_SHIFT_COMMENTS_ENDPOINT = f"{ARCTIC_SHIFT_BASE_URL}/comments/search"

# Rate limiting (be respectful to APIs)
REQUEST_DELAY = 1.0  # seconds between requests
MAX_RETRIES = 3
