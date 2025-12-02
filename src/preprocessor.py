"""
Data Preprocessing Module
- Text cleaning
- Language detection (English only)
- Date parsing and normalization
"""

import re
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Optional language detection
try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("langdetect not installed. Language filtering will be skipped.")


def clean_text(text: str) -> str:
    """
    Clean text for analysis

    - Remove URLs
    - Remove special characters
    - Normalize whitespace
    - Convert to lowercase
    """
    if not text or not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Remove Reddit-specific formatting
    text = re.sub(r'\[deleted\]|\[removed\]', '', text)

    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_english(text: str, min_length: int = 20) -> bool:
    """Check if text is in English"""
    if not LANGDETECT_AVAILABLE:
        return True  # Skip check if langdetect not available

    if not text or len(text) < min_length:
        return False

    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False


def parse_timestamp(timestamp: int) -> datetime:
    """Convert Unix timestamp to datetime"""
    try:
        return datetime.fromtimestamp(timestamp)
    except (ValueError, TypeError, OSError):
        return None


def preprocess_posts(posts: List[Dict], filter_english: bool = True) -> pd.DataFrame:
    """
    Preprocess raw posts into clean DataFrame

    Args:
        posts: List of post dictionaries
        filter_english: Whether to filter non-English posts

    Returns:
        Cleaned pandas DataFrame
    """
    processed = []

    for post in posts:
        # Extract fields
        post_id = post.get('id', '')
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        author = post.get('author', '[deleted]')
        score = post.get('score', 0)
        num_comments = post.get('num_comments', 0)
        created_utc = post.get('created_utc', 0)
        subreddit = post.get('subreddit', '')
        permalink = post.get('permalink', '')
        source = post.get('source', 'unknown')
        query_term = post.get('query_term', '')

        # Combine title and selftext for full text analysis
        combined_text = f"{title} {selftext}".strip()

        # Clean text
        clean_title = clean_text(title)
        clean_selftext = clean_text(selftext)
        clean_combined = clean_text(combined_text)

        # Skip empty posts
        if not clean_combined:
            continue

        # Language filter
        if filter_english and not is_english(clean_combined):
            continue

        # Parse date
        dt = parse_timestamp(created_utc)
        if not dt:
            continue

        processed.append({
            'id': post_id,
            'title': title,
            'title_clean': clean_title,
            'selftext': selftext,
            'selftext_clean': clean_selftext,
            'text_combined': clean_combined,
            'author': author,
            'score': score,
            'num_comments': num_comments,
            'created_utc': created_utc,
            'datetime': dt,
            'date': dt.date(),
            'year': dt.year,
            'month': dt.month,
            'year_month': dt.strftime('%Y-%m'),
            'subreddit': subreddit,
            'permalink': permalink,
            'source': source,
            'query_term': query_term
        })

    df = pd.DataFrame(processed)

    if len(df) > 0:
        # Sort by date
        df = df.sort_values('datetime').reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=['id'], keep='first')

    print(f"Preprocessed {len(df)} posts (from {len(posts)} raw)")
    return df


def add_event_labels(df: pd.DataFrame, events: Dict[str, str], window_days: int = 7) -> pd.DataFrame:
    """
    Add labels for posts near key events

    Args:
        df: DataFrame with 'date' column
        events: Dict of {date_str: event_description}
        window_days: Days before/after event to label

    Returns:
        DataFrame with 'event' and 'event_window' columns
    """
    df = df.copy()
    df['event'] = None
    df['event_window'] = False

    for event_date_str, event_name in events.items():
        event_date = datetime.strptime(event_date_str, '%Y-%m-%d').date()

        # Find posts within window
        mask = (
            (df['date'] >= event_date - pd.Timedelta(days=window_days)) &
            (df['date'] <= event_date + pd.Timedelta(days=window_days))
        )

        df.loc[mask, 'event'] = event_name
        df.loc[mask, 'event_window'] = True

    event_posts = df['event_window'].sum()
    print(f"Labeled {event_posts} posts near key events")

    return df


def get_time_series(df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    Aggregate posts by time period

    Args:
        df: DataFrame with 'datetime' column
        freq: Pandas frequency string ('D'=daily, 'W'=weekly, 'M'=monthly)

    Returns:
        Time series DataFrame with post counts
    """
    ts = df.set_index('datetime').resample(freq).agg({
        'id': 'count',
        'score': 'mean',
        'num_comments': 'mean'
    }).rename(columns={'id': 'post_count'})

    ts = ts.fillna(0)
    return ts


if __name__ == "__main__":
    # Test with sample data
    sample_posts = [
        {
            "id": "test1",
            "title": "North Korea launches missile test",
            "selftext": "Kim Jong Un announced new military capabilities.",
            "score": 1500,
            "num_comments": 250,
            "created_utc": 1672531200,  # 2023-01-01
            "subreddit": "worldnews",
            "source": "test"
        },
        {
            "id": "test2",
            "title": "한국어 제목 테스트",  # Korean text
            "selftext": "이것은 테스트입니다",
            "score": 100,
            "num_comments": 10,
            "created_utc": 1672617600,
            "subreddit": "korea",
            "source": "test"
        }
    ]

    df = preprocess_posts(sample_posts, filter_english=True)
    print(f"\nResult: {len(df)} posts after filtering")
    print(df[['title_clean', 'date', 'subreddit']].head())
