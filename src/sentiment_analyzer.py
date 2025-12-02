"""
Sentiment Analysis Module
- VADER (rule-based, good for social media)
- TextBlob (pattern-based)
"""

import pandas as pd
from typing import List, Dict
from tqdm import tqdm

# VADER Sentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("vaderSentiment not installed. Run: pip install vaderSentiment")

# TextBlob
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("textblob not installed. Run: pip install textblob")


class SentimentAnalyzer:
    """
    Multi-method sentiment analysis for Reddit posts
    """

    def __init__(self):
        self.vader = None
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()

    def analyze_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER

        Returns:
            Dict with 'neg', 'neu', 'pos', 'compound' scores
            compound: -1 (most negative) to +1 (most positive)
        """
        if not self.vader or not text:
            return {'neg': 0, 'neu': 1, 'pos': 0, 'compound': 0}

        return self.vader.polarity_scores(text)

    def analyze_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob

        Returns:
            Dict with 'polarity' (-1 to 1) and 'subjectivity' (0 to 1)
        """
        if not TEXTBLOB_AVAILABLE or not text:
            return {'polarity': 0, 'subjectivity': 0}

        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }

    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Convert compound score to categorical label

        Thresholds (standard VADER):
            positive: compound >= 0.05
            negative: compound <= -0.05
            neutral: -0.05 < compound < 0.05
        """
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text_combined') -> pd.DataFrame:
        """
        Add sentiment columns to DataFrame

        Args:
            df: DataFrame with text column
            text_column: Name of column containing text to analyze

        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()

        # Initialize columns
        df['vader_compound'] = 0.0
        df['vader_pos'] = 0.0
        df['vader_neg'] = 0.0
        df['vader_neu'] = 0.0
        df['textblob_polarity'] = 0.0
        df['textblob_subjectivity'] = 0.0
        df['sentiment_label'] = 'neutral'

        print("Analyzing sentiment...")
        for idx in tqdm(df.index, desc="Sentiment Analysis"):
            text = df.loc[idx, text_column]

            # VADER analysis
            vader_scores = self.analyze_vader(text)
            df.loc[idx, 'vader_compound'] = vader_scores['compound']
            df.loc[idx, 'vader_pos'] = vader_scores['pos']
            df.loc[idx, 'vader_neg'] = vader_scores['neg']
            df.loc[idx, 'vader_neu'] = vader_scores['neu']

            # TextBlob analysis
            tb_scores = self.analyze_textblob(text)
            df.loc[idx, 'textblob_polarity'] = tb_scores['polarity']
            df.loc[idx, 'textblob_subjectivity'] = tb_scores['subjectivity']

            # Label
            df.loc[idx, 'sentiment_label'] = self.get_sentiment_label(vader_scores['compound'])

        # Summary
        sentiment_counts = df['sentiment_label'].value_counts()
        print(f"\nSentiment Distribution:")
        print(f"  Positive: {sentiment_counts.get('positive', 0)}")
        print(f"  Neutral:  {sentiment_counts.get('neutral', 0)}")
        print(f"  Negative: {sentiment_counts.get('negative', 0)}")
        print(f"  Mean VADER compound: {df['vader_compound'].mean():.3f}")

        return df


def get_sentiment_time_series(df: pd.DataFrame, freq: str = 'W') -> pd.DataFrame:
    """
    Aggregate sentiment by time period

    Args:
        df: DataFrame with sentiment columns and 'datetime'
        freq: Pandas frequency ('D', 'W', 'M')

    Returns:
        Time series DataFrame with sentiment trends
    """
    ts = df.set_index('datetime').resample(freq).agg({
        'vader_compound': 'mean',
        'vader_pos': 'mean',
        'vader_neg': 'mean',
        'textblob_polarity': 'mean',
        'id': 'count'
    }).rename(columns={'id': 'post_count'})

    ts = ts.fillna(0)
    return ts


def get_sentiment_by_subreddit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare sentiment across subreddits
    """
    return df.groupby('subreddit').agg({
        'vader_compound': ['mean', 'std'],
        'textblob_polarity': ['mean', 'std'],
        'id': 'count'
    }).round(3)


def get_sentiment_by_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare sentiment around key events
    """
    # Filter to event-related posts
    event_df = df[df['event'].notna()]

    if len(event_df) == 0:
        print("No event labels found. Run add_event_labels() first.")
        return pd.DataFrame()

    return event_df.groupby('event').agg({
        'vader_compound': ['mean', 'std', 'count'],
        'textblob_polarity': 'mean'
    }).round(3)


if __name__ == "__main__":
    # Test sentiment analysis
    analyzer = SentimentAnalyzer()

    test_texts = [
        "North Korea's aggressive missile tests threaten regional security.",
        "The US-ROK alliance remains strong and committed to peace.",
        "Diplomatic talks continue between the nations.",
        "Kim Jong Un's dangerous rhetoric escalates tensions.",
        "South Korea and US conduct joint military exercises."
    ]

    print("Testing Sentiment Analysis\n" + "="*50)

    for text in test_texts:
        vader = analyzer.analyze_vader(text)
        tb = analyzer.analyze_textblob(text)
        label = analyzer.get_sentiment_label(vader['compound'])

        print(f"\nText: {text[:60]}...")
        print(f"  VADER compound: {vader['compound']:.3f} ({label})")
        print(f"  TextBlob polarity: {tb['polarity']:.3f}")
