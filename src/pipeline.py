"""
Main Analysis Pipeline
Orchestrates the full workflow:
1. Data Collection (Arctic Shift + PRAW)
2. Preprocessing
3. Sentiment Analysis
4. Topic Modeling
5. Visualization
"""

import os
import sys
import json
import argparse
from datetime import datetime

import pandas as pd

from config import QUERY_TERMS, SUBREDDITS, DATE_CONFIG, KEY_EVENTS
from data_collector import ArcticShiftCollector, save_data, load_data
from preprocessor import preprocess_posts, add_event_labels
from sentiment_analyzer import SentimentAnalyzer, get_sentiment_time_series
from topic_modeler import LDATopicModeler, get_topic_trends
from visualizer import create_dashboard


def run_collection(
    output_path: str = 'data/raw/reddit_posts.json',
    queries: list = None,
    subreddits: list = None,
    limit_per_query: int = 100
) -> list:
    """
    Step 1: Collect data from Arctic Shift API

    Args:
        output_path: Path to save raw data
        queries: Search terms (default: config.QUERY_TERMS)
        subreddits: Target subreddits (default: config.SUBREDDITS)
        limit_per_query: Max posts per query

    Returns:
        List of collected posts
    """
    print("\n" + "="*60)
    print("STEP 1: DATA COLLECTION")
    print("="*60)

    collector = ArcticShiftCollector()

    posts = collector.collect_all(
        queries=queries or QUERY_TERMS,
        subreddits=subreddits or SUBREDDITS,
        limit_per_query=limit_per_query
    )

    if posts:
        save_data(posts, output_path)

    return posts


def run_preprocessing(
    posts: list = None,
    input_path: str = 'data/raw/reddit_posts.json',
    output_path: str = 'data/processed/posts_clean.csv',
    filter_english: bool = True
) -> pd.DataFrame:
    """
    Step 2: Preprocess and clean data

    Args:
        posts: Raw posts list (or load from input_path)
        input_path: Path to raw data
        output_path: Path to save processed data
        filter_english: Filter non-English posts

    Returns:
        Cleaned DataFrame
    """
    print("\n" + "="*60)
    print("STEP 2: PREPROCESSING")
    print("="*60)

    if posts is None:
        posts = load_data(input_path)

    df = preprocess_posts(posts, filter_english=filter_english)

    # Add event labels
    df = add_event_labels(df, KEY_EVENTS)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data: {output_path}")

    return df


def run_sentiment_analysis(
    df: pd.DataFrame = None,
    input_path: str = 'data/processed/posts_clean.csv',
    output_path: str = 'data/processed/posts_sentiment.csv'
) -> pd.DataFrame:
    """
    Step 3: Sentiment analysis

    Args:
        df: Preprocessed DataFrame (or load from input_path)
        input_path: Path to preprocessed data
        output_path: Path to save results

    Returns:
        DataFrame with sentiment scores
    """
    print("\n" + "="*60)
    print("STEP 3: SENTIMENT ANALYSIS")
    print("="*60)

    if df is None:
        df = pd.read_csv(input_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['date'])

    analyzer = SentimentAnalyzer()
    df = analyzer.analyze_dataframe(df)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment data: {output_path}")

    return df


def run_topic_modeling(
    df: pd.DataFrame = None,
    input_path: str = 'data/processed/posts_sentiment.csv',
    output_path: str = 'data/processed/posts_final.csv',
    n_topics: int = 6
) -> tuple:
    """
    Step 4: Topic modeling

    Args:
        df: DataFrame with text
        input_path: Path to load data
        output_path: Path to save results
        n_topics: Number of topics

    Returns:
        Tuple of (DataFrame, topic_modeler)
    """
    print("\n" + "="*60)
    print("STEP 4: TOPIC MODELING")
    print("="*60)

    if df is None:
        df = pd.read_csv(input_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['date'])

    # Fit LDA model
    modeler = LDATopicModeler(n_topics=n_topics)
    modeler.fit(df['text_combined'].tolist())

    # Print topics
    modeler.print_topics()

    # Add to DataFrame
    df = modeler.add_topics_to_df(df)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved final data: {output_path}")

    return df, modeler


def run_visualization(
    df: pd.DataFrame = None,
    input_path: str = 'data/processed/posts_final.csv',
    output_dir: str = 'outputs/figures',
    topic_labels: dict = None
):
    """
    Step 5: Generate visualizations

    Args:
        df: Complete DataFrame
        input_path: Path to load data
        output_dir: Directory for figures
        topic_labels: Optional topic name mapping
    """
    print("\n" + "="*60)
    print("STEP 5: VISUALIZATION")
    print("="*60)

    if df is None:
        df = pd.read_csv(input_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = pd.to_datetime(df['date'])

    create_dashboard(
        df,
        events=KEY_EVENTS,
        output_dir=output_dir,
        topic_labels=topic_labels
    )


def run_full_pipeline(
    queries: list = None,
    subreddits: list = None,
    n_topics: int = 6,
    limit_per_query: int = 100
) -> pd.DataFrame:
    """
    Run the complete analysis pipeline

    Args:
        queries: Search terms
        subreddits: Target subreddits
        n_topics: Number of topics for LDA
        limit_per_query: Max posts per query

    Returns:
        Final analyzed DataFrame
    """
    print("\n" + "#"*60)
    print("REDDIT US-NK ANALYSIS PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#"*60)

    # Step 1: Collection
    posts = run_collection(
        queries=queries,
        subreddits=subreddits,
        limit_per_query=limit_per_query
    )

    if not posts:
        print("No posts collected. Exiting.")
        return None

    # Step 2: Preprocessing
    df = run_preprocessing(posts=posts)

    if len(df) < 10:
        print("Too few posts for analysis. Exiting.")
        return df

    # Step 3: Sentiment
    df = run_sentiment_analysis(df=df)

    # Step 4: Topics
    df, modeler = run_topic_modeling(df=df, n_topics=n_topics)

    # Step 5: Visualization
    run_visualization(df=df)

    print("\n" + "#"*60)
    print("PIPELINE COMPLETE")
    print(f"Total posts analyzed: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("#"*60)

    return df


def main():
    parser = argparse.ArgumentParser(description='Reddit US-NK Analysis Pipeline')

    parser.add_argument('--step', type=str, default='all',
                       choices=['collect', 'preprocess', 'sentiment', 'topics', 'visualize', 'all'],
                       help='Which step to run')
    parser.add_argument('--n-topics', type=int, default=6,
                       help='Number of topics for LDA')
    parser.add_argument('--limit', type=int, default=100,
                       help='Max posts per query')

    args = parser.parse_args()

    if args.step == 'all':
        run_full_pipeline(n_topics=args.n_topics, limit_per_query=args.limit)
    elif args.step == 'collect':
        run_collection(limit_per_query=args.limit)
    elif args.step == 'preprocess':
        run_preprocessing()
    elif args.step == 'sentiment':
        run_sentiment_analysis()
    elif args.step == 'topics':
        run_topic_modeling(n_topics=args.n_topics)
    elif args.step == 'visualize':
        run_visualization()


if __name__ == "__main__":
    main()
