"""
Visualization Module
- Time series plots
- Sentiment trends
- Topic distributions
- Word clouds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime

# Optional wordcloud
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'positive': '#2ecc71',
    'neutral': '#95a5a6',
    'negative': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#9b59b6'
}


def plot_post_volume(df: pd.DataFrame, freq: str = 'W',
                     title: str = 'Reddit Post Volume Over Time',
                     figsize: tuple = (14, 6),
                     save_path: str = None) -> plt.Figure:
    """
    Plot post volume over time

    Args:
        df: DataFrame with 'datetime' column
        freq: Aggregation frequency ('D', 'W', 'M')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate
    ts = df.set_index('datetime').resample(freq)['id'].count()

    # Plot
    ax.fill_between(ts.index, ts.values, alpha=0.3, color=COLORS['primary'])
    ax.plot(ts.index, ts.values, color=COLORS['primary'], linewidth=2)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Posts', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_sentiment_trend(df: pd.DataFrame, freq: str = 'W',
                         title: str = 'Sentiment Trend Over Time',
                         figsize: tuple = (14, 6),
                         events: Dict[str, str] = None,
                         save_path: str = None) -> plt.Figure:
    """
    Plot sentiment trend with optional event markers

    Args:
        df: DataFrame with 'datetime' and 'vader_compound'
        freq: Aggregation frequency
        events: Dict of {date_str: event_name} to mark
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate sentiment
    ts = df.set_index('datetime').resample(freq).agg({
        'vader_compound': 'mean',
        'id': 'count'
    })

    # Plot sentiment
    ax.plot(ts.index, ts['vader_compound'], color=COLORS['primary'],
            linewidth=2, label='Sentiment (VADER)')

    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Fill positive/negative regions
    ax.fill_between(ts.index, ts['vader_compound'], 0,
                    where=(ts['vader_compound'] >= 0),
                    alpha=0.3, color=COLORS['positive'], label='Positive')
    ax.fill_between(ts.index, ts['vader_compound'], 0,
                    where=(ts['vader_compound'] < 0),
                    alpha=0.3, color=COLORS['negative'], label='Negative')

    # Add event markers
    if events:
        for date_str, event_name in events.items():
            try:
                event_date = pd.to_datetime(date_str)
                if ts.index.min() <= event_date <= ts.index.max():
                    ax.axvline(x=event_date, color='red', linestyle='--', alpha=0.7)
                    ax.annotate(event_name[:30], xy=(event_date, ax.get_ylim()[1]),
                               rotation=45, fontsize=8, ha='left')
            except:
                pass

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sentiment Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_sentiment_distribution(df: pd.DataFrame,
                                 title: str = 'Sentiment Distribution',
                                 figsize: tuple = (10, 6),
                                 save_path: str = None) -> plt.Figure:
    """Plot histogram of sentiment scores"""
    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    ax.hist(df['vader_compound'], bins=50, color=COLORS['primary'],
            alpha=0.7, edgecolor='white')

    # Add vertical lines for thresholds
    ax.axvline(x=-0.05, color=COLORS['negative'], linestyle='--',
               label='Negative threshold')
    ax.axvline(x=0.05, color=COLORS['positive'], linestyle='--',
               label='Positive threshold')

    ax.set_xlabel('VADER Compound Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_sentiment_by_subreddit(df: pd.DataFrame,
                                 title: str = 'Sentiment by Subreddit',
                                 figsize: tuple = (12, 6),
                                 save_path: str = None) -> plt.Figure:
    """Compare sentiment across subreddits"""
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate
    sub_sentiment = df.groupby('subreddit')['vader_compound'].agg(['mean', 'std', 'count'])
    sub_sentiment = sub_sentiment[sub_sentiment['count'] >= 10]  # Min 10 posts
    sub_sentiment = sub_sentiment.sort_values('mean')

    # Bar plot with error bars
    colors = [COLORS['positive'] if m >= 0 else COLORS['negative']
              for m in sub_sentiment['mean']]

    bars = ax.barh(sub_sentiment.index, sub_sentiment['mean'],
                   xerr=sub_sentiment['std'], color=colors, alpha=0.7)

    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Mean Sentiment Score', fontsize=12)
    ax.set_ylabel('Subreddit', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Add count labels
    for i, (idx, row) in enumerate(sub_sentiment.iterrows()):
        ax.annotate(f'n={int(row["count"])}',
                   xy=(row['mean'], i), ha='left', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_topic_distribution(df: pd.DataFrame,
                            topic_labels: Dict[int, str] = None,
                            title: str = 'Topic Distribution',
                            figsize: tuple = (10, 6),
                            save_path: str = None) -> plt.Figure:
    """Plot distribution of topics"""
    fig, ax = plt.subplots(figsize=figsize)

    topic_counts = df['topic_id'].value_counts().sort_index()

    # Use labels if provided
    if topic_labels:
        labels = [topic_labels.get(i, f'Topic {i}') for i in topic_counts.index]
    else:
        labels = [f'Topic {i}' for i in topic_counts.index]

    colors = plt.cm.Set2(np.linspace(0, 1, len(topic_counts)))

    wedges, texts, autotexts = ax.pie(topic_counts, labels=labels,
                                       autopct='%1.1f%%', colors=colors,
                                       pctdistance=0.8)

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_topic_trends(df: pd.DataFrame, freq: str = 'M',
                      topic_labels: Dict[int, str] = None,
                      title: str = 'Topic Trends Over Time',
                      figsize: tuple = (14, 6),
                      save_path: str = None) -> plt.Figure:
    """Plot topic trends over time"""
    fig, ax = plt.subplots(figsize=figsize)

    # Group by time and topic
    df_copy = df.copy()
    df_copy['period'] = df_copy['datetime'].dt.to_period(freq)

    topic_trends = df_copy.groupby(['period', 'topic_id']).size().unstack(fill_value=0)
    topic_trends.index = topic_trends.index.to_timestamp()

    # Convert to percentages
    topic_pct = topic_trends.div(topic_trends.sum(axis=1), axis=0) * 100

    # Plot stacked area
    if topic_labels:
        labels = [topic_labels.get(i, f'Topic {i}') for i in topic_pct.columns]
    else:
        labels = [f'Topic {i}' for i in topic_pct.columns]

    ax.stackplot(topic_pct.index, topic_pct.T.values, labels=labels,
                 alpha=0.8)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Topic Share (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def generate_wordcloud(texts: List[str], title: str = 'Word Cloud',
                       figsize: tuple = (12, 8),
                       save_path: str = None) -> plt.Figure:
    """Generate word cloud from texts"""
    if not WORDCLOUD_AVAILABLE:
        print("WordCloud not installed. Run: pip install wordcloud")
        return None

    fig, ax = plt.subplots(figsize=figsize)

    # Combine texts
    all_text = ' '.join(texts)

    # Generate word cloud
    wc = WordCloud(width=1200, height=800,
                   background_color='white',
                   max_words=100,
                   colormap='viridis',
                   collocations=False).generate(all_text)

    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def create_dashboard(df: pd.DataFrame,
                     events: Dict[str, str] = None,
                     output_dir: str = 'outputs/figures',
                     topic_labels: Dict[int, str] = None) -> None:
    """
    Generate all visualizations and save to directory

    Args:
        df: Complete analyzed DataFrame
        events: Key events dict
        output_dir: Directory to save figures
        topic_labels: Optional topic label mapping
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualizations...")

    # 1. Post volume
    plot_post_volume(df, save_path=f'{output_dir}/01_post_volume.png')

    # 2. Sentiment trend
    plot_sentiment_trend(df, events=events,
                         save_path=f'{output_dir}/02_sentiment_trend.png')

    # 3. Sentiment distribution
    plot_sentiment_distribution(df,
                                 save_path=f'{output_dir}/03_sentiment_distribution.png')

    # 4. Sentiment by subreddit
    plot_sentiment_by_subreddit(df,
                                 save_path=f'{output_dir}/04_sentiment_by_subreddit.png')

    # 5. Topic distribution (if topics exist)
    if 'topic_id' in df.columns:
        plot_topic_distribution(df, topic_labels=topic_labels,
                                save_path=f'{output_dir}/05_topic_distribution.png')

        plot_topic_trends(df, topic_labels=topic_labels,
                          save_path=f'{output_dir}/06_topic_trends.png')

    # 6. Word cloud
    if WORDCLOUD_AVAILABLE:
        generate_wordcloud(df['text_combined'].tolist(),
                           save_path=f'{output_dir}/07_wordcloud.png')

    print(f"\nAll visualizations saved to: {output_dir}/")


if __name__ == "__main__":
    print("Visualization module loaded.")
    print(f"WordCloud available: {WORDCLOUD_AVAILABLE}")
