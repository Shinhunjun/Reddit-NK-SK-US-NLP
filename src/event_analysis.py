"""
Event-Centered Hourly Analysis for Reddit US-NK Data
Analyzes discourse patterns around key geopolitical events
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# Key Events for Korean Peninsula Security
KEY_EVENTS = {
    # Obama Era
    "2009-05-25": {
        "name": "NK 2nd Nuclear Test",
        "era": "obama",
        "type": "provocation",
        "window_hours": 72
    },
    "2010-03-26": {
        "name": "Cheonan Sinking",
        "era": "obama",
        "type": "provocation",
        "window_hours": 72
    },
    "2010-11-23": {
        "name": "Yeonpyeong Shelling",
        "era": "obama",
        "type": "provocation",
        "window_hours": 72
    },
    "2011-12-19": {
        "name": "Kim Jong-il Death",
        "era": "obama",
        "type": "transition",
        "window_hours": 96
    },
    "2013-02-12": {
        "name": "NK 3rd Nuclear Test",
        "era": "obama",
        "type": "provocation",
        "window_hours": 72
    },
    "2016-01-06": {
        "name": "NK 4th Nuclear Test (H-bomb claim)",
        "era": "obama",
        "type": "provocation",
        "window_hours": 72
    },
    "2016-09-09": {
        "name": "NK 5th Nuclear Test",
        "era": "obama",
        "type": "provocation",
        "window_hours": 72
    },

    # Trump Era
    "2017-08-08": {
        "name": "Fire and Fury Speech",
        "era": "trump",
        "type": "rhetoric",
        "window_hours": 72
    },
    "2017-09-03": {
        "name": "NK 6th Nuclear Test",
        "era": "trump",
        "type": "provocation",
        "window_hours": 72
    },
    "2017-11-29": {
        "name": "Hwasong-15 ICBM Test",
        "era": "trump",
        "type": "provocation",
        "window_hours": 72
    },
    "2018-06-12": {
        "name": "Singapore Summit",
        "era": "trump",
        "type": "diplomacy",
        "window_hours": 96
    },
    "2019-02-28": {
        "name": "Hanoi Summit Failure",
        "era": "trump",
        "type": "diplomacy",
        "window_hours": 72
    },
    "2019-06-30": {
        "name": "DMZ Meeting (Trump-Kim)",
        "era": "trump",
        "type": "diplomacy",
        "window_hours": 72
    },

    # Biden Era
    "2022-03-24": {
        "name": "Hwasong-17 ICBM Test",
        "era": "biden",
        "type": "provocation",
        "window_hours": 72
    },
    "2022-10-04": {
        "name": "NK Missile Over Japan",
        "era": "biden",
        "type": "provocation",
        "window_hours": 72
    },
    "2023-08-18": {
        "name": "Camp David Summit (US-ROK-Japan)",
        "era": "biden",
        "type": "diplomacy",
        "window_hours": 96
    },
    "2023-11-21": {
        "name": "NK Satellite Launch",
        "era": "biden",
        "type": "provocation",
        "window_hours": 72
    },
}


class EventAnalyzer:
    """
    Analyze Reddit discourse around key geopolitical events
    """

    def __init__(self, df: pd.DataFrame, date_column: str = 'date'):
        """
        Initialize event analyzer

        Args:
            df: DataFrame with Reddit posts
            date_column: Column containing datetime
        """
        self.df = df.copy()
        self.date_column = date_column

        # Ensure datetime type
        self.df[date_column] = pd.to_datetime(self.df[date_column])

        # Add hour column for hourly analysis
        self.df['hour'] = self.df[date_column].dt.hour
        self.df['datetime'] = self.df[date_column]

    def get_event_window(
        self,
        event_date: str,
        hours_before: int = 24,
        hours_after: int = 48
    ) -> pd.DataFrame:
        """
        Get posts within a time window around an event

        Args:
            event_date: Event date string (YYYY-MM-DD)
            hours_before: Hours before event to include
            hours_after: Hours after event to include

        Returns:
            DataFrame with posts in the window
        """
        event_dt = pd.to_datetime(event_date)
        start = event_dt - timedelta(hours=hours_before)
        end = event_dt + timedelta(hours=hours_after)

        mask = (self.df['datetime'] >= start) & (self.df['datetime'] <= end)
        return self.df[mask].copy()

    def analyze_event(
        self,
        event_date: str,
        event_info: Dict,
        sentiment_column: str = 'sentiment_compound'
    ) -> Dict:
        """
        Comprehensive analysis of a single event

        Args:
            event_date: Event date string
            event_info: Event metadata dict
            sentiment_column: Column for sentiment scores

        Returns:
            Dict with analysis results
        """
        hours_before = 24
        hours_after = event_info.get('window_hours', 72)

        window_df = self.get_event_window(event_date, hours_before, hours_after)

        if len(window_df) == 0:
            return {
                'event_date': event_date,
                'event_name': event_info['name'],
                'posts_count': 0,
                'before_count': 0,
                'after_count': 0,
                'sentiment_before': None,
                'sentiment_after': None,
                'sentiment_change': None,
                'peak_hour': None
            }

        event_dt = pd.to_datetime(event_date)

        # Split before/after
        before_mask = window_df['datetime'] < event_dt
        after_mask = window_df['datetime'] >= event_dt

        before_df = window_df[before_mask]
        after_df = window_df[after_mask]

        # Calculate metrics
        before_sentiment = before_df[sentiment_column].mean() if len(before_df) > 0 else 0
        after_sentiment = after_df[sentiment_column].mean() if len(after_df) > 0 else 0

        # Hourly counts for peak detection
        window_df['hours_from_event'] = (
            (window_df['datetime'] - event_dt).dt.total_seconds() / 3600
        ).round()

        hourly_counts = window_df.groupby('hours_from_event').size()
        peak_hour = hourly_counts.idxmax() if len(hourly_counts) > 0 else 0

        return {
            'event_date': event_date,
            'event_name': event_info['name'],
            'event_type': event_info.get('type', 'unknown'),
            'era': event_info.get('era', 'unknown'),
            'posts_count': len(window_df),
            'before_count': len(before_df),
            'after_count': len(after_df),
            'sentiment_before': before_sentiment,
            'sentiment_after': after_sentiment,
            'sentiment_change': after_sentiment - before_sentiment,
            'peak_hour': peak_hour,
            'hourly_data': hourly_counts.to_dict()
        }

    def analyze_all_events(
        self,
        events: Dict = None,
        sentiment_column: str = 'sentiment_compound'
    ) -> pd.DataFrame:
        """
        Analyze all key events

        Args:
            events: Dict of events (default: KEY_EVENTS)
            sentiment_column: Sentiment column name

        Returns:
            DataFrame with analysis results for each event
        """
        events = events or KEY_EVENTS
        results = []

        for date, info in events.items():
            result = self.analyze_event(date, info, sentiment_column)
            results.append(result)

        return pd.DataFrame(results)

    def visualize_event_hourly(
        self,
        event_date: str,
        event_info: Dict,
        sentiment_column: str = 'sentiment_compound',
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize hourly activity around a single event

        Args:
            event_date: Event date string
            event_info: Event metadata
            sentiment_column: Sentiment column
            output_path: Optional save path
        """
        hours_before = 24
        hours_after = event_info.get('window_hours', 72)

        window_df = self.get_event_window(event_date, hours_before, hours_after)

        if len(window_df) == 0:
            print(f"No data found for event: {event_info['name']}")
            return None

        event_dt = pd.to_datetime(event_date)
        window_df['hours_from_event'] = (
            (window_df['datetime'] - event_dt).dt.total_seconds() / 3600
        )

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

        # 1. Hourly post count
        hourly_counts = window_df.groupby(
            window_df['hours_from_event'].round()
        ).size()

        ax1.bar(hourly_counts.index, hourly_counts.values, color='steelblue', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Event Time')
        ax1.set_ylabel('Number of Posts')
        ax1.set_title(f'{event_info["name"]} ({event_date})\nHourly Post Volume')
        ax1.legend()

        # 2. Rolling average sentiment
        hourly_sentiment = window_df.groupby(
            window_df['hours_from_event'].round()
        )[sentiment_column].mean()

        ax2.plot(hourly_sentiment.index, hourly_sentiment.values,
                 'b-', linewidth=2, marker='o', markersize=4)
        ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.fill_between(hourly_sentiment.index, 0, hourly_sentiment.values,
                         where=(hourly_sentiment.values > 0), alpha=0.3, color='green')
        ax2.fill_between(hourly_sentiment.index, 0, hourly_sentiment.values,
                         where=(hourly_sentiment.values < 0), alpha=0.3, color='red')
        ax2.set_xlabel('Hours from Event')
        ax2.set_ylabel('Average Sentiment')
        ax2.set_title('Hourly Sentiment Trend')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig

    def visualize_all_events_comparison(
        self,
        events: Dict = None,
        sentiment_column: str = 'sentiment_compound',
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare sentiment changes across all events

        Args:
            events: Dict of events
            sentiment_column: Sentiment column
            output_path: Optional save path
        """
        events = events or KEY_EVENTS
        results_df = self.analyze_all_events(events, sentiment_column)

        # Filter events with data
        results_df = results_df[results_df['posts_count'] > 0]

        if len(results_df) == 0:
            print("No events with data found")
            return None

        # Sort by date
        results_df['event_date'] = pd.to_datetime(results_df['event_date'])
        results_df = results_df.sort_values('event_date')

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Sentiment change by event
        ax1 = axes[0, 0]
        colors = ['green' if x > 0 else 'red' for x in results_df['sentiment_change'].fillna(0)]
        bars = ax1.barh(range(len(results_df)), results_df['sentiment_change'].fillna(0), color=colors)
        ax1.set_yticks(range(len(results_df)))
        ax1.set_yticklabels(results_df['event_name'], fontsize=8)
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_xlabel('Sentiment Change (After - Before)')
        ax1.set_title('Sentiment Change by Event')

        # 2. Post volume by event
        ax2 = axes[0, 1]
        colors_era = {'obama': 'blue', 'trump': 'red', 'biden': 'green'}
        bar_colors = [colors_era.get(era, 'gray') for era in results_df['era']]
        ax2.barh(range(len(results_df)), results_df['posts_count'], color=bar_colors)
        ax2.set_yticks(range(len(results_df)))
        ax2.set_yticklabels(results_df['event_name'], fontsize=8)
        ax2.set_xlabel('Number of Posts')
        ax2.set_title('Post Volume by Event')

        # Legend for eras
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, label=e.title()) for e, c in colors_era.items()]
        ax2.legend(handles=legend_elements, loc='lower right')

        # 3. Sentiment change by event type
        ax3 = axes[1, 0]
        type_sentiment = results_df.groupby('event_type')['sentiment_change'].mean()
        colors_type = {'provocation': 'red', 'diplomacy': 'blue', 'rhetoric': 'orange',
                       'transition': 'purple', 'unknown': 'gray'}
        bar_colors = [colors_type.get(t, 'gray') for t in type_sentiment.index]
        ax3.bar(type_sentiment.index, type_sentiment.values, color=bar_colors)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Event Type')
        ax3.set_ylabel('Average Sentiment Change')
        ax3.set_title('Average Sentiment Change by Event Type')

        # 4. Timeline of events
        ax4 = axes[1, 1]
        for i, row in results_df.iterrows():
            color = colors_era.get(row['era'], 'gray')
            marker = 'o' if row['event_type'] == 'diplomacy' else 's'
            ax4.scatter(row['event_date'], row['sentiment_change'],
                        c=color, s=row['posts_count']/10, marker=marker, alpha=0.7)

        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sentiment Change')
        ax4.set_title('Event Timeline (size = post volume)')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig


def run_event_analysis(
    csv_path: str,
    date_column: str = 'date',
    sentiment_column: str = 'sentiment_compound',
    output_dir: str = '../outputs/figures'
) -> Tuple[pd.DataFrame, EventAnalyzer]:
    """
    Run full event analysis on Reddit data

    Args:
        csv_path: Path to processed CSV
        date_column: Date column name
        sentiment_column: Sentiment column name
        output_dir: Output directory

    Returns:
        Tuple of (results DataFrame, EventAnalyzer instance)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} posts")

    # Initialize analyzer
    analyzer = EventAnalyzer(df, date_column=date_column)

    # Analyze all events
    print("\nAnalyzing events...")
    results = analyzer.analyze_all_events(KEY_EVENTS, sentiment_column)

    # Print summary
    print("\n" + "="*60)
    print("EVENT ANALYSIS SUMMARY")
    print("="*60)

    results_with_data = results[results['posts_count'] > 0]
    print(f"\nEvents with data: {len(results_with_data)}/{len(results)}")

    for _, row in results_with_data.iterrows():
        print(f"\n{row['event_name']} ({row['event_date'][:10]}):")
        print(f"  Posts: {row['posts_count']}")
        print(f"  Sentiment change: {row['sentiment_change']:.3f}" if row['sentiment_change'] else "  No sentiment data")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Overall comparison
    analyzer.visualize_all_events_comparison(
        KEY_EVENTS,
        sentiment_column,
        output_path=os.path.join(output_dir, 'events_comparison.png')
    )

    # Individual event visualizations (top 5 by post count)
    top_events = results_with_data.nlargest(5, 'posts_count')

    for _, row in top_events.iterrows():
        event_date = row['event_date']
        if isinstance(event_date, pd.Timestamp):
            event_date = event_date.strftime('%Y-%m-%d')

        if event_date in KEY_EVENTS:
            safe_name = row['event_name'].replace(' ', '_').replace('(', '').replace(')', '')
            analyzer.visualize_event_hourly(
                event_date,
                KEY_EVENTS[event_date],
                sentiment_column,
                output_path=os.path.join(output_dir, f'event_hourly_{safe_name}.png')
            )

    # Save results
    output_csv = os.path.join(output_dir, 'event_analysis_results.csv')
    results.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    return results, analyzer


if __name__ == "__main__":
    print("Event Analysis for Reddit US-NK Data")
    print("="*50)

    data_path = "../data/processed/posts_final.csv"

    if os.path.exists(data_path):
        results, analyzer = run_event_analysis(
            csv_path=data_path,
            date_column='date',
            sentiment_column='sentiment_compound',
            output_dir='../outputs/figures'
        )
    else:
        print(f"Data file not found: {data_path}")
