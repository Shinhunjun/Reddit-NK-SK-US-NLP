"""
BERTopic-based Topic Modeling for Reddit US-NK Analysis
Optimized for M4 Pro (Apple Silicon)
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple, Optional
from tqdm import tqdm

# BERTopic imports
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualization
import plotly.express as px
import plotly.graph_objects as go


class BERTopicModeler:
    """
    BERTopic-based topic modeling with interactive visualizations
    Optimized for M4 Pro Mac
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        n_topics: Optional[int] = None,
        min_topic_size: int = 30,
        verbose: bool = True
    ):
        """
        Initialize BERTopic modeler

        Args:
            embedding_model: Sentence transformer model name (default: all-MiniLM-L6-v2, 22M params)
            n_topics: Number of topics (None = auto-detect)
            min_topic_size: Minimum documents per topic
            verbose: Show progress
        """
        self.verbose = verbose

        if self.verbose:
            print(f"Loading embedding model: {embedding_model}")

        # Sentence transformer for embeddings
        self.sentence_model = SentenceTransformer(embedding_model)

        # UMAP for dimensionality reduction (optimized for CPU)
        umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Vectorizer for topic representation
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=5
        )

        # Initialize BERTopic
        self.model = BERTopic(
            embedding_model=self.sentence_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=n_topics,
            top_n_words=10,
            verbose=verbose,
            calculate_probabilities=True
        )

        self.topics = None
        self.probs = None
        self.docs = None
        self.timestamps = None

    def fit(self, documents: List[str], timestamps: Optional[List[datetime]] = None) -> 'BERTopicModeler':
        """
        Fit BERTopic model on documents

        Args:
            documents: List of text documents
            timestamps: Optional list of datetime objects for temporal analysis
        """
        if self.verbose:
            print(f"\nFitting BERTopic on {len(documents):,} documents...")

        self.docs = documents
        self.timestamps = timestamps

        # Fit model
        self.topics, self.probs = self.model.fit_transform(documents)

        if self.verbose:
            topic_info = self.model.get_topic_info()
            n_topics = len(topic_info) - 1  # Exclude outlier topic (-1)
            print(f"\nDiscovered {n_topics} topics")
            print(f"Outlier documents: {sum(t == -1 for t in self.topics):,}")

        return self

    def get_topic_info(self) -> pd.DataFrame:
        """Get topic information including keywords and counts"""
        return self.model.get_topic_info()

    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """Get top keywords for a specific topic"""
        topic = self.model.get_topic(topic_id)
        return topic[:n_words] if topic else []

    def print_topics(self, n_words: int = 8):
        """Print all topics with their top words"""
        topic_info = self.get_topic_info()

        print("\n" + "="*60)
        print("BERTOPIC TOPICS")
        print("="*60)

        for _, row in topic_info.iterrows():
            if row['Topic'] == -1:
                continue
            print(f"\nTopic {row['Topic']} ({row['Count']:,} docs):")
            print(f"  Keywords: {row['Name']}")

    def get_document_topics(self) -> pd.DataFrame:
        """Get topic assignments for all documents"""
        return pd.DataFrame({
            'document': self.docs,
            'topic': self.topics,
            'probability': [max(p) if len(p) > 0 else 0 for p in self.probs]
        })

    # ============ VISUALIZATIONS ============

    def visualize_topics(self, output_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive 2D topic cluster visualization

        Args:
            output_path: Optional path to save HTML file
        """
        fig = self.model.visualize_topics()

        if output_path:
            fig.write_html(output_path)
            if self.verbose:
                print(f"Saved topic clusters to {output_path}")

        return fig

    def visualize_hierarchy(self, output_path: Optional[str] = None) -> go.Figure:
        """
        Create hierarchical topic clustering visualization
        """
        fig = self.model.visualize_hierarchy()

        if output_path:
            fig.write_html(output_path)
            if self.verbose:
                print(f"Saved topic hierarchy to {output_path}")

        return fig

    def visualize_barchart(self, n_topics: int = 10, output_path: Optional[str] = None) -> go.Figure:
        """
        Create bar chart showing top words per topic
        """
        fig = self.model.visualize_barchart(top_n_topics=n_topics)

        if output_path:
            fig.write_html(output_path)
            if self.verbose:
                print(f"Saved topic barchart to {output_path}")

        return fig

    def visualize_heatmap(self, output_path: Optional[str] = None) -> go.Figure:
        """
        Create topic similarity heatmap
        """
        fig = self.model.visualize_heatmap()

        if output_path:
            fig.write_html(output_path)
            if self.verbose:
                print(f"Saved topic heatmap to {output_path}")

        return fig

    def visualize_topics_over_time(
        self,
        timestamps: Optional[List[datetime]] = None,
        nr_bins: int = 20,
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create topics over time visualization

        Args:
            timestamps: List of datetime objects (uses stored if None)
            nr_bins: Number of time bins
            output_path: Optional path to save HTML
        """
        timestamps = timestamps or self.timestamps

        if timestamps is None:
            raise ValueError("Timestamps required for temporal analysis")

        # Calculate topics over time
        topics_over_time = self.model.topics_over_time(
            self.docs,
            timestamps,
            nr_bins=nr_bins
        )

        fig = self.model.visualize_topics_over_time(topics_over_time)

        if output_path:
            fig.write_html(output_path)
            if self.verbose:
                print(f"Saved topics over time to {output_path}")

        return fig

    def visualize_topics_by_era(
        self,
        df: pd.DataFrame,
        era_column: str = 'era',
        output_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create topic distribution by presidential era

        Args:
            df: DataFrame with era column
            era_column: Name of era column
            output_path: Optional save path
        """
        # Calculate topic counts by era
        df_with_topics = df.copy()
        df_with_topics['topic'] = self.topics

        topic_era_counts = df_with_topics.groupby([era_column, 'topic']).size().reset_index(name='count')

        # Normalize within each era
        era_totals = topic_era_counts.groupby(era_column)['count'].transform('sum')
        topic_era_counts['percentage'] = topic_era_counts['count'] / era_totals * 100

        # Get topic names
        topic_info = self.get_topic_info()
        topic_names = {row['Topic']: row['Name'][:30] for _, row in topic_info.iterrows()}
        topic_era_counts['topic_name'] = topic_era_counts['topic'].map(topic_names)

        # Filter out outlier topic
        topic_era_counts = topic_era_counts[topic_era_counts['topic'] != -1]

        # Create grouped bar chart
        fig = px.bar(
            topic_era_counts,
            x='topic_name',
            y='percentage',
            color=era_column,
            barmode='group',
            title='Topic Distribution by Presidential Era',
            labels={'percentage': 'Percentage (%)', 'topic_name': 'Topic'}
        )

        fig.update_layout(
            xaxis_tickangle=-45,
            height=600,
            legend_title="Era"
        )

        if output_path:
            fig.write_html(output_path)
            if self.verbose:
                print(f"Saved topics by era to {output_path}")

        return fig

    def save_model(self, path: str):
        """Save the trained BERTopic model"""
        self.model.save(path)
        if self.verbose:
            print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load a saved BERTopic model"""
        self.model = BERTopic.load(path)
        if self.verbose:
            print(f"Model loaded from {path}")


def analyze_with_bertopic(
    csv_path: str,
    text_column: str = 'text_combined',
    date_column: str = 'date',
    output_dir: str = '../outputs/figures'
) -> Tuple[BERTopicModeler, pd.DataFrame]:
    """
    Run full BERTopic analysis on Reddit data

    Args:
        csv_path: Path to processed CSV file
        text_column: Column containing text
        date_column: Column containing dates
        output_dir: Directory to save visualizations
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)

    # Clean text - remove NaN and empty strings
    df = df.dropna(subset=[text_column])
    df = df[df[text_column].str.len() > 10]

    print(f"Analyzing {len(df):,} documents")

    # Parse dates
    df[date_column] = pd.to_datetime(df[date_column])

    # Initialize and fit model
    modeler = BERTopicModeler(
        embedding_model="all-MiniLM-L6-v2",
        min_topic_size=30
    )

    modeler.fit(
        documents=df[text_column].tolist(),
        timestamps=df[date_column].tolist()
    )

    # Print topics
    modeler.print_topics()

    # Generate visualizations
    print("\nGenerating visualizations...")

    modeler.visualize_topics(
        output_path=os.path.join(output_dir, 'bertopic_clusters.html')
    )

    modeler.visualize_hierarchy(
        output_path=os.path.join(output_dir, 'bertopic_hierarchy.html')
    )

    modeler.visualize_barchart(
        n_topics=15,
        output_path=os.path.join(output_dir, 'bertopic_barchart.html')
    )

    modeler.visualize_heatmap(
        output_path=os.path.join(output_dir, 'bertopic_heatmap.html')
    )

    modeler.visualize_topics_over_time(
        nr_bins=30,
        output_path=os.path.join(output_dir, 'bertopic_over_time.html')
    )

    # Topics by era if era column exists
    if 'era' in df.columns:
        modeler.visualize_topics_by_era(
            df=df,
            era_column='era',
            output_path=os.path.join(output_dir, 'bertopic_by_era.html')
        )

    # Add topic assignments to dataframe
    df['bertopic_topic'] = modeler.topics
    df['bertopic_prob'] = [max(p) if len(p) > 0 else 0 for p in modeler.probs]

    # Save results
    output_csv = csv_path.replace('.csv', '_bertopic.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    return modeler, df


# Quick test
if __name__ == "__main__":
    print("BERTopic Analysis for Reddit US-NK Data")
    print("="*50)

    # Check if data exists
    data_path = "../data/processed/posts_final.csv"

    if os.path.exists(data_path):
        modeler, df = analyze_with_bertopic(
            csv_path=data_path,
            text_column='text_combined',
            date_column='date',
            output_dir='../outputs/figures'
        )

        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        print("\nVisualization files created in outputs/figures/:")
        print("  - bertopic_clusters.html")
        print("  - bertopic_hierarchy.html")
        print("  - bertopic_barchart.html")
        print("  - bertopic_heatmap.html")
        print("  - bertopic_over_time.html")
    else:
        print(f"Data file not found: {data_path}")
        print("Please run the data collection pipeline first.")
