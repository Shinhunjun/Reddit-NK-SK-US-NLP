"""
Structural Topic Modeling (STM) for Reddit US-NK Analysis
Python implementation using Guided LDA with covariates
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

# NLP
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


class STMAnalyzer:
    """
    Structural Topic Modeling implementation in Python
    Uses LDA with document-level covariates for prevalence analysis
    """

    def __init__(
        self,
        n_topics: int = 7,
        max_features: int = 5000,
        min_df: int = 5,
        max_df: float = 0.95,
        random_state: int = 42
    ):
        """
        Initialize STM analyzer

        Args:
            n_topics: Number of topics
            max_features: Max vocabulary size
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            random_state: Random seed
        """
        self.n_topics = n_topics
        self.random_state = random_state

        # Vectorizer
        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            ngram_range=(1, 2)
        )

        # LDA model
        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=50,
            learning_method='online',
            random_state=random_state,
            n_jobs=-1
        )

        self.doc_topic_dist = None
        self.feature_names = None

    def fit(self, documents: List[str]) -> 'STMAnalyzer':
        """
        Fit STM model on documents

        Args:
            documents: List of text documents
        """
        print(f"Fitting STM on {len(documents):,} documents with {self.n_topics} topics...")

        # Create document-term matrix
        dtm = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"Vocabulary size: {len(self.feature_names)}")

        # Fit LDA
        self.doc_topic_dist = self.lda.fit_transform(dtm)

        print("STM fitting complete")
        return self

    def get_topic_words(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """Get top words for a topic"""
        topic = self.lda.components_[topic_id]
        top_indices = topic.argsort()[:-n_words-1:-1]

        return [
            (self.feature_names[i], topic[i])
            for i in top_indices
        ]

    def print_topics(self, n_words: int = 10):
        """Print all topics with top words"""
        print("\n" + "="*60)
        print("STM TOPICS")
        print("="*60)

        for i in range(self.n_topics):
            words = self.get_topic_words(i, n_words)
            word_str = ", ".join([w[0] for w in words])
            print(f"\nTopic {i}: {word_str}")

    def get_document_topics(self) -> np.ndarray:
        """Get topic distributions for all documents"""
        return self.doc_topic_dist

    def analyze_topic_prevalence(
        self,
        df: pd.DataFrame,
        covariate_column: str,
        topic_labels: Optional[Dict[int, str]] = None
    ) -> pd.DataFrame:
        """
        Analyze how topic prevalence varies by covariate

        Args:
            df: DataFrame with documents
            covariate_column: Column to use as covariate (e.g., 'era', 'subreddit')
            topic_labels: Optional topic names

        Returns:
            DataFrame with topic prevalence by covariate
        """
        df = df.copy()

        # Assign dominant topic
        df['dominant_topic'] = self.doc_topic_dist.argmax(axis=1)

        # Add topic distribution columns
        for i in range(self.n_topics):
            label = topic_labels.get(i, f'Topic_{i}') if topic_labels else f'Topic_{i}'
            df[f'topic_{i}_prob'] = self.doc_topic_dist[:, i]

        # Calculate average topic distribution by covariate
        topic_cols = [f'topic_{i}_prob' for i in range(self.n_topics)]
        prevalence = df.groupby(covariate_column)[topic_cols].mean()

        return prevalence

    def analyze_temporal_prevalence(
        self,
        df: pd.DataFrame,
        date_column: str = 'date',
        freq: str = 'M'
    ) -> pd.DataFrame:
        """
        Analyze topic prevalence over time

        Args:
            df: DataFrame with documents
            date_column: Date column name
            freq: Time frequency ('M' = monthly, 'Q' = quarterly, 'Y' = yearly)

        Returns:
            DataFrame with temporal topic prevalence
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Add topic probabilities
        for i in range(self.n_topics):
            df[f'topic_{i}'] = self.doc_topic_dist[:, i]

        # Set date index
        df = df.set_index(date_column)

        # Resample and calculate mean topic proportions
        topic_cols = [f'topic_{i}' for i in range(self.n_topics)]
        temporal = df[topic_cols].resample(freq).mean()

        return temporal

    def estimate_topic_correlation(self) -> np.ndarray:
        """
        Estimate correlation between topics based on co-occurrence
        """
        return np.corrcoef(self.doc_topic_dist.T)

    def visualize_topic_prevalence_by_covariate(
        self,
        prevalence_df: pd.DataFrame,
        covariate_name: str,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic prevalence by covariate

        Args:
            prevalence_df: DataFrame from analyze_topic_prevalence
            covariate_name: Name of covariate for title
            output_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        # Plot as stacked bar
        prevalence_df.plot(kind='bar', stacked=True, ax=ax, colormap='tab10')

        ax.set_xlabel(covariate_name)
        ax.set_ylabel('Topic Proportion')
        ax.set_title(f'Topic Prevalence by {covariate_name}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Topics')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig

    def visualize_temporal_topics(
        self,
        temporal_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic evolution over time

        Args:
            temporal_df: DataFrame from analyze_temporal_prevalence
            output_path: Optional save path
        """
        fig, ax = plt.subplots(figsize=(14, 8))

        for col in temporal_df.columns:
            ax.plot(temporal_df.index, temporal_df[col], label=col, linewidth=2)

        ax.set_xlabel('Date')
        ax.set_ylabel('Topic Proportion')
        ax.set_title('Topic Evolution Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig

    def visualize_topic_correlation(
        self,
        topic_labels: Optional[Dict[int, str]] = None,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize topic correlation matrix

        Args:
            topic_labels: Optional topic names
            output_path: Optional save path
        """
        corr = self.estimate_topic_correlation()

        labels = [topic_labels.get(i, f'Topic {i}') for i in range(self.n_topics)] \
            if topic_labels else [f'Topic {i}' for i in range(self.n_topics)]

        fig, ax = plt.subplots(figsize=(10, 8))

        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    vmin=-1, vmax=1, center=0)

        ax.set_title('Topic Correlation Matrix')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig

    def compare_topic_prevalence(
        self,
        df: pd.DataFrame,
        group1_mask: pd.Series,
        group2_mask: pd.Series,
        group1_name: str = 'Group 1',
        group2_name: str = 'Group 2',
        output_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Compare topic prevalence between two groups

        Args:
            df: DataFrame with documents
            group1_mask: Boolean mask for group 1
            group2_mask: Boolean mask for group 2
            group1_name: Name for group 1
            group2_name: Name for group 2
            output_path: Optional save path

        Returns:
            Tuple of (comparison DataFrame, figure)
        """
        # Calculate mean topic proportions for each group
        group1_topics = self.doc_topic_dist[group1_mask].mean(axis=0)
        group2_topics = self.doc_topic_dist[group2_mask].mean(axis=0)

        comparison = pd.DataFrame({
            group1_name: group1_topics,
            group2_name: group2_topics,
            'Difference': group2_topics - group1_topics
        }, index=[f'Topic {i}' for i in range(self.n_topics)])

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Side by side comparison
        x = np.arange(self.n_topics)
        width = 0.35

        ax1.bar(x - width/2, group1_topics, width, label=group1_name)
        ax1.bar(x + width/2, group2_topics, width, label=group2_name)
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'T{i}' for i in range(self.n_topics)])
        ax1.set_ylabel('Mean Topic Proportion')
        ax1.set_title('Topic Prevalence Comparison')
        ax1.legend()

        # Difference plot
        colors = ['green' if d > 0 else 'red' for d in comparison['Difference']]
        ax2.bar(range(self.n_topics), comparison['Difference'], color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xticks(range(self.n_topics))
        ax2.set_xticklabels([f'T{i}' for i in range(self.n_topics)])
        ax2.set_ylabel('Difference in Prevalence')
        ax2.set_title(f'{group2_name} - {group1_name}')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return comparison, fig


def run_stm_analysis(
    csv_path: str,
    text_column: str = 'text_combined',
    date_column: str = 'date',
    n_topics: int = 7,
    output_dir: str = '../outputs/figures'
) -> Tuple[STMAnalyzer, pd.DataFrame]:
    """
    Run full STM analysis

    Args:
        csv_path: Path to processed CSV
        text_column: Text column name
        date_column: Date column name
        n_topics: Number of topics
        output_dir: Output directory

    Returns:
        Tuple of (STMAnalyzer, results DataFrame)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[text_column])
    print(f"Loaded {len(df):,} documents")

    # Initialize and fit
    stm = STMAnalyzer(n_topics=n_topics)
    stm.fit(df[text_column].tolist())

    # Print topics
    stm.print_topics()

    # Analyze by era if available
    if 'era' in df.columns:
        print("\nAnalyzing topic prevalence by era...")
        era_prevalence = stm.analyze_topic_prevalence(df, 'era')
        print(era_prevalence)

        stm.visualize_topic_prevalence_by_covariate(
            era_prevalence,
            'Presidential Era',
            output_path=os.path.join(output_dir, 'stm_era_prevalence.png')
        )

    # Analyze by subreddit if available
    if 'subreddit' in df.columns:
        print("\nAnalyzing topic prevalence by subreddit...")
        sub_prevalence = stm.analyze_topic_prevalence(df, 'subreddit')
        print(sub_prevalence)

        stm.visualize_topic_prevalence_by_covariate(
            sub_prevalence,
            'Subreddit',
            output_path=os.path.join(output_dir, 'stm_subreddit_prevalence.png')
        )

    # Temporal analysis
    if date_column in df.columns:
        print("\nAnalyzing temporal topic evolution...")
        temporal = stm.analyze_temporal_prevalence(df, date_column, freq='Q')

        stm.visualize_temporal_topics(
            temporal,
            output_path=os.path.join(output_dir, 'stm_temporal_topics.png')
        )

    # Topic correlation
    stm.visualize_topic_correlation(
        output_path=os.path.join(output_dir, 'stm_topic_correlation.png')
    )

    # Compare eras if available
    if 'era' in df.columns:
        print("\nComparing Trump vs Biden era topics...")
        trump_mask = df['era'] == 'trump'
        biden_mask = df['era'] == 'biden'

        if trump_mask.sum() > 0 and biden_mask.sum() > 0:
            comparison, fig = stm.compare_topic_prevalence(
                df, trump_mask, biden_mask,
                'Trump Era', 'Biden Era',
                output_path=os.path.join(output_dir, 'stm_trump_vs_biden.png')
            )
            print(comparison)

    # Add topic assignments to dataframe
    df['stm_dominant_topic'] = stm.doc_topic_dist.argmax(axis=1)
    for i in range(n_topics):
        df[f'stm_topic_{i}'] = stm.doc_topic_dist[:, i]

    # Save results
    output_csv = csv_path.replace('.csv', '_stm.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    return stm, df


if __name__ == "__main__":
    print("STM Analysis for Reddit US-NK Data")
    print("="*50)

    data_path = "../data/processed/posts_final.csv"

    if os.path.exists(data_path):
        stm, df = run_stm_analysis(
            csv_path=data_path,
            text_column='text_combined',
            date_column='date',
            n_topics=7,
            output_dir='../outputs/figures'
        )
    else:
        print(f"Data file not found: {data_path}")
