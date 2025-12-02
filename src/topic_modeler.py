"""
Topic Modeling Module
- LDA (Latent Dirichlet Allocation) - lightweight, interpretable
- BERTopic - transformer-based, better quality (optional)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import Counter

# Sklearn for LDA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Optional: Gensim for alternative LDA
try:
    import gensim
    from gensim import corpora
    from gensim.models import LdaModel
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Optional: BERTopic for advanced topic modeling
try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    print("BERTopic not installed. Using LDA only. Install: pip install bertopic")


class LDATopicModeler:
    """
    LDA-based topic modeling using sklearn
    Lightweight and interpretable
    """

    def __init__(self, n_topics: int = 6, max_features: int = 5000):
        self.n_topics = n_topics
        self.max_features = max_features
        self.vectorizer = None
        self.lda_model = None
        self.feature_names = None

    def fit(self, texts: List[str], random_state: int = 42) -> 'LDATopicModeler':
        """
        Fit LDA model on texts

        Args:
            texts: List of document texts
            random_state: For reproducibility

        Returns:
            self
        """
        # Vectorize
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words='english',
            max_df=0.95,  # Ignore terms in >95% of docs
            min_df=5,     # Ignore terms in <5 docs
            ngram_range=(1, 2)  # Include bigrams
        )

        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Fit LDA
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=random_state,
            max_iter=20,
            learning_method='online',
            learning_offset=50.,
            n_jobs=-1
        )

        self.lda_model.fit(doc_term_matrix)

        print(f"LDA model fitted with {self.n_topics} topics")
        return self

    def get_topics(self, n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top words for each topic

        Returns:
            Dict mapping topic_id to list of (word, weight) tuples
        """
        topics = {}

        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get indices of top words
            top_indices = topic.argsort()[:-n_words-1:-1]

            # Get words and weights
            topic_words = [
                (self.feature_names[i], topic[i])
                for i in top_indices
            ]
            topics[topic_idx] = topic_words

        return topics

    def print_topics(self, n_words: int = 10):
        """Print topics in readable format"""
        topics = self.get_topics(n_words)

        print(f"\n{'='*60}")
        print(f"LDA TOPICS (n={self.n_topics})")
        print('='*60)

        for topic_id, words in topics.items():
            word_str = ', '.join([w[0] for w in words])
            print(f"\nTopic {topic_id}: {word_str}")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Get topic distribution for new texts

        Returns:
            Array of shape (n_docs, n_topics)
        """
        doc_term_matrix = self.vectorizer.transform(texts)
        return self.lda_model.transform(doc_term_matrix)

    def get_dominant_topic(self, texts: List[str]) -> List[int]:
        """Get dominant topic ID for each text"""
        topic_dist = self.transform(texts)
        return topic_dist.argmax(axis=1).tolist()

    def add_topics_to_df(self, df: pd.DataFrame, text_column: str = 'text_combined') -> pd.DataFrame:
        """
        Add topic assignments to DataFrame

        Args:
            df: DataFrame with text column
            text_column: Name of text column

        Returns:
            DataFrame with 'topic_id' and 'topic_prob' columns
        """
        df = df.copy()
        texts = df[text_column].tolist()

        # Get topic distributions
        topic_dist = self.transform(texts)

        # Add columns
        df['topic_id'] = topic_dist.argmax(axis=1)
        df['topic_prob'] = topic_dist.max(axis=1)

        # Topic distribution summary
        topic_counts = df['topic_id'].value_counts().sort_index()
        print(f"\nTopic Distribution:")
        for topic_id, count in topic_counts.items():
            print(f"  Topic {topic_id}: {count} posts ({count/len(df)*100:.1f}%)")

        return df


class BERTopicModeler:
    """
    BERTopic-based topic modeling
    Uses sentence transformers for better semantic understanding
    """

    def __init__(self, n_topics: int = 6):
        if not BERTOPIC_AVAILABLE:
            raise ImportError("BERTopic not installed. Run: pip install bertopic")

        self.n_topics = n_topics
        self.model = None

    def fit(self, texts: List[str]) -> 'BERTopicModeler':
        """Fit BERTopic model"""
        self.model = BERTopic(
            nr_topics=self.n_topics,
            verbose=True,
            calculate_probabilities=True
        )
        self.topics, self.probs = self.model.fit_transform(texts)

        print(f"BERTopic model fitted")
        return self

    def get_topics(self) -> Dict:
        """Get topic information"""
        return self.model.get_topic_info()

    def print_topics(self):
        """Print topics"""
        topic_info = self.model.get_topic_info()
        print(f"\n{'='*60}")
        print("BERTOPIC TOPICS")
        print('='*60)
        print(topic_info)

    def add_topics_to_df(self, df: pd.DataFrame, text_column: str = 'text_combined') -> pd.DataFrame:
        """Add topic assignments to DataFrame"""
        df = df.copy()
        texts = df[text_column].tolist()

        topics, probs = self.model.transform(texts)
        df['topic_id'] = topics
        df['topic_prob'] = [p.max() if len(p) > 0 else 0 for p in probs]

        return df


def get_topic_trends(df: pd.DataFrame, freq: str = 'M') -> pd.DataFrame:
    """
    Analyze topic trends over time

    Args:
        df: DataFrame with 'datetime' and 'topic_id'
        freq: Time frequency ('W', 'M')

    Returns:
        Pivot table of topic counts by time period
    """
    # Group by time and topic
    df_copy = df.copy()
    df_copy['period'] = df_copy['datetime'].dt.to_period(freq)

    topic_trends = df_copy.groupby(['period', 'topic_id']).size().unstack(fill_value=0)

    # Convert to percentages
    topic_pct = topic_trends.div(topic_trends.sum(axis=1), axis=0) * 100

    return topic_pct


def get_topic_by_subreddit(df: pd.DataFrame) -> pd.DataFrame:
    """Compare topic distribution across subreddits"""
    return pd.crosstab(df['subreddit'], df['topic_id'], normalize='index') * 100


if __name__ == "__main__":
    # Test with sample data
    sample_texts = [
        "North Korea launches ballistic missile over Japan threatening regional security",
        "US and South Korea strengthen military alliance with joint exercises",
        "Kim Jong Un threatens nuclear response to US military presence",
        "Diplomatic talks between North Korea and US stall over sanctions",
        "South Korean president visits Washington to discuss peninsula security",
        "North Korea conducts nuclear test defying international sanctions",
        "US deploys THAAD missile defense system to South Korea",
        "China opposes US military buildup near Korean peninsula",
        "Japan increases defense budget citing North Korean threat",
        "Inter-Korean summit aims for peace agreement",
    ]

    print("Testing LDA Topic Modeling")
    print("="*60)

    modeler = LDATopicModeler(n_topics=3)
    modeler.fit(sample_texts)
    modeler.print_topics(n_words=5)

    # Get dominant topics
    topics = modeler.get_dominant_topic(sample_texts)
    print(f"\nDocument topic assignments: {topics}")
