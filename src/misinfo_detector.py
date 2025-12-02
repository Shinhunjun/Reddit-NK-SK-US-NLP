"""
Misinformation Detection using Vertex AI Gemini
LLM-based classification for Reddit posts about North Korea/US-ROK relations
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns


# Classification categories
MISINFO_CATEGORIES = [
    "FACTUAL",       # Verifiable facts, news reports
    "MISLEADING",    # Partially true but misleading context
    "OPINION",       # Personal opinion, subjective analysis
    "SATIRE",        # Humor, sarcasm, parody
    "UNVERIFIABLE"   # Claims that cannot be verified
]


class GeminiMisinfoDetector:
    """
    Misinformation detection using Vertex AI Gemini
    """

    def __init__(
        self,
        project_id: str = None,
        location: str = "us-central1",
        model_name: str = "gemini-1.5-flash-001",
        use_mock: bool = False
    ):
        """
        Initialize Gemini detector

        Args:
            project_id: GCP project ID
            location: GCP region
            model_name: Gemini model name
            use_mock: Use mock responses for testing (no API calls)
        """
        self.use_mock = use_mock
        self.model_name = model_name

        if use_mock:
            print("Using MOCK mode - no actual API calls")
            self.model = None
        else:
            try:
                import vertexai
                from vertexai.generative_models import GenerativeModel

                project_id = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
                if not project_id:
                    print("Warning: No project ID provided. Set GOOGLE_CLOUD_PROJECT env var.")
                    print("Switching to mock mode.")
                    self.use_mock = True
                    self.model = None
                else:
                    vertexai.init(project=project_id, location=location)
                    self.model = GenerativeModel(model_name)
                    print(f"Initialized Gemini model: {model_name}")
            except ImportError:
                print("Warning: google-cloud-aiplatform not installed.")
                print("Switching to mock mode.")
                self.use_mock = True
                self.model = None

        # Prompt template
        self.prompt_template = """You are an expert fact-checker analyzing social media posts about North Korea,
South Korea, and US-ROK relations.

Analyze the following Reddit post and classify it into ONE of these categories:
- FACTUAL: Contains verifiable facts, cites news sources, or reports confirmed events
- MISLEADING: Contains partial truths but misrepresents context, uses exaggeration, or makes unsupported claims
- OPINION: Personal viewpoint, analysis, or subjective interpretation without false claims
- SATIRE: Obvious humor, sarcasm, or parody not meant to be taken literally
- UNVERIFIABLE: Makes claims that cannot be verified with available information

Post Title: {title}
Post Body: {body}

Respond in this exact JSON format:
{{
    "category": "CATEGORY_NAME",
    "confidence": 0.XX,
    "reasoning": "Brief explanation (1-2 sentences)"
}}

JSON Response:"""

    def classify_post(self, title: str, body: str = "") -> Dict:
        """
        Classify a single post

        Args:
            title: Post title
            body: Post body (optional)

        Returns:
            Dict with category, confidence, and reasoning
        """
        if self.use_mock:
            return self._mock_classify(title, body)

        prompt = self.prompt_template.format(
            title=title[:500],  # Limit length
            body=body[:1000] if body else "[No body text]"
        )

        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()

            # Parse JSON response
            # Handle potential markdown formatting
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text)

            # Validate category
            if result.get("category") not in MISINFO_CATEGORIES:
                result["category"] = "UNVERIFIABLE"

            return {
                "category": result.get("category", "UNVERIFIABLE"),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": result.get("reasoning", "No reasoning provided"),
                "raw_response": response.text
            }

        except Exception as e:
            return {
                "category": "ERROR",
                "confidence": 0.0,
                "reasoning": str(e),
                "raw_response": str(e)
            }

    def _mock_classify(self, title: str, body: str) -> Dict:
        """Mock classification for testing"""
        import random

        # Simple heuristic-based mock
        title_lower = title.lower()

        if any(word in title_lower for word in ["test", "launch", "missile", "nuclear"]):
            category = random.choice(["FACTUAL", "MISLEADING"])
        elif any(word in title_lower for word in ["think", "opinion", "believe", "should"]):
            category = "OPINION"
        elif any(word in title_lower for word in ["lol", "joke", "humor"]):
            category = "SATIRE"
        else:
            category = random.choice(MISINFO_CATEGORIES)

        return {
            "category": category,
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "reasoning": f"Mock classification: detected keywords suggesting {category}",
            "raw_response": "[MOCK RESPONSE]"
        }

    def classify_batch(
        self,
        posts: List[Dict],
        delay: float = 0.5,
        max_posts: Optional[int] = None
    ) -> List[Dict]:
        """
        Classify a batch of posts

        Args:
            posts: List of dicts with 'title' and optionally 'body'
            delay: Delay between API calls (rate limiting)
            max_posts: Maximum posts to process

        Returns:
            List of classification results
        """
        posts_to_process = posts[:max_posts] if max_posts else posts
        results = []

        for post in tqdm(posts_to_process, desc="Classifying posts"):
            result = self.classify_post(
                title=post.get("title", ""),
                body=post.get("body", "") or post.get("selftext", "")
            )
            result["post_id"] = post.get("id", "")
            results.append(result)

            if not self.use_mock:
                time.sleep(delay)

        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        title_column: str = "title",
        body_column: str = "selftext",
        delay: float = 0.5,
        max_posts: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Classify all posts in a DataFrame

        Args:
            df: DataFrame with posts
            title_column: Title column name
            body_column: Body column name
            delay: Delay between API calls
            max_posts: Max posts to classify

        Returns:
            DataFrame with added classification columns
        """
        df = df.copy()

        posts = [
            {
                "id": row.get("id", ""),
                "title": row.get(title_column, ""),
                "body": row.get(body_column, "")
            }
            for _, row in df.iterrows()
        ]

        results = self.classify_batch(posts, delay=delay, max_posts=max_posts)

        # Add columns
        df["misinfo_category"] = [r["category"] for r in results]
        df["misinfo_confidence"] = [r["confidence"] for r in results]
        df["misinfo_reasoning"] = [r["reasoning"] for r in results]

        return df


def analyze_misinfo_distribution(
    df: pd.DataFrame,
    category_column: str = "misinfo_category",
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize misinformation category distribution

    Args:
        df: DataFrame with misinfo categories
        category_column: Category column name
        output_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Overall distribution
    ax = axes[0, 0]
    counts = df[category_column].value_counts()
    colors = {
        "FACTUAL": "green",
        "MISLEADING": "red",
        "OPINION": "blue",
        "SATIRE": "purple",
        "UNVERIFIABLE": "gray",
        "ERROR": "black"
    }
    bar_colors = [colors.get(c, "gray") for c in counts.index]
    ax.bar(counts.index, counts.values, color=bar_colors)
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    ax.set_title("Misinformation Category Distribution")
    ax.tick_params(axis='x', rotation=45)

    # Add percentages
    total = counts.sum()
    for i, (cat, count) in enumerate(counts.items()):
        ax.text(i, count + 10, f'{count/total*100:.1f}%', ha='center')

    # 2. By era if available
    ax = axes[0, 1]
    if 'era' in df.columns:
        era_dist = pd.crosstab(df['era'], df[category_column], normalize='index') * 100
        era_dist.plot(kind='bar', ax=ax, colormap='tab10')
        ax.set_xlabel("Era")
        ax.set_ylabel("Percentage")
        ax.set_title("Category Distribution by Era")
        ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=0)
    else:
        ax.text(0.5, 0.5, "No 'era' column available", ha='center', va='center')
        ax.set_title("Category Distribution by Era")

    # 3. Confidence distribution
    ax = axes[1, 0]
    if 'misinfo_confidence' in df.columns:
        for cat in MISINFO_CATEGORIES:
            cat_data = df[df[category_column] == cat]['misinfo_confidence']
            if len(cat_data) > 0:
                ax.hist(cat_data, bins=20, alpha=0.5, label=cat)
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Confidence Distribution by Category")
        ax.legend()

    # 4. By subreddit if available
    ax = axes[1, 1]
    if 'subreddit' in df.columns:
        sub_dist = pd.crosstab(df['subreddit'], df[category_column], normalize='index') * 100
        # Only show MISLEADING proportion
        if 'MISLEADING' in sub_dist.columns:
            misleading = sub_dist['MISLEADING'].sort_values(ascending=False)
            ax.barh(range(len(misleading)), misleading.values, color='red', alpha=0.7)
            ax.set_yticks(range(len(misleading)))
            ax.set_yticklabels(misleading.index)
            ax.set_xlabel("Misleading Content (%)")
            ax.set_title("Misleading Content by Subreddit")
    else:
        ax.text(0.5, 0.5, "No 'subreddit' column available", ha='center', va='center')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {output_path}")

    return fig


def run_misinfo_analysis(
    csv_path: str,
    output_dir: str = '../outputs/figures',
    use_mock: bool = True,
    max_posts: Optional[int] = None,
    project_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Run full misinformation analysis

    Args:
        csv_path: Path to posts CSV
        output_dir: Output directory
        use_mock: Use mock mode (no API calls)
        max_posts: Max posts to analyze (None = all)
        project_id: GCP project ID

    Returns:
        DataFrame with misinformation labels
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} posts")

    # Initialize detector
    detector = GeminiMisinfoDetector(
        project_id=project_id,
        use_mock=use_mock
    )

    # Analyze
    print(f"\nClassifying {'up to ' + str(max_posts) if max_posts else 'all'} posts...")
    df = detector.analyze_dataframe(
        df,
        title_column='title',
        body_column='selftext' if 'selftext' in df.columns else 'text_combined',
        max_posts=max_posts
    )

    # Summary
    print("\n" + "="*50)
    print("MISINFORMATION ANALYSIS SUMMARY")
    print("="*50)
    print("\nCategory Distribution:")
    print(df['misinfo_category'].value_counts())

    # Visualize
    analyze_misinfo_distribution(
        df,
        category_column='misinfo_category',
        output_path=os.path.join(output_dir, 'misinfo_distribution.png')
    )

    # Save results
    output_csv = os.path.join(output_dir, 'misinfo_labels.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    return df


# Cost estimation
def estimate_cost(n_posts: int, avg_chars: int = 500) -> float:
    """
    Estimate API cost for Gemini classification

    Args:
        n_posts: Number of posts
        avg_chars: Average characters per post

    Returns:
        Estimated cost in USD
    """
    # Gemini 1.5 Flash pricing (as of late 2024)
    # Input: $0.00001875 per 1K characters
    # Output: $0.000075 per 1K characters

    # Prompt template is ~500 chars
    input_chars = n_posts * (avg_chars + 500)
    output_chars = n_posts * 200  # JSON response ~200 chars

    input_cost = (input_chars / 1000) * 0.00001875
    output_cost = (output_chars / 1000) * 0.000075

    return input_cost + output_cost


if __name__ == "__main__":
    print("Misinformation Detection for Reddit US-NK Data")
    print("="*50)

    # Cost estimation
    print("\nCost Estimation:")
    for n in [100, 1000, 10000]:
        cost = estimate_cost(n)
        print(f"  {n:,} posts: ${cost:.4f}")

    # Run analysis
    data_path = "../data/processed/posts_final.csv"

    if os.path.exists(data_path):
        # Start with mock mode for testing
        print("\n" + "="*50)
        print("Running in MOCK mode (no API calls)")
        print("To use real Gemini API, set use_mock=False and provide project_id")
        print("="*50)

        df = run_misinfo_analysis(
            csv_path=data_path,
            output_dir='../outputs/figures',
            use_mock=True,  # Set to False and provide project_id for real API
            max_posts=100   # Test with 100 posts first
        )
    else:
        print(f"Data file not found: {data_path}")
