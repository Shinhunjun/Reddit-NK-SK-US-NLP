"""
BERT-based Sentiment Analysis for Reddit US-NK Analysis
Using DistilBERT - optimized for M4 Pro (Apple Silicon MPS)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns


class BERTSentimentAnalyzer:
    """
    BERT-based sentiment analysis using DistilBERT
    Optimized for Apple Silicon (M4 Pro with MPS)
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        batch_size: int = 32,
        max_length: int = 512,
        use_mps: bool = True,
        verbose: bool = True
    ):
        """
        Initialize BERT sentiment analyzer

        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for inference
            max_length: Max tokens per document
            use_mps: Use Apple Silicon GPU (MPS)
            verbose: Show progress
        """
        self.verbose = verbose
        self.batch_size = batch_size
        self.max_length = max_length

        # Determine device
        if use_mps and torch.backends.mps.is_available():
            self.device = "mps"
            if self.verbose:
                print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            if self.verbose:
                print("Using CUDA GPU")
        else:
            self.device = "cpu"
            if self.verbose:
                print("Using CPU")

        # Load model
        if self.verbose:
            print(f"Loading model: {model_name}")

        # Use pipeline for easy inference
        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=self.device if self.device != "mps" else -1,  # MPS not fully supported in pipeline
            truncation=True,
            max_length=max_length
        )

        # For MPS, load model separately
        if self.device == "mps":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            self._use_manual = True
        else:
            self._use_manual = False

        if self.verbose:
            print("Model loaded successfully")

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment of a single text

        Returns:
            Dict with 'label' (POSITIVE/NEGATIVE), 'score', and 'compound'
        """
        if self._use_manual:
            return self._analyze_mps(text)
        else:
            result = self.classifier(text[:self.max_length * 4])[0]  # Approximate char limit
            return {
                'label': result['label'],
                'score': result['score'],
                'compound': result['score'] if result['label'] == 'POSITIVE' else -result['score']
            }

    def _analyze_mps(self, text: str) -> Dict:
        """Manual inference for MPS device"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred_class].item()

        label = "POSITIVE" if pred_class == 1 else "NEGATIVE"
        compound = confidence if label == "POSITIVE" else -confidence

        return {
            'label': label,
            'score': confidence,
            'compound': compound
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze a batch of texts"""
        if self._use_manual:
            return [self._analyze_mps(t) for t in texts]
        else:
            results = self.classifier(texts)
            return [
                {
                    'label': r['label'],
                    'score': r['score'],
                    'compound': r['score'] if r['label'] == 'POSITIVE' else -r['score']
                }
                for r in results
            ]

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text_combined',
        prefix: str = 'bert'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame

        Args:
            df: DataFrame with text column
            text_column: Name of text column
            prefix: Prefix for new columns

        Returns:
            DataFrame with added sentiment columns
        """
        df = df.copy()

        # Clean text - handle NaN
        texts = df[text_column].fillna('').astype(str).tolist()

        # Filter empty texts
        valid_mask = [len(t.strip()) > 0 for t in texts]

        if self.verbose:
            print(f"\nAnalyzing {sum(valid_mask):,} texts with BERT...")

        results = []
        valid_texts = [t for t, v in zip(texts, valid_mask) if v]

        # Process in batches with progress bar
        for i in tqdm(range(0, len(valid_texts), self.batch_size), desc="BERT Sentiment"):
            batch = valid_texts[i:i + self.batch_size]
            batch_results = self.analyze_batch(batch)
            results.extend(batch_results)

        # Map results back to full dataframe
        full_results = []
        result_idx = 0
        for is_valid in valid_mask:
            if is_valid:
                full_results.append(results[result_idx])
                result_idx += 1
            else:
                full_results.append({'label': 'NEUTRAL', 'score': 0.0, 'compound': 0.0})

        # Add columns
        df[f'{prefix}_label'] = [r['label'] for r in full_results]
        df[f'{prefix}_score'] = [r['score'] for r in full_results]
        df[f'{prefix}_compound'] = [r['compound'] for r in full_results]

        # Add sentiment category
        df[f'{prefix}_sentiment'] = df[f'{prefix}_compound'].apply(
            lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
        )

        if self.verbose:
            print("\nSentiment Distribution (BERT):")
            print(df[f'{prefix}_sentiment'].value_counts())

        return df


def compare_sentiment_methods(
    df: pd.DataFrame,
    vader_column: str = 'sentiment_compound',
    bert_column: str = 'bert_compound',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare VADER vs BERT sentiment scores

    Args:
        df: DataFrame with both sentiment columns
        vader_column: VADER compound score column
        bert_column: BERT compound score column
        output_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot comparison
    ax1 = axes[0, 0]
    ax1.scatter(df[vader_column], df[bert_column], alpha=0.3, s=10)
    ax1.set_xlabel('VADER Compound')
    ax1.set_ylabel('BERT Compound')
    ax1.set_title('VADER vs BERT Sentiment Scores')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)

    # Add correlation
    corr = df[vader_column].corr(df[bert_column])
    ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=ax1.transAxes, fontsize=12)

    # 2. Distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(df[vader_column], bins=50, alpha=0.5, label='VADER', density=True)
    ax2.hist(df[bert_column], bins=50, alpha=0.5, label='BERT', density=True)
    ax2.set_xlabel('Compound Score')
    ax2.set_ylabel('Density')
    ax2.set_title('Score Distribution Comparison')
    ax2.legend()

    # 3. Sentiment category comparison
    ax3 = axes[1, 0]

    # VADER categories
    vader_cats = df[vader_column].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    ).value_counts()

    # BERT categories
    bert_cats = df[bert_column].apply(
        lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
    ).value_counts()

    x = np.arange(3)
    width = 0.35
    categories = ['positive', 'neutral', 'negative']

    vader_vals = [vader_cats.get(c, 0) for c in categories]
    bert_vals = [bert_cats.get(c, 0) for c in categories]

    ax3.bar(x - width/2, vader_vals, width, label='VADER')
    ax3.bar(x + width/2, bert_vals, width, label='BERT')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.set_ylabel('Count')
    ax3.set_title('Sentiment Category Distribution')
    ax3.legend()

    # 4. Agreement analysis
    ax4 = axes[1, 1]

    vader_sent = df[vader_column].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )
    bert_sent = df[bert_column].apply(
        lambda x: 'positive' if x > 0.2 else ('negative' if x < -0.2 else 'neutral')
    )

    agreement = (vader_sent == bert_sent).mean()

    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(vader_sent, bert_sent, labels=categories)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories, ax=ax4)
    ax4.set_xlabel('BERT')
    ax4.set_ylabel('VADER')
    ax4.set_title(f'Agreement Matrix (Agreement: {agreement:.1%})')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")

    return fig


def visualize_bert_sentiment_trend(
    df: pd.DataFrame,
    date_column: str = 'date',
    sentiment_column: str = 'bert_compound',
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize BERT sentiment trend over time

    Args:
        df: DataFrame with date and sentiment columns
        date_column: Date column name
        sentiment_column: BERT sentiment column name
        output_path: Optional save path
    """
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Monthly average
    monthly = df.set_index(date_column).resample('M')[sentiment_column].agg(['mean', 'std', 'count'])
    monthly = monthly.reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot mean with confidence interval
    ax.plot(monthly[date_column], monthly['mean'], 'b-', linewidth=2, label='BERT Sentiment')
    ax.fill_between(
        monthly[date_column],
        monthly['mean'] - monthly['std'],
        monthly['mean'] + monthly['std'],
        alpha=0.2
    )

    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Neutral')
    ax.set_xlabel('Date')
    ax.set_ylabel('BERT Compound Score')
    ax.set_title('BERT Sentiment Trend (Monthly Average)')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trend to {output_path}")

    return fig


def analyze_with_bert_sentiment(
    csv_path: str,
    text_column: str = 'text_combined',
    output_dir: str = '../outputs/figures'
) -> pd.DataFrame:
    """
    Run full BERT sentiment analysis on Reddit data

    Args:
        csv_path: Path to processed CSV file
        text_column: Column containing text
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} posts")

    # Initialize analyzer
    analyzer = BERTSentimentAnalyzer(
        model_name="distilbert-base-uncased-finetuned-sst-2-english",
        batch_size=32
    )

    # Analyze
    df = analyzer.analyze_dataframe(df, text_column=text_column)

    # Compare with VADER if available
    if 'sentiment_compound' in df.columns:
        compare_sentiment_methods(
            df,
            vader_column='sentiment_compound',
            bert_column='bert_compound',
            output_path=os.path.join(output_dir, 'sentiment_bert_vs_vader.png')
        )

    # Visualize trend
    if 'date' in df.columns:
        visualize_bert_sentiment_trend(
            df,
            date_column='date',
            sentiment_column='bert_compound',
            output_path=os.path.join(output_dir, 'sentiment_bert_trend.png')
        )

    # Save results
    output_csv = csv_path.replace('.csv', '_bert_sentiment.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    return df


# Quick test
if __name__ == "__main__":
    print("BERT Sentiment Analysis for Reddit US-NK Data")
    print("="*50)

    # Check if data exists
    data_path = "../data/processed/posts_final.csv"

    if os.path.exists(data_path):
        df = analyze_with_bert_sentiment(
            csv_path=data_path,
            text_column='text_combined',
            output_dir='../outputs/figures'
        )

        print("\n" + "="*50)
        print("BERT SENTIMENT ANALYSIS COMPLETE")
        print("="*50)

        # Summary stats
        print("\nSummary Statistics:")
        print(f"  Mean compound: {df['bert_compound'].mean():.3f}")
        print(f"  Std compound: {df['bert_compound'].std():.3f}")
        print(f"\nSentiment Distribution:")
        print(df['bert_sentiment'].value_counts())
    else:
        print(f"Data file not found: {data_path}")
