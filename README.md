# Reddit Discourse on North Korea, US-ROK Alliance, and Korean Peninsula Security

**A Temporal and Sentiment Analysis (2022-2025)**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This project analyzes U.S. public perception of North Korea, the U.S.-ROK alliance, and Korean Peninsula security issues through Reddit discourse. Using natural language processing techniques, we examine:

- **Temporal patterns**: How discussion volume changes around key security events
- **Sentiment trends**: Public sentiment toward North Korea and the alliance over time
- **Topic evolution**: Main themes in the discourse and how they shift

## Key Features

- **Multi-source data collection**: Arctic Shift API (historical) + PRAW (recent)
- **Sentiment analysis**: VADER + TextBlob dual-method approach
- **Topic modeling**: LDA for interpretable topic discovery
- **Event-based analysis**: Correlation with missile tests, summits, and exercises
- **Interactive visualizations**: Time series, sentiment trends, topic evolution

## Project Structure

```
reddit_US_NK/
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/              # Raw collected data (JSON)
│   └── processed/        # Cleaned and analyzed data (CSV)
├── notebooks/
│   └── analysis.ipynb    # Main analysis notebook
├── outputs/
│   ├── figures/          # Generated visualizations
│   └── reports/          # Analysis reports
└── src/
    ├── __init__.py
    ├── config.py         # Configuration and constants
    ├── data_collector.py # Arctic Shift & PRAW collectors
    ├── preprocessor.py   # Text cleaning and preprocessing
    ├── sentiment_analyzer.py  # VADER & TextBlob analysis
    ├── topic_modeler.py  # LDA topic modeling
    ├── visualizer.py     # Visualization functions
    └── pipeline.py       # Main analysis pipeline
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reddit_US_NK.git
cd reddit_US_NK

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for sentiment analysis)
python -c "import nltk; nltk.download('vader_lexicon')"
```

## Quick Start

### Option 1: Jupyter Notebook (Recommended)

```bash
jupyter notebook notebooks/analysis.ipynb
```

### Option 2: Command Line

```bash
# Run full pipeline
cd src
python pipeline.py --step all --n-topics 6 --limit 100

# Or run individual steps
python pipeline.py --step collect
python pipeline.py --step preprocess
python pipeline.py --step sentiment
python pipeline.py --step topics
python pipeline.py --step visualize
```

### Option 3: Python Script

```python
from src.data_collector import ArcticShiftCollector
from src.preprocessor import preprocess_posts
from src.sentiment_analyzer import SentimentAnalyzer
from src.topic_modeler import LDATopicModeler

# Collect data
collector = ArcticShiftCollector()
posts = collector.collect_all(
    queries=["north korea", "south korea"],
    subreddits=["worldnews", "geopolitics"],
    after="2023-01-01",
    before="2023-12-31"
)

# Preprocess
df = preprocess_posts(posts)

# Analyze sentiment
analyzer = SentimentAnalyzer()
df = analyzer.analyze_dataframe(df)

# Topic modeling
modeler = LDATopicModeler(n_topics=5)
modeler.fit(df['text_combined'].tolist())
modeler.print_topics()
```

## Data Sources

| Source | Time Range | API Key Required |
|--------|------------|------------------|
| [Arctic Shift](https://arctic-shift.photon-reddit.com/) | 2005 - 2023 | No |
| [PRAW (Reddit API)](https://www.reddit.com/prefs/apps) | 2024 - Present | Yes |

### Setting up Reddit API (for PRAW)

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Create a new app (script type)
3. Copy credentials to `.env`:

```bash
cp .env.example .env
# Edit .env with your credentials
```

## Key Events Analyzed

| Date | Event |
|------|-------|
| 2022-03-24 | North Korea ICBM test (Hwasong-17) |
| 2022-10-04 | North Korea missile over Japan |
| 2023-04-13 | North Korea Hwasong-18 solid-fuel ICBM |
| 2023-08-18 | Camp David Summit (US-ROK-Japan) |
| 2023-11-21 | North Korea satellite launch |

## Sample Outputs

### Sentiment Trend
![Sentiment Trend](outputs/figures/sentiment_trend_example.png)

### Topic Distribution
![Topic Distribution](outputs/figures/topic_dist_example.png)

## Research Applications

This analysis framework can be adapted for:

- **Longitudinal conflict discourse analysis** (e.g., India-Pakistan)
- **Public opinion tracking** around security events
- **Cross-platform narrative comparison**
- **Event-driven sentiment analysis**

## Technical Details

### Sentiment Analysis
- **VADER**: Rule-based, optimized for social media
- **TextBlob**: Pattern-based polarity detection
- **Thresholds**: Positive (≥0.05), Negative (≤-0.05), Neutral (between)

### Topic Modeling
- **LDA** (Latent Dirichlet Allocation): Interpretable, lightweight
- **BERTopic** (optional): Transformer-based, higher quality
- **Typical topics discovered**:
  - Military threats / missile launches
  - US-ROK alliance dynamics
  - Nuclear weapons discourse
  - Diplomatic negotiations
  - Regional security (China, Japan)

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{reddit_us_nk_analysis,
  author = {Jun Sin},
  title = {Reddit Discourse on North Korea and US-ROK Alliance},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/reddit_US_NK}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) for historical Reddit data access
- [PRAW](https://praw.readthedocs.io/) for Reddit API wrapper
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment) for social media sentiment analysis

---

*This project was created as part of research preparation for longitudinal conflict discourse analysis.*
