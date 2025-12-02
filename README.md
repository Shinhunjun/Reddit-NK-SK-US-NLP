# Reddit US-NK Analysis
## 15-Year Longitudinal Discourse Analysis (2009-2023)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project analyzes U.S. public perception of North Korea, the U.S.-ROK alliance, and Korean Peninsula security issues through Reddit discourse across four presidential administrations.

---

## Executive Summary

### Key Findings
- **Obama 2nd term (2013-2016)** showed the most negative sentiment (-0.395)
- **r/AskAnAmerican** has 90.5% negative posts - reflecting direct public opinion
- **Yeonpyeong Shelling (2010)** caused the largest sentiment drop (-0.688)
- Statistically significant differences between presidential eras (p < 0.001)

### Dataset Scale
| Metric | Count |
|--------|-------|
| Posts | 10,442 |
| Comments | 88,891 |
| Time Period | 2009-2023 (15 years) |
| Subreddits | 7 |
| Presidential Eras | 4 |

---

## Dataset Overview

### Distribution by Presidential Era
| Era | Posts | Percentage |
|-----|-------|------------|
| Obama 1st (2009-2012) | 2,087 | 20.0% |
| Obama 2nd (2013-2016) | 3,105 | 29.7% |
| Trump (2017-2020) | 2,087 | 20.0% |
| Biden (2021-2023) | 3,163 | 30.3% |

### Distribution by Subreddit
| Subreddit | Description | Posts |
|-----------|-------------|-------|
| r/korea | General Korea topics | 2,322 |
| r/northkorea | NK-focused discussion | 2,215 |
| r/worldnews | International news | 2,121 |
| r/politics | US politics | 1,407 |
| r/news | General news | 1,303 |
| r/geopolitics | International relations analysis | 875 |
| r/AskAnAmerican | Q&A with Americans | 199 |

### Data Access

Raw and processed data files are available on Google Drive:

**[Download Dataset](https://drive.google.com/drive/folders/1LDcj3RzOh04PVBVzRO7cJAIHS5Gcafjb?usp=sharing)**

| File | Size | Description |
|------|------|-------------|
| `reddit_comments_linked.json` | 154MB | Raw comment data (88,891 comments) |
| `comments_bert_sentiment.csv` | 67MB | Comment sentiment analysis results |
| `reddit_posts_combined.json` | 33MB | Raw post data (10,442 posts) |
| `comments_bertopic.csv` | 29MB | Comment topic modeling results |

---

## Analysis Results

### 1. BERT Sentiment Analysis

#### Sentiment by Presidential Era
![Sentiment by Era](outputs/figures/bert_sentiment_by_era.png)

| Era | Mean Score | Negative % | Posts |
|-----|------------|------------|-------|
| Obama 1st (2009-2012) | -0.276 | 64.0% | 2,087 |
| **Obama 2nd (2013-2016)** | **-0.395** | **70.5%** | 3,105 |
| Trump (2017-2020) | -0.352 | 68.3% | 2,087 |
| Biden (2021-2023) | -0.253 | 63.2% | 3,163 |

**Statistical Significance**: Kruskal-Wallis H = 75.81, p < 0.001

#### Sentiment by Subreddit
![Sentiment by Subreddit](outputs/figures/bert_sentiment_by_subreddit.png)

| Subreddit | Mean Score | Negative % | Notes |
|-----------|------------|------------|-------|
| **r/AskAnAmerican** | **-0.794** | **90.5%** | Direct public opinion |
| r/politics | -0.372 | 69.3% | US political discourse |
| r/geopolitics | -0.356 | 68.2% | Expert analysis |
| r/worldnews | -0.342 | 67.8% | News-focused |
| r/northkorea | -0.314 | 66.3% | NK specialists |
| r/korea | -0.276 | 64.2% | Korea general |
| r/news | -0.216 | 61.2% | Neutral reporting |

#### Sentiment by Topic
![Sentiment by Topic](outputs/figures/bert_sentiment_by_topic.png)

| Topic | Mean Score | Negative % | Posts |
|-------|------------|------------|-------|
| **General NK News** | **-0.405** | **70.8%** | 2,883 |
| Nuclear/Missiles | -0.364 | 68.9% | 3,696 |
| Culture/Society | -0.337 | 66.6% | 587 |
| US Politics | -0.245 | 62.7% | 1,111 |
| Diplomacy/Talks | -0.185 | 60.0% | 230 |
| Historical | -0.176 | 60.4% | 245 |
| Security/Military | -0.160 | 58.5% | 1,690 |

#### Monthly Sentiment Trend
![Sentiment Trend](outputs/figures/sentiment_bert_trend.png)

---

### 2. Comment Analysis (83,300 comments)

#### Comment Sentiment by Era
| Era | Mean Score | Negative % | Comments |
|-----|------------|------------|----------|
| Trump (2017-2020) | -0.474 | 73.9% | 22,315 |
| Obama 2nd (2013-2016) | -0.460 | 73.3% | 19,937 |
| Biden (2021-2023) | -0.452 | 72.9% | 32,126 |
| Obama 1st (2009-2012) | -0.432 | 71.8% | 8,922 |

**Key Finding**: Comments are more negative than posts (mean -0.457 vs -0.32)

#### Comment BERTopic Analysis
**274 topics discovered** from 83,300 comments using BERTopic (min_cluster_size=30)

##### Top Topics by Sentiment
| Topic | Sentiment | Neg% | Keywords | Interpretation |
|-------|-----------|------|----------|----------------|
| 3 | **-0.767** | 89.3% | tax, money, currency | Economic Criticism |
| 12 | **-0.765** | 89.3% | food, starving, famine | Humanitarian Crisis |
| 9 | **-0.741** | 87.3% | missile, icbm, defense | Missile Threats |
| 2 | -0.736 | 86.7% | nukes, nuclear, weapons | Nuclear Weapons |
| 1 | -0.586 | 80.7% | nukes, nuclear, weapons | Nuclear Discussion |
| 0 | -0.569 | 78.7% | china, chinese, ccp | China Relations |
| 16 | -0.555 | 78.0% | moon, moon jae, president | SK Politics |
| 4 | -0.220 | 61.3% | city, buildings, tours | Tourism |
| 18 | **+0.706** | 14.7% | thanks, sharing | Gratitude (only positive) |

#### Interactive Comment Visualizations
- [Comment Topic Hierarchy](outputs/figures/comments_bertopic_hierarchy.html)
- [Comment Topic Barchart](outputs/figures/comments_bertopic_barchart.html)

---

### 3. BERTopic - Posts (Topic Modeling)

**59 topics discovered** using BERTopic with all-MiniLM-L6-v2 embeddings.

#### Interactive Visualizations
- [Topic Clusters (2D)](outputs/figures/bertopic_clusters.html)
- [Topic Hierarchy](outputs/figures/bertopic_hierarchy.html)
- [Topic Barchart](outputs/figures/bertopic_barchart.html)
- [Topics Over Time](outputs/figures/bertopic_over_time.html)
- [Topic Heatmap](outputs/figures/bertopic_heatmap.html)

---

### 3. Event Analysis (17 Key Events)

![Events Comparison](outputs/figures/events_comparison.png)

#### Top Events by Sentiment Impact
| Event | Date | Era | Posts | Sentiment Change |
|-------|------|-----|-------|------------------|
| **Yeonpyeong Shelling** | 2010-11-23 | Obama | 43 | **-0.688** |
| NK 3rd Nuclear Test | 2013-02-12 | Obama | 7 | -0.556 |
| Hanoi Summit Failure | 2019-02-28 | Trump | 2 | -0.518 |
| Fire and Fury Speech | 2017-08-08 | Trump | 7 | -0.378 |
| Hwasong-17 ICBM Test | 2022-03-24 | Biden | 19 | +0.348 |
| NK 6th Nuclear Test | 2017-09-03 | Trump | 29 | +0.196 |

#### Hourly Analysis Examples
| Yeonpyeong Shelling | Kim Jong-il Death |
|---------------------|-------------------|
| ![Yeonpyeong](outputs/figures/event_hourly_Yeonpyeong_Shelling.png) | ![Kim Death](outputs/figures/event_hourly_Kim_Jong-il_Death.png) |

---

### 4. Network Analysis

![Network Metrics](outputs/figures/network_metrics_distribution.png)

#### Comment Tree Metrics
| Metric | Value |
|--------|-------|
| Posts Analyzed | 6,260 |
| Avg Comment Depth | 3.2 levels |
| Avg Branching Factor | 1.8 |
| Avg Comments/Post | 13.7 |

#### Top 10 Hub Users
![Hub Users](outputs/figures/hub_users_analysis.png)

| Rank | User | Comments | Avg Score | Reply Ratio |
|------|------|----------|-----------|-------------|
| 1 | AutoModerator | 724 | 1.0 | 0.7% |
| 2 | autotldr | 284 | 2.4 | 0% |
| 3 | christ0ph | 255 | 1.1 | 71.4% |
| 4 | DaBIGmeow888 | 219 | 3.7 | 92.2% |
| 5 | chernobog95 | 201 | 0.3 | 89.1% |
| 6 | glitterlok | 196 | 3.3 | 74.5% |
| 7 | daehanmindecline | 181 | 4.2 | 49.2% |
| 8 | kulcoria | 176 | 2.6 | 74.4% |
| 9 | imnotyourman | 154 | 4.7 | 74.0% |
| 10 | jaywalker1982 | 141 | 1.6 | 80.9% |

---

### 5. GraphRAG - Knowledge Graph Analysis

[Microsoft GraphRAG](https://github.com/microsoft/graphrag) was used to automatically extract a knowledge graph from Reddit discourse, enabling structured querying of entities, relationships, and thematic communities.

#### How GraphRAG Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GraphRAG Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│  1. Text Chunking                                                   │
│     └── Split documents into analyzable text units                  │
│                                                                     │
│  2. Entity & Relationship Extraction (LLM)                          │
│     └── GPT extracts: PERSON, GEO, ORGANIZATION, EVENT              │
│     └── GPT identifies relationships between entities               │
│                                                                     │
│  3. Graph Construction                                              │
│     └── Nodes = Entities (with descriptions)                        │
│     └── Edges = Relationships (with weights)                        │
│                                                                     │
│  4. Community Detection (Leiden Algorithm)                          │
│     └── Cluster nodes by modularity (connection density)            │
│     └── Hierarchical community structure                            │
│                                                                     │
│  5. Community Summarization (LLM)                                   │
│     └── Generate reports for each community                         │
│                                                                     │
│  6. Query Processing                                                │
│     └── Global: Aggregate community reports                         │
│     └── Local: Entity-specific context retrieval                    │
└─────────────────────────────────────────────────────────────────────┘
```

#### Extracted Knowledge Graph (100 posts sample)

| Component | Count | Description |
|-----------|-------|-------------|
| **Entities** | 58 | People, places, organizations, events |
| **Relationships** | 53 | Connections between entities |
| **Communities** | 4 | Thematic clusters |

#### Top Entities by Connectivity (Degree)

| Entity | Type | Connections | Role |
|--------|------|-------------|------|
| **NORTH KOREA** | GEO | 29 | Central node |
| KIM JONG UN | PERSON | 7 | Supreme Leader |
| SOUTH KOREA | GEO | 4 | Key counterpart |
| PUTIN | PERSON | 3 | Russia relations |
| TRUMP | PERSON | 2 | US diplomacy |

#### Key Relationships Extracted

```
TRUMP ──[direct summit diplomacy]──> KIM JONG-UN
KIM JONG-UN ──[supreme leader of]──> NORTH KOREA
KIM JONG-UN ──[strategic meetings]──> PUTIN
NORTH KOREA ──[divided nation]──> SOUTH KOREA
NORTH KOREA ──[opposes deployment]──> THAAD
SOUTH KOREA ──[hosts]──> US TROOPS
```

#### Community Structure

| Community | Title | Rank | Key Entities |
|-----------|-------|------|--------------|
| 0 | North Korea and Global Leaders | 8.5 | Kim Jong Un, Putin, Park Geun-hye |
| 1 | NK-US Diplomatic Relations | 8.5 | Trump, Russia, Nuclear talks |
| **2** | **Korean Peninsula Dynamics** | **9.0** | NK-SK relations, China, US influence |
| 3 | Travis King Incident | 7.5 | US soldier, Military tensions |

#### GraphRAG vs Vector RAG Comparison

We compared three query methods using the same question: **"What is the relationship between North Korea and the United States?"**

| Method | Type | Data Source | Characteristics |
|--------|------|-------------|-----------------|
| **Global Search** | GraphRAG | Community Reports | Structured, thematic overview |
| **Local Search** | GraphRAG | Entity + Relationships | Entity-centered, relationship-focused |
| **Basic Search** | Vector RAG | Text Chunks (Embedding Similarity) | Raw text details, source-heavy |

---

##### Global Search Response (GraphRAG)

> The relationship between North Korea and the United States is characterized by **deep-seated tensions, diplomatic challenges, and ongoing security concerns**. This analysis draws on various community reports to provide a comprehensive understanding of the multifaceted dynamics at play.
>
> **Military Tensions and Security Concerns**
> The presence of U.S. troops in South Korea remains a critical issue in the North Korea-U.S. relationship. North Korea views this military presence as a significant threat, which exacerbates tensions between the two nations. This sentiment is reflected in the broader geopolitical dynamics of the Korean Peninsula, where North Korea's actions, including missile tests and military provocations, are often seen as responses to perceived U.S. aggression [Data: Reports (1, 2)].
>
> **Sanctions and Human Rights**
> The United States has implemented various sanctions against North Korea, primarily targeting its nuclear and missile programs. These sanctions also address broader concerns about human rights abuses within the country. The U.S. stance on these issues reflects a commitment to curbing North Korea's nuclear ambitions while also highlighting the humanitarian concerns associated with the regime [Data: Reports (1, 3)].
>
> **Incidents Affecting Relations**
> Recent incidents, such as the case of Travis King, have further complicated the relationship. King's defection to North Korea and subsequent return to the U.S. underscored the unpredictable nature of individual actions in the context of international relations. This incident highlighted the ongoing tensions and the potential for sudden developments that can impact diplomatic efforts [Data: Reports (3)].
>
> **Diplomatic Engagement and Challenges**
> The potential for diplomatic engagement between North Korea and the U States has been a recurring theme, though it remains fraught with challenges. Historical summits, such as those between former President Trump and Kim Jong-un, have attempted to address nuclear disarmament but have often ended without significant progress. The failure to reach substantial agreements reflects the deep mistrust and differing priorities between the two nations [Data: Reports (1, 2)].
>
> **Conclusion**
> In summary, the relationship between North Korea and the United States is marked by military tensions, economic sanctions, and sporadic diplomatic engagements. The interplay of these factors continues to shape the dynamics of this complex relationship, with both nations navigating a landscape of mutual distrust and strategic calculations.

---

##### Local Search Response (GraphRAG)

> The relationship between North Korea and the United States is characterized by a complex interplay of diplomatic efforts, military tensions, and geopolitical maneuvering. Here is a detailed analysis based on the available data:
>
> **Historical Context and Diplomatic Efforts**
> The relationship has been marked by periods of tension and attempts at diplomacy. Notably, former U.S. President Donald Trump engaged in unprecedented diplomatic efforts with North Korean leader Kim Jong-un, including historic summits aimed at denuclearization. These summits, such as the one in Singapore in June 2018, marked a significant shift in U.S.-North Korea relations, although they did not result in a comprehensive agreement on denuclearization [Data: Entities (2, 39); Relationships (6, 20, 23)].
>
> **Military Tensions**
> Military tensions remain a significant aspect of the U.S.-North Korea relationship. The presence of U.S. troops in South Korea is a point of contention for North Korea, which views this military presence as a threat. Additionally, North Korea's development of nuclear weapons and ballistic missiles, including ICBMs, has been a major source of concern for the United States and its allies [Data: Entities (16, 17); Relationships (2)].
>
> **Geopolitical Dynamics**
> The geopolitical dynamics involving North Korea and the United States are influenced by other regional powers, particularly China and Russia. China has historically been a key ally of North Korea, providing economic and political support. Meanwhile, Russia's involvement in the region, including meetings between Kim Jong-un and Russian President Vladimir Putin, adds another layer of complexity to the U.S.-North Korea relationship [Data: Entities (31); Relationships (13, 14, 15)].
>
> **Recent Developments**
> Recent developments include warnings from the Biden administration regarding North Korea's nuclear and missile tests. The U.S. has continued to impose sanctions on North Korea in response to its weapons programs, while also calling for diplomatic solutions to the ongoing tensions [Data: Relationships (21)].
>
> **Conclusion**
> In summary, the U.S.-North Korea relationship is defined by a history of diplomatic engagements, ongoing military tensions, and the influence of regional powers. While there have been attempts at dialogue and diplomacy, significant challenges remain in achieving lasting peace and denuclearization on the Korean Peninsula.

---

##### Basic Search Response (Vector RAG)

> The relationship between North Korea and the United States is characterized by **deep-seated hostility, intermittent diplomatic efforts, and ongoing security tensions** that have persisted since the Korean War.
>
> **Historical Foundation of Tensions**
> The fundamental tension stems from the unresolved Korean War (1950-1953). As one source notes regarding Americans' collective memory of the Korean War, there's significant debate about its place in national consciousness. Some argue it was "a righteous war" - a view that shaped subsequent U.S. policy toward North Korea for decades [Source: 1].
>
> **Nuclear and Missile Programs**
> North Korea's development of nuclear weapons and ballistic missiles represents the most critical flashpoint in the relationship. Reddit discussions reveal ongoing concern about the regime's capabilities, with users noting that North Korea has conducted multiple nuclear tests and developed ICBMs capable of reaching the continental United States [Sources: 2, 3, 4].
>
> **Diplomatic Attempts**
> The Trump administration's engagement with Kim Jong-un represented a dramatic shift in approach, with historic summits in Singapore (2018) and Hanoi (2019). However, these efforts ultimately failed to produce denuclearization agreements, leaving the fundamental issues unresolved [Source: 5].
>
> **Sanctions Regime**
> The U.S. maintains comprehensive sanctions against North Korea targeting its weapons programs. Reddit discussions highlight how these sanctions have failed to halt nuclear development while contributing to humanitarian concerns within the country [Sources: 6, 7].
>
> **Cyber Activities**
> A notable dimension of the relationship involves North Korea's cyber operations. Sources indicate that North Korea has engaged in extensive cryptocurrency theft and hacking activities, allegedly stealing billions to fund its weapons programs - creating a new vector of conflict with the United States [Source: 8].
>
> **Current State**
> Under the Biden administration, relations remain frozen with no active diplomatic track. North Korea continues missile testing while the U.S. maintains its sanctions and military presence in South Korea, perpetuating the decades-long standoff.

---

##### Key Differences Summary

| Aspect | Global Search (GraphRAG) | Local Search (GraphRAG) | Basic Search (Vector RAG) |
|--------|-------------------------|------------------------|--------------------------|
| **Structure** | Thematic sections with community report citations | Entity-relationship focused with specific data citations | Source-based narrative |
| **Strength** | Big-picture overview, policy implications | Specific actors and their connections | Detailed factual content from original text |
| **Citations** | Reports (1, 2, 3) - Community summaries | Entities (2, 39), Relationships (6, 20) - Graph nodes | Sources (1-8) - Raw text chunks |
| **Best For** | Understanding themes and patterns | Tracking specific people/organizations | Finding specific quotes and details |

**Insight**: GraphRAG excels at providing structured, relationship-aware answers while Vector RAG provides richer textual details. For complex geopolitical analysis, combining both approaches yields the most comprehensive understanding.

---

### 6. STM (Structural Topic Modeling)

**7 topics discovered** using LDA-based Structural Topic Modeling with document covariates.

#### STM Topics
| Topic | Top Keywords | Interpretation |
|-------|--------------|----------------|
| **Topic 0** | kim, korean, jong, kim jong, war, korean war, peninsula | Korean War & Kim Dynasty |
| **Topic 1** | pyongyang, korea, dprk, like, life, know | Daily Life in NK |
| **Topic 2** | china, dprk, chinese, news, park, propaganda, border | China-NK Relations |
| **Topic 3** | time, year, years, japanese, history, moon, food | Historical & Humanitarian |
| **Topic 4** | korean, south korean, nk, people, government, american | South Korea Politics |
| **Topic 5** | korea, nuclear, missile, missiles, test, launch, weapons | Nuclear/Missile Program |
| **Topic 6** | korea, south, north, nuclear, japan, trump, united | Diplomacy & US Policy |

#### Temporal Topic Evolution
![STM Temporal](outputs/figures/stm_temporal_topics.png)

#### Topic Correlation
![Topic Correlation](outputs/figures/stm_topic_correlation.png)

---

## Methodology

### Models Used
| Task | Model | Parameters |
|------|-------|------------|
| Sentiment | DistilBERT | 66M (MPS optimized) |
| Topic Embedding | all-MiniLM-L6-v2 | 22M |
| Topic Modeling | BERTopic | UMAP + HDBSCAN |
| STM | LDA | 7 topics |
| Knowledge Graph | GraphRAG + GPT-4o-mini | Entity extraction + Leiden clustering |

### Hardware
- Apple M4 Pro (MPS acceleration)
- Processing time: ~53 seconds for 10,442 posts

---

## Project Structure

```
reddit_US_NK/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                    # Raw JSON data
│   │   ├── reddit_posts_combined.json
│   │   └── reddit_comments_linked.json
│   └── processed/              # Processed CSV
│       ├── posts_final.csv
│       └── posts_final_bert_sentiment.csv
├── graphrag_test/              # GraphRAG knowledge graph
│   ├── input/                  # Input text files
│   ├── output/                 # Generated graph data
│   │   ├── entities.parquet    # Extracted entities
│   │   ├── relationships.parquet
│   │   ├── communities.parquet
│   │   └── community_reports.parquet
│   └── settings.yaml           # GraphRAG configuration
├── outputs/
│   └── figures/                # Visualizations
│       ├── bert_sentiment_*.png
│       ├── bertopic_*.html
│       ├── event_*.png
│       ├── network_*.png
│       └── stm_*.png
└── src/
    ├── config.py               # Configuration
    ├── data_collector.py       # Arctic Shift API
    ├── preprocessor.py         # Text preprocessing
    ├── sentiment_analyzer.py   # VADER analysis
    ├── sentiment_bert.py       # BERT analysis
    ├── topic_modeler.py        # LDA
    ├── topic_modeler_bert.py   # BERTopic
    ├── stm_analysis.py         # STM
    ├── event_analysis.py       # Event analysis
    ├── network_analyzer.py     # Network analysis
    └── misinfo_detector.py     # Gemini misinformation
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/reddit_US_NK.git
cd reddit_US_NK

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For Apple Silicon (MPS)
pip install torch torchvision torchaudio
```

---

## Data Sources

| Source | Time Range | API Key |
|--------|------------|---------|
| [Arctic Shift](https://arctic-shift.photon-reddit.com/) | 2005-2023 | No |
| [PRAW](https://praw.readthedocs.io/) | 2024+ | Yes |

---

## Key Events Analyzed

| Date | Event | Type |
|------|-------|------|
| 2009-05-25 | NK 2nd Nuclear Test | Provocation |
| 2010-03-26 | Cheonan Sinking | Provocation |
| 2010-11-23 | Yeonpyeong Shelling | Provocation |
| 2011-12-19 | Kim Jong-il Death | Transition |
| 2013-02-12 | NK 3rd Nuclear Test | Provocation |
| 2016-01-06 | NK 4th Nuclear Test | Provocation |
| 2016-09-09 | NK 5th Nuclear Test | Provocation |
| 2017-08-08 | Fire and Fury Speech | Rhetoric |
| 2017-09-03 | NK 6th Nuclear Test | Provocation |
| 2017-11-29 | Hwasong-15 ICBM | Provocation |
| 2018-06-12 | Singapore Summit | Diplomacy |
| 2019-02-28 | Hanoi Summit Failure | Diplomacy |
| 2019-06-30 | DMZ Meeting | Diplomacy |
| 2022-03-24 | Hwasong-17 ICBM | Provocation |
| 2022-10-04 | NK Missile Over Japan | Provocation |
| 2023-08-18 | Camp David Summit | Diplomacy |
| 2023-11-21 | NK Satellite Launch | Provocation |

---

## Research Applications

This framework can be adapted for:
- **Longitudinal conflict discourse** (e.g., India-Pakistan, China-Taiwan)
- **Public opinion tracking** around security events
- **Cross-platform narrative comparison**
- **Event-driven sentiment analysis**

---

## Citation

```bibtex
@misc{reddit_us_nk_analysis,
  author = {Jun Sin},
  title = {Reddit Discourse on North Korea and US-ROK Alliance: 15-Year Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/reddit_US_NK}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Arctic Shift](https://github.com/ArthurHeitmann/arctic_shift) - Historical Reddit data
- [PRAW](https://praw.readthedocs.io/) - Reddit API wrapper
- [BERTopic](https://maartengr.github.io/BERTopic/) - Topic modeling
- [HuggingFace Transformers](https://huggingface.co/) - BERT sentiment
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) - Knowledge graph extraction

---

*Created as part of research preparation for collaboration with Professor Mohit Singhal, UT Austin.*
