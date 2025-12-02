"""
Network Analysis for Reddit Comment Trees
Analyzes propagation patterns, depth, branching, and hub users
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class CommentNetworkAnalyzer:
    """
    Analyze Reddit comment tree networks for propagation patterns
    """

    def __init__(self, comments_path: str, posts_path: Optional[str] = None):
        """
        Initialize with comments data

        Args:
            comments_path: Path to JSON file with comments
            posts_path: Optional path to posts CSV for additional context
        """
        print("Loading comments data...")
        with open(comments_path, 'r', encoding='utf-8') as f:
            self.comments = json.load(f)

        # Filter valid comments (not deleted/removed)
        self.comments = [
            c for c in self.comments
            if c.get('body') not in ['[deleted]', '[removed]', None, '']
            and c.get('author') not in ['[deleted]', '[removed]']
        ]

        print(f"Loaded {len(self.comments):,} valid comments")

        # Group comments by post
        self.comments_by_post = defaultdict(list)
        for c in self.comments:
            post_id = c.get('post_id') or c.get('link_id', '').replace('t3_', '')
            self.comments_by_post[post_id].append(c)

        print(f"Comments span {len(self.comments_by_post):,} posts")

        # Load posts if provided
        self.posts_df = None
        if posts_path and os.path.exists(posts_path):
            self.posts_df = pd.read_csv(posts_path)
            print(f"Loaded {len(self.posts_df):,} posts")

    def build_comment_tree(self, post_id: str) -> nx.DiGraph:
        """
        Build a directed graph representing comment tree for a post

        Args:
            post_id: The post ID

        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        comments = self.comments_by_post.get(post_id, [])

        # Add root node (the post itself)
        G.add_node(post_id, type='post', author='[OP]')

        for c in comments:
            comment_id = c.get('id')
            parent_id = c.get('parent_id', '').replace('t1_', '').replace('t3_', '')
            author = c.get('author', '[deleted]')
            score = c.get('score', 0)
            created_utc = c.get('created_utc', 0)

            # Add comment node
            G.add_node(comment_id, type='comment', author=author,
                       score=score, created_utc=created_utc)

            # Add edge from parent to comment
            if parent_id:
                G.add_edge(parent_id, comment_id)

        return G

    def calculate_post_metrics(self, post_id: str) -> Dict:
        """
        Calculate network metrics for a single post's comment tree

        Args:
            post_id: The post ID

        Returns:
            Dict with metrics
        """
        G = self.build_comment_tree(post_id)
        comments = self.comments_by_post.get(post_id, [])

        if G.number_of_nodes() <= 1:  # Only root node
            return {
                'post_id': post_id,
                'comment_count': 0,
                'depth': 0,
                'max_depth': 0,
                'branching_factor': 0,
                'propagation_speed': 0,
                'unique_authors': 0,
                'avg_score': 0,
                'total_score': 0
            }

        # Calculate depth (longest path from root)
        try:
            if nx.is_directed_acyclic_graph(G):
                max_depth = nx.dag_longest_path_length(G)
            else:
                max_depth = 0
        except:
            max_depth = 0

        # Calculate branching factor (average out-degree)
        out_degrees = [d for n, d in G.out_degree() if d > 0]
        branching_factor = np.mean(out_degrees) if out_degrees else 0

        # Propagation speed (time from first to last comment)
        timestamps = [c.get('created_utc', 0) for c in comments if c.get('created_utc')]
        if len(timestamps) >= 2:
            propagation_speed = max(timestamps) - min(timestamps)  # in seconds
        else:
            propagation_speed = 0

        # Author analysis
        authors = [c.get('author') for c in comments if c.get('author')]
        unique_authors = len(set(authors))

        # Score analysis
        scores = [c.get('score', 0) for c in comments]
        avg_score = np.mean(scores) if scores else 0
        total_score = sum(scores)

        return {
            'post_id': post_id,
            'comment_count': len(comments),
            'node_count': G.number_of_nodes(),
            'edge_count': G.number_of_edges(),
            'max_depth': max_depth,
            'branching_factor': branching_factor,
            'propagation_speed': propagation_speed,
            'propagation_speed_hours': propagation_speed / 3600 if propagation_speed else 0,
            'unique_authors': unique_authors,
            'avg_score': avg_score,
            'total_score': total_score
        }

    def analyze_all_posts(self, max_posts: Optional[int] = None) -> pd.DataFrame:
        """
        Calculate metrics for all posts

        Args:
            max_posts: Limit number of posts (None = all)

        Returns:
            DataFrame with metrics
        """
        post_ids = list(self.comments_by_post.keys())
        if max_posts:
            post_ids = post_ids[:max_posts]

        results = []
        for post_id in tqdm(post_ids, desc="Analyzing comment networks"):
            metrics = self.calculate_post_metrics(post_id)
            results.append(metrics)

        df = pd.DataFrame(results)
        return df

    def identify_hub_users(self, top_n: int = 50) -> pd.DataFrame:
        """
        Identify most active and influential users

        Args:
            top_n: Number of top users to return

        Returns:
            DataFrame with user statistics
        """
        user_stats = defaultdict(lambda: {
            'comment_count': 0,
            'total_score': 0,
            'posts_participated': set(),
            'avg_depth': [],
            'reply_count': 0
        })

        # Build full graph to calculate centrality
        print("Building full comment graph...")

        for post_id, comments in tqdm(self.comments_by_post.items(), desc="Processing posts"):
            for c in comments:
                author = c.get('author', '[deleted]')
                if author in ['[deleted]', '[removed]']:
                    continue

                user_stats[author]['comment_count'] += 1
                user_stats[author]['total_score'] += c.get('score', 0)
                user_stats[author]['posts_participated'].add(post_id)

                # Check if this is a reply (parent is another comment)
                parent_id = c.get('parent_id', '')
                if parent_id.startswith('t1_'):
                    user_stats[author]['reply_count'] += 1

        # Convert to DataFrame
        rows = []
        for author, stats in user_stats.items():
            rows.append({
                'author': author,
                'comment_count': stats['comment_count'],
                'total_score': stats['total_score'],
                'posts_participated': len(stats['posts_participated']),
                'reply_count': stats['reply_count'],
                'avg_score': stats['total_score'] / stats['comment_count'] if stats['comment_count'] > 0 else 0,
                'reply_ratio': stats['reply_count'] / stats['comment_count'] if stats['comment_count'] > 0 else 0
            })

        df = pd.DataFrame(rows)
        df = df.sort_values('comment_count', ascending=False).head(top_n)

        return df

    def analyze_by_era(self, posts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze network metrics by presidential era

        Args:
            posts_df: DataFrame with 'id' and 'era' columns

        Returns:
            DataFrame with era-level statistics
        """
        # Create post_id to era mapping
        post_era_map = dict(zip(posts_df['id'].astype(str), posts_df['era']))

        # Calculate metrics for all posts
        all_metrics = self.analyze_all_posts()

        # Add era column
        all_metrics['era'] = all_metrics['post_id'].map(post_era_map)

        # Aggregate by era
        era_stats = all_metrics.groupby('era').agg({
            'comment_count': ['mean', 'sum', 'count'],
            'max_depth': ['mean', 'max'],
            'branching_factor': 'mean',
            'propagation_speed_hours': 'mean',
            'unique_authors': 'mean',
            'avg_score': 'mean'
        }).round(2)

        era_stats.columns = ['_'.join(col) for col in era_stats.columns]

        return era_stats

    def visualize_metrics_distribution(
        self,
        metrics_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize distribution of network metrics

        Args:
            metrics_df: DataFrame with metrics
            output_path: Optional save path
        """
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # 1. Comment count distribution
        ax = axes[0, 0]
        ax.hist(metrics_df['comment_count'], bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Comment Count')
        ax.set_ylabel('Frequency')
        ax.set_title('Comment Count Distribution')
        ax.axvline(metrics_df['comment_count'].median(), color='r', linestyle='--',
                   label=f'Median: {metrics_df["comment_count"].median():.0f}')
        ax.legend()

        # 2. Max depth distribution
        ax = axes[0, 1]
        ax.hist(metrics_df['max_depth'], bins=30, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Max Depth')
        ax.set_ylabel('Frequency')
        ax.set_title('Conversation Depth Distribution')
        ax.axvline(metrics_df['max_depth'].median(), color='r', linestyle='--',
                   label=f'Median: {metrics_df["max_depth"].median():.0f}')
        ax.legend()

        # 3. Branching factor distribution
        ax = axes[0, 2]
        valid_bf = metrics_df['branching_factor'][metrics_df['branching_factor'] > 0]
        ax.hist(valid_bf, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax.set_xlabel('Branching Factor')
        ax.set_ylabel('Frequency')
        ax.set_title('Branching Factor Distribution')
        ax.axvline(valid_bf.median(), color='r', linestyle='--',
                   label=f'Median: {valid_bf.median():.2f}')
        ax.legend()

        # 4. Propagation speed
        ax = axes[1, 0]
        valid_speed = metrics_df['propagation_speed_hours'][metrics_df['propagation_speed_hours'] > 0]
        ax.hist(valid_speed, bins=50, edgecolor='black', alpha=0.7, color='purple')
        ax.set_xlabel('Propagation Speed (hours)')
        ax.set_ylabel('Frequency')
        ax.set_title('Discussion Duration Distribution')
        ax.axvline(valid_speed.median(), color='r', linestyle='--',
                   label=f'Median: {valid_speed.median():.1f}h')
        ax.legend()

        # 5. Unique authors
        ax = axes[1, 1]
        ax.hist(metrics_df['unique_authors'], bins=30, edgecolor='black', alpha=0.7, color='brown')
        ax.set_xlabel('Unique Authors')
        ax.set_ylabel('Frequency')
        ax.set_title('Participation Breadth Distribution')

        # 6. Depth vs Comment count scatter
        ax = axes[1, 2]
        ax.scatter(metrics_df['comment_count'], metrics_df['max_depth'],
                   alpha=0.3, s=10)
        ax.set_xlabel('Comment Count')
        ax.set_ylabel('Max Depth')
        ax.set_title('Depth vs Volume')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig

    def visualize_hub_users(
        self,
        hub_df: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize hub user statistics

        Args:
            hub_df: DataFrame with hub user statistics
            output_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # 1. Top users by comment count
        ax = axes[0, 0]
        top_10 = hub_df.head(10)
        ax.barh(range(len(top_10)), top_10['comment_count'])
        ax.set_yticks(range(len(top_10)))
        ax.set_yticklabels(top_10['author'])
        ax.set_xlabel('Comment Count')
        ax.set_title('Top 10 Most Active Users')
        ax.invert_yaxis()

        # 2. Top users by total score
        ax = axes[0, 1]
        top_score = hub_df.nlargest(10, 'total_score')
        ax.barh(range(len(top_score)), top_score['total_score'], color='green')
        ax.set_yticks(range(len(top_score)))
        ax.set_yticklabels(top_score['author'])
        ax.set_xlabel('Total Score')
        ax.set_title('Top 10 Highest Total Score')
        ax.invert_yaxis()

        # 3. Activity vs Score scatter
        ax = axes[1, 0]
        ax.scatter(hub_df['comment_count'], hub_df['total_score'], alpha=0.5)
        ax.set_xlabel('Comment Count')
        ax.set_ylabel('Total Score')
        ax.set_title('Activity vs Engagement')

        # Highlight top users
        for _, row in hub_df.head(5).iterrows():
            ax.annotate(row['author'][:15], (row['comment_count'], row['total_score']),
                        fontsize=8, alpha=0.8)

        # 4. Reply ratio distribution
        ax = axes[1, 1]
        ax.hist(hub_df['reply_ratio'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Reply Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Reply Ratio Distribution (Hub Users)')
        ax.axvline(hub_df['reply_ratio'].median(), color='r', linestyle='--',
                   label=f'Median: {hub_df["reply_ratio"].median():.2f}')
        ax.legend()

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig

    def visualize_by_era(
        self,
        era_metrics: pd.DataFrame,
        output_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize network metrics by presidential era

        Args:
            era_metrics: DataFrame with era-level stats
            output_path: Optional save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        era_order = ['obama1', 'obama2', 'trump', 'biden']
        era_colors = {'obama1': 'blue', 'obama2': 'lightblue', 'trump': 'red', 'biden': 'green'}

        # Filter and reorder
        era_metrics = era_metrics.reindex([e for e in era_order if e in era_metrics.index])

        # 1. Average depth by era
        ax = axes[0, 0]
        colors = [era_colors.get(e, 'gray') for e in era_metrics.index]
        ax.bar(era_metrics.index, era_metrics['max_depth_mean'], color=colors)
        ax.set_ylabel('Average Max Depth')
        ax.set_title('Conversation Depth by Era')

        # 2. Average branching factor by era
        ax = axes[0, 1]
        ax.bar(era_metrics.index, era_metrics['branching_factor_mean'], color=colors)
        ax.set_ylabel('Average Branching Factor')
        ax.set_title('Branching Factor by Era')

        # 3. Average propagation speed by era
        ax = axes[1, 0]
        ax.bar(era_metrics.index, era_metrics['propagation_speed_hours_mean'], color=colors)
        ax.set_ylabel('Average Duration (hours)')
        ax.set_title('Discussion Duration by Era')

        # 4. Average comment count by era
        ax = axes[1, 1]
        ax.bar(era_metrics.index, era_metrics['comment_count_mean'], color=colors)
        ax.set_ylabel('Average Comments per Post')
        ax.set_title('Comment Volume by Era')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {output_path}")

        return fig


def run_network_analysis(
    comments_path: str,
    posts_path: str,
    output_dir: str = '../outputs/figures'
) -> Tuple[pd.DataFrame, pd.DataFrame, CommentNetworkAnalyzer]:
    """
    Run full network analysis

    Args:
        comments_path: Path to comments JSON
        posts_path: Path to posts CSV
        output_dir: Output directory

    Returns:
        Tuple of (metrics_df, hub_users_df, analyzer)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize analyzer
    analyzer = CommentNetworkAnalyzer(comments_path, posts_path)

    # Calculate metrics for all posts
    print("\nCalculating network metrics...")
    metrics_df = analyzer.analyze_all_posts()

    print("\nNetwork Metrics Summary:")
    print(f"  Posts analyzed: {len(metrics_df):,}")
    print(f"  Avg comments/post: {metrics_df['comment_count'].mean():.1f}")
    print(f"  Avg max depth: {metrics_df['max_depth'].mean():.1f}")
    print(f"  Avg branching factor: {metrics_df['branching_factor'].mean():.2f}")

    # Identify hub users
    print("\nIdentifying hub users...")
    hub_users_df = analyzer.identify_hub_users(top_n=50)

    print("\nTop 10 Most Active Users:")
    print(hub_users_df.head(10)[['author', 'comment_count', 'total_score', 'posts_participated']])

    # Generate visualizations
    print("\nGenerating visualizations...")

    analyzer.visualize_metrics_distribution(
        metrics_df,
        output_path=os.path.join(output_dir, 'network_metrics_distribution.png')
    )

    analyzer.visualize_hub_users(
        hub_users_df,
        output_path=os.path.join(output_dir, 'hub_users_analysis.png')
    )

    # Era analysis if posts have era column
    posts_df = pd.read_csv(posts_path)
    if 'era' in posts_df.columns:
        print("\nAnalyzing by era...")
        era_metrics = analyzer.analyze_by_era(posts_df)
        print(era_metrics)

        analyzer.visualize_by_era(
            era_metrics,
            output_path=os.path.join(output_dir, 'network_by_era.png')
        )

        era_metrics.to_csv(os.path.join(output_dir, 'network_era_stats.csv'))

    # Save results
    metrics_df.to_csv(os.path.join(output_dir, 'network_metrics.csv'), index=False)
    hub_users_df.to_csv(os.path.join(output_dir, 'hub_users.csv'), index=False)

    print(f"\nResults saved to {output_dir}")

    return metrics_df, hub_users_df, analyzer


if __name__ == "__main__":
    print("Network Analysis for Reddit US-NK Data")
    print("="*50)

    comments_path = "../data/raw/reddit_comments_linked.json"
    posts_path = "../data/processed/posts_final.csv"

    if os.path.exists(comments_path):
        metrics_df, hub_users_df, analyzer = run_network_analysis(
            comments_path=comments_path,
            posts_path=posts_path,
            output_dir='../outputs/figures'
        )
    else:
        print(f"Comments file not found: {comments_path}")
