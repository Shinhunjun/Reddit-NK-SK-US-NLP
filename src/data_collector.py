"""
Reddit Data Collection Module
- Arctic Shift API: Historical data (2022-2023)
- PRAW: Recent data (2024-present)

API Documentation: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Optional
from tqdm import tqdm

# Optional PRAW import
try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False
    print("PRAW not installed. Only Arctic Shift collection will be available.")

from config import (
    QUERY_TERMS, SUBREDDITS, DATE_CONFIG,
    ARCTIC_SHIFT_POSTS_ENDPOINT, ARCTIC_SHIFT_COMMENTS_ENDPOINT,
    REQUEST_DELAY, MAX_RETRIES
)


class ArcticShiftCollector:
    """
    Collect historical Reddit data using Arctic Shift API
    Data range: 2005 - 2023

    API Reference: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md
    """

    def __init__(self):
        self.base_url = "https://arctic-shift.photon-reddit.com/api/posts/search"
        self.comments_url = "https://arctic-shift.photon-reddit.com/api/comments/search"
        self.session = requests.Session()

    def search_posts(
        self,
        title_query: str,
        subreddit: str,
        after: str,
        before: str,
        limit: int = 100
    ) -> list:
        """
        Search posts from Arctic Shift API

        Args:
            title_query: Search term for title (full text search)
            subreddit: Target subreddit (required for title search)
            after: Start date (YYYY-MM-DD)
            before: End date (YYYY-MM-DD)
            limit: Max results per request (max 100, or "auto" for 100-1000)

        Returns:
            List of post dictionaries
        """
        params = {
            "subreddit": subreddit,
            "title": title_query,  # Use 'title' for full text search
            "after": after,
            "before": before,
            "limit": limit,
            "sort": "desc"  # Newest first
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                # Arctic Shift returns {"data": [...]} format
                return data.get("data", [])
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 2)
                else:
                    return []

    def collect_all(
        self,
        queries: list = None,
        subreddits: list = None,
        after: str = None,
        before: str = None,
        limit_per_query: int = 100
    ) -> list:
        """
        Collect posts for all query-subreddit combinations

        Args:
            queries: List of search terms (default: QUERY_TERMS)
            subreddits: List of subreddits (default: SUBREDDITS)
            after: Start date
            before: End date
            limit_per_query: Max posts per query

        Returns:
            List of all collected posts
        """
        queries = queries or QUERY_TERMS
        subreddits = subreddits or SUBREDDITS
        after = after or DATE_CONFIG["arctic_shift"]["start"]
        before = before or DATE_CONFIG["arctic_shift"]["end"]

        all_posts = []
        seen_ids = set()

        total_combinations = len(queries) * len(subreddits)
        pbar = tqdm(total=total_combinations, desc="Arctic Shift Collection")

        for subreddit in subreddits:
            for query in queries:
                posts = self.search_posts(
                    title_query=query,
                    subreddit=subreddit,
                    after=after,
                    before=before,
                    limit=limit_per_query
                )

                # Deduplicate
                for post in posts:
                    post_id = post.get("id")
                    if post_id and post_id not in seen_ids:
                        seen_ids.add(post_id)
                        post["source"] = "arctic_shift"
                        post["query_term"] = query
                        all_posts.append(post)

                pbar.update(1)
                time.sleep(REQUEST_DELAY)

        pbar.close()
        print(f"\nCollected {len(all_posts)} unique posts from Arctic Shift")
        return all_posts

    def get_comments_for_post(
        self,
        link_id: str,
        limit: int = 100
    ) -> list:
        """
        Get comments for a specific post using link_id

        Args:
            link_id: The post ID (e.g., 't3_abc123' or just 'abc123')
            limit: Max comments to retrieve

        Returns:
            List of comment dictionaries
        """
        # Remove t3_ prefix if present (API requires raw ID)
        if link_id.startswith('t3_'):
            link_id = link_id[3:]

        params = {
            "link_id": link_id,
            "limit": limit,
            "sort": "desc"  # Newest first (API only supports asc/desc)
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(self.comments_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed for {link_id}: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 2)
                else:
                    return []

    def collect_comments_for_posts(
        self,
        posts: list,
        comments_per_post: int = 50,
        max_posts: int = None
    ) -> list:
        """
        Collect comments for a list of posts

        Args:
            posts: List of post dictionaries (must have 'id' field)
            comments_per_post: Max comments per post
            max_posts: Limit number of posts to process (None = all)

        Returns:
            List of all collected comments
        """
        all_comments = []
        seen_ids = set()

        posts_to_process = posts[:max_posts] if max_posts else posts
        pbar = tqdm(total=len(posts_to_process), desc="Collecting Comments")

        for post in posts_to_process:
            post_id = post.get("id")
            if not post_id:
                pbar.update(1)
                continue

            comments = self.get_comments_for_post(post_id, limit=comments_per_post)

            for comment in comments:
                comment_id = comment.get("id")
                if comment_id and comment_id not in seen_ids:
                    seen_ids.add(comment_id)
                    comment["post_id"] = post_id
                    comment["post_title"] = post.get("title", "")
                    comment["source"] = "arctic_shift"
                    all_comments.append(comment)

            pbar.update(1)
            time.sleep(REQUEST_DELAY)

        pbar.close()
        print(f"\nCollected {len(all_comments)} unique comments from {len(posts_to_process)} posts")
        return all_comments

    def search_comments(
        self,
        body_query: str,
        subreddit: str,
        after: str,
        before: str,
        limit: int = 100
    ) -> list:
        """
        Search comments by body text

        Args:
            body_query: Search term for comment body
            subreddit: Target subreddit
            after: Start date (YYYY-MM-DD)
            before: End date (YYYY-MM-DD)
            limit: Max results

        Returns:
            List of comment dictionaries
        """
        params = {
            "subreddit": subreddit,
            "body": body_query,
            "after": after,
            "before": before,
            "limit": limit,
            "sort": "desc"
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(self.comments_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(REQUEST_DELAY * 2)
                else:
                    return []

    def collect_all_comments(
        self,
        queries: list = None,
        subreddits: list = None,
        after: str = None,
        before: str = None,
        limit_per_query: int = 100
    ) -> list:
        """
        Collect comments for all query-subreddit combinations (by body search)

        Args:
            queries: List of search terms
            subreddits: List of subreddits
            after: Start date
            before: End date
            limit_per_query: Max comments per query

        Returns:
            List of all collected comments
        """
        queries = queries or QUERY_TERMS
        subreddits = subreddits or SUBREDDITS
        after = after or DATE_CONFIG["arctic_shift"]["start"]
        before = before or DATE_CONFIG["arctic_shift"]["end"]

        all_comments = []
        seen_ids = set()

        total_combinations = len(queries) * len(subreddits)
        pbar = tqdm(total=total_combinations, desc="Arctic Shift Comments")

        for subreddit in subreddits:
            for query in queries:
                comments = self.search_comments(
                    body_query=query,
                    subreddit=subreddit,
                    after=after,
                    before=before,
                    limit=limit_per_query
                )

                for comment in comments:
                    comment_id = comment.get("id")
                    if comment_id and comment_id not in seen_ids:
                        seen_ids.add(comment_id)
                        comment["source"] = "arctic_shift"
                        comment["query_term"] = query
                        all_comments.append(comment)

                pbar.update(1)
                time.sleep(REQUEST_DELAY)

        pbar.close()
        print(f"\nCollected {len(all_comments)} unique comments from Arctic Shift")
        return all_comments


class PRAWCollector:
    """
    Collect recent Reddit data using PRAW (Reddit API)
    Requires Reddit API credentials
    """

    def __init__(
        self,
        client_id: str = None,
        client_secret: str = None,
        user_agent: str = None
    ):
        if not PRAW_AVAILABLE:
            raise ImportError("PRAW is not installed. Run: pip install praw")

        self.reddit = praw.Reddit(
            client_id=client_id or os.getenv("REDDIT_CLIENT_ID"),
            client_secret=client_secret or os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=user_agent or os.getenv("REDDIT_USER_AGENT", "reddit_us_nk_analysis/1.0")
        )

    def search_posts(
        self,
        query: str,
        subreddit: str,
        limit: int = 100,
        time_filter: str = "year"
    ) -> list:
        """
        Search posts using PRAW

        Args:
            query: Search term
            subreddit: Target subreddit
            limit: Max results
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'

        Returns:
            List of post dictionaries
        """
        posts = []
        try:
            sub = self.reddit.subreddit(subreddit)
            for submission in sub.search(query, limit=limit, time_filter=time_filter):
                post = {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext,
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "score": submission.score,
                    "num_comments": submission.num_comments,
                    "created_utc": int(submission.created_utc),
                    "subreddit": subreddit,
                    "permalink": submission.permalink,
                    "url": submission.url,
                    "upvote_ratio": submission.upvote_ratio,
                    "source": "praw",
                    "query_term": query
                }
                posts.append(post)
        except Exception as e:
            print(f"Error searching r/{subreddit} for '{query}': {e}")

        return posts

    def collect_all(
        self,
        queries: list = None,
        subreddits: list = None,
        limit_per_query: int = 100,
        time_filter: str = "year"
    ) -> list:
        """
        Collect posts for all query-subreddit combinations
        """
        queries = queries or QUERY_TERMS
        subreddits = subreddits or SUBREDDITS

        all_posts = []
        seen_ids = set()

        total_combinations = len(queries) * len(subreddits)
        pbar = tqdm(total=total_combinations, desc="PRAW Collection")

        for subreddit in subreddits:
            for query in queries:
                posts = self.search_posts(
                    query=query,
                    subreddit=subreddit,
                    limit=limit_per_query,
                    time_filter=time_filter
                )

                for post in posts:
                    post_id = post.get("id")
                    if post_id and post_id not in seen_ids:
                        seen_ids.add(post_id)
                        all_posts.append(post)

                pbar.update(1)
                time.sleep(0.5)  # Rate limiting for Reddit API

        pbar.close()
        print(f"\nCollected {len(all_posts)} unique posts from PRAW")
        return all_posts


def save_data(posts: list, filepath: str):
    """Save collected posts to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(posts)} posts to {filepath}")


def load_data(filepath: str) -> list:
    """Load posts from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# Quick test
if __name__ == "__main__":
    print("Testing Arctic Shift API...")
    print("API Docs: https://github.com/ArthurHeitmann/arctic_shift/blob/master/api/README.md\n")

    collector = ArcticShiftCollector()

    # Test single query - using 'title' parameter correctly
    test_posts = collector.search_posts(
        title_query="north korea",
        subreddit="worldnews",
        after="2023-01-01",
        before="2023-12-31",
        limit=10
    )

    print(f"\nTest results: {len(test_posts)} posts found")

    if test_posts:
        sample = test_posts[0]
        print(f"\nSample post:")
        print(f"  Title: {sample.get('title', 'N/A')[:80]}...")
        print(f"  Date: {datetime.fromtimestamp(sample.get('created_utc', 0))}")
        print(f"  Score: {sample.get('score', 'N/A')}")
        print(f"  Subreddit: r/{sample.get('subreddit', 'N/A')}")
    else:
        print("\nNo posts found. Checking API response...")
        # Debug: try raw request
        import requests
        r = requests.get(
            "https://arctic-shift.photon-reddit.com/api/posts/search",
            params={"subreddit": "worldnews", "title": "north korea", "after": "2023-01-01", "before": "2023-12-31", "limit": 5}
        )
        print(f"Status: {r.status_code}")
        print(f"Response: {r.text[:500]}")
