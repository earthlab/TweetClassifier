import os

import tweepy
from datetime import datetime


class Twitter:
    def __init__(self):
        self._client = tweepy.Client(bearer_token=os.environ.get('X_BEARER_TOKEN'))

    def search_tweets(self, query: str, start_time: datetime, end_time: datetime):
        pass

    def get_user_tweets(self):
        pass