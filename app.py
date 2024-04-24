import os

from src.run_inference import TweetAuthorInference, TweetTypeInference
from src.derive_lda_columns import LDA

from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
import tweepy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db'
db = SQLAlchemy(app)

contrib_role = 'civic/public sector', 'distribution', 'em', 'media', 'personalized'
contrib_type = 'feed based', 'individual', 'organization'

class Author(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    author_id = db.Column(db.String(200), nullable=False, unique=True)
    username = db.Column(db.String(200), nullable=False, unique=True)

    contributor_type = db.Column(db.String(100))
    contributor_role = db.Column(db.String(100))


with app.app_context():
    db.create_all()
    tweet_author_inference = TweetAuthorInference()
    tweet_type_inference = TweetTypeInference()
    lda_model = LDA()

    app.tweet_author_inference = tweet_author_inference
    app.tweet_type_inference = tweet_type_inference
    app.lda_model = lda_model
    app.tweepy_client = tweepy.Client(bearer_token=os.environ.get('X_BEARER_TOKEN'))


@app.route('/classify_authors')
def classify_authors():
    query = request.args.get('search_query')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    max_results = request.args.get('max_results')
    reclassify = request.args.get('reclassify')
    max_results = int(max_results) if max_results is not None else None

    tweets = app.tweepy_client.search_recent_tweets(query=query, start_time=start_time, end_time=end_time,
                                                    max_results=max_results, user_fields=['username', 'id'],
                                                    expansions='author_id')

    # Access expanded objects from the includes attribute
    users = tweets.includes['users']

    # Iterate over the tweets and print tweet text, author ID, and username
    authors = []
    for tweet in tweets.data:
        tweet_text = tweet.text
        author_id = tweet.author_id

        # Retrieve user object from includes using author ID
        for user in users:
            if user.id == author_id:
                authors.append({
                    'id': user.id,
                    'usernmae': user.username
                })

    # Classify each author
    for author in authors:

        if not reclassify:
            # First query the database
