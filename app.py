import os

from src.run_inference import TweetAuthorInference, TweetTypeInference
from src.pull_user_tweets import TweetFilter
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

    tweets_df = db.Column(db.String(300), unique=True)

    # relationships

    # many-to-one
    tweets = db.relationship('Tweet', back_populates='author', cascade="delete, delete-orphan")


class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tweet_id = db.Column(db.String(100), unique=True)
    text = db.Column(db.String(4000), unique=False)

    # relationships

    # one-to-many
    author_id = db.Column(db.Integer, db.ForeignKey('author.id'))
    author = db.relationship('Author', back_populates='tweets')


with app.app_context():
    db.create_all()
    tweet_author_inference = TweetAuthorInference()
    tweet_type_inference = TweetTypeInference()
    lda_model = LDA()

    app.tweet_author_inference = tweet_author_inference
    app.tweet_type_inference = tweet_type_inference
    app.lda_model = lda_model
    app.tweepy_client = tweepy.Client(bearer_token=os.environ.get('X_BEARER_TOKEN'))
    tweet_puller = TweetFilter(app.tweepy_client)


@app.route('/classify_authors')
def classify_authors():
    query = request.args.get('search_query')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    max_results = request.args.get('max_results')
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
    needs_classification = []
    classified = []
    for author in authors:
        existing_author = db.session.query(Author).filter(Author.autohr_id == author['id']).first()

        if existing_author is None:
            new_author = Author(author_id=author['id'], username=author['username'])
            db.session.add(new_author)
            db.session.flush()
            needs_classification.append(new_author)
        else:
            if existing_author.contributor_type is not None and existing_author.contributor_role is not None:
                classified.append(existing_author)

    db.session.commit()

    for author in needs_classification:
        existing_tweets = author.tweets
