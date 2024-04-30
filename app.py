import json
import os

import pandas as pd

from src.run_inference import InferenceType, InferenceRole
from src.pull_user_tweets import TweetFilter
from src.derive_lda_columns import LDA

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import tweepy


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////app/persistent_data/database.db'
db = SQLAlchemy(app)

contrib_role = 'civic/public sector', 'distribution', 'em', 'media', 'personalized'
contrib_type = 'feed based', 'individual', 'organization'
PROJ_DIR = os.path.dirname(__file__)


class Author(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    author_id = db.Column(db.String(200), nullable=False, unique=True)
    username = db.Column(db.String(200), nullable=False, unique=True)

    contributor_type = db.Column(db.String(100))
    contributor_role = db.Column(db.String(100))

    tweets_df = db.Column(db.String(300), unique=True)

    def serialize(self):
        return {
            'author_id': self.author_id,
            'username': self.username,
            'contributor_type': self.contributor_type,
            'contributor_role': self.contributor_role,
            'profile_url': self.get_twitter_profile_url()
        }

    def get_twitter_profile_url(self):
        if self.username:
            return f"https://twitter.com/{self.username}"
        elif self.author_id:
            return f"https://twitter.com/user_id/{self.author_id}"
        else:
            return None


with app.app_context():
    db.create_all()
    type_inference = InferenceType()
    role_inference = InferenceRole()
    lda_model = LDA()

    app.type_inference = type_inference
    app.role_inference = role_inference
    app.lda_model = lda_model
    app.tweepy_client = tweepy.Client(bearer_token=os.environ.get('X_BEARER_TOKEN'))
    os.makedirs(os.path.join(PROJ_DIR, 'persistent_data', 'user_tweet_dfs'), exist_ok=True)


@app.route('/classify_authors')
def classify_authors():
    query = request.args.get('search_query')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    max_results = request.args.get('max_results')
    max_results = int(max_results) if max_results is not None else None

    try:
        tweets = app.tweepy_client.search_recent_tweets(query=query, start_time=start_time, end_time=end_time,
                                                        max_results=max_results, user_fields=['username', 'id'],
                                                        expansions='author_id')
    except tweepy.errors.TooManyRequests:
        return jsonify({
            'error': 'Too Many Requests.',
            'message': 'The request limit for search_recent_tweets was exceeded. Please try again later.'
        }), 429

    # Access expanded objects from the includes attribute
    users = tweets.includes['users']

    # Iterate over the tweets and print tweet text, author ID, and username
    authors = {}
    for user in users:
        if user.id not in authors:
            authors[user.id] = {
                'id': user.id,
                'username': user.username
            }

    # Classify each author
    needs_classification = []
    classified = []
    for author in authors.values():
        existing_author = db.session.query(Author).filter(Author.author_id == author['id']).first()

        if existing_author is None:
            print('New author')
            existing_author = Author(author_id=author['id'], username=author['username'])
            db.session.add(existing_author)
            db.session.flush()

        if existing_author.contributor_type is not None and existing_author.contributor_role is not None:
            classified.append(existing_author)
        else:
            needs_classification.append(existing_author)

    db.session.commit()

    print(classified)
    print(needs_classification)

    exceeded_requests = False
    for author in needs_classification:
        if author.tweets_df is None:
            print('Getting tweets')
            try:
                tweet_puller = TweetFilter(app.tweepy_client)
                tweets_df = tweet_puller.etf_user_timeline_extract_apiv2(author.author_id, max_results=50)
                tweets_df_path = os.path.join(PROJ_DIR, 'persistent_data', 'user_tweet_dfs', f'{author.author_id}.csv')
                tweets_df.to_csv(tweets_df_path)
                author.tweets_df = tweets_df_path
                db.session.commit()
            except tweepy.errors.TooManyRequests:
                exceeded_requests = True
                break
        else:
            print('Tweet df already found')

    print(needs_classification)

    for author in needs_classification:
        if author.tweets_df is not None:
            try:
                print(f'Classifying author: {author.author_id}')
                tweets_df = pd.read_csv(author.tweets_df)
                tweets_lda_df = app.lda_model.add_lda_columns(tweets_df)
                tweets_lda_df.columns = [str(c) for c in tweets_lda_df.columns]

                contributor_type = app.type_inference.run_inference(tweets_lda_df)
                if contributor_type:
                    author.contributor_type = contributor_type[0]
                contributor_role = app.role_inference.run_inference(tweets_lda_df)
                if contributor_role:
                    author.contributor_role = contributor_role[0]

                db.session.commit()

            except Exception as e:
                print(str(e))

    classifications = [a.serialize() for a in needs_classification + classified]

    response = {
        'classifications': classifications
    }
    if exceeded_requests:
        response["error"] = "Too Many Requests"
        response["message"] = ("The get users tweets request amount has been exceeded. As many accounts as could be"
                               " classified were, but you will need to call this later to classify the rest of them.")

        return jsonify(json.loads(json.dumps(response, indent=4))), 429
    else:
        response["message"] = "Request processed successfully."
        return jsonify(json.loads(json.dumps(response, indent=4))), 200


@app.route('/get_all_authors')
def get_all_classifications():
    response = {
        'classifications': [a.serialize() for a in db.session.query(Author).all()]
    }

    return jsonify(json.loads(json.dumps(response, indent=4))), 200
