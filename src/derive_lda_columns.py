import os
import pandas as pd
import string
import re
import gensim
from gensim import corpora
import numpy as np
from gensim.corpora.sharded_corpus import ShardedCorpus
import multiprocessing
from src.utils import clean_tweet


PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


class LDA:
    def __init__(self):
        self._dictionary_path = os.path.join(PROJ_DIR, 'data', 'models', 'dictionary.dict')
        self._sharded_corpus_dest = os.path.join(PROJ_DIR, 'data', 'models', 'corpus.shdat')
        self._lda_model_path = os.path.join(PROJ_DIR, 'data', 'models', 'LDA_model')

    @staticmethod
    def _prep_new(tweet, dict_to_use):
        phase1 = clean_tweet(tweet).split()
        phase2 = dict_to_use.doc2bow(phase1)
        return phase2

    # Defining function to get consistent LDA topic arrays
    @staticmethod
    def _make_lda_consistent(lda_output):
        current_lda_scores = []
        topics_considered = []
        for i in range(len(lda_output)):
            topics_considered.append(lda_output[i][0])
        for topics in range(50):
            if topics in topics_considered:
                topic_index = topics_considered.index(topics)
                current_lda_scores.append(lda_output[topic_index][1])
            else:
                current_lda_scores.append(0)
        current_lda_scores_array = np.asarray(current_lda_scores)
        return current_lda_scores_array

    @staticmethod
    def _merge_df(user_tweets_df: pd.DataFrame, lda_df: pd.DataFrame):
        lda_df.rename(columns={'user': 'screen_name'}, inplace=True)
        ensemble_ready_df = pd.merge(user_tweets_df, lda_df, on='screen_name')
        return ensemble_ready_df.rename(columns={'followers': 'u_followers_count',
                                                 'status_count': 'u_statuses_count'})

    def add_lda_columns(self, user_tweets_df: pd.DataFrame):
        screen_name_column = 'screen_name'

        # Removing duplicate users and resetting dataframe indices
        user_tweets_df = user_tweets_df.drop_duplicates(subset=screen_name_column)
        user_tweets_df = user_tweets_df.reset_index(drop=True)

        # Getting each of all user's tweets as their own 'document' for topic model results
        # Empty lists to build on
        tweet_list = []
        user_list = []
        tweet_num_list = []
        tweet_num_list_true = []
        # The loop over all users
        for user_index in range(len(user_tweets_df)):
            # Getting the user's name
            specific_user = user_tweets_df[screen_name_column][user_index]
            # Getting their condensed tweets
            specific_user_tweets = user_tweets_df[user_tweets_df[screen_name_column] == specific_user]['condensed_tweets']
            # Splitting those tweets based on my EOL word
            specific_user_tweets = specific_user_tweets.values.item().split('EndOfTweet')
            # Getting the number of tweets for that user
            num_tweets = len(specific_user_tweets)
            # For all those tweets
            for tweet_index in range(len(specific_user_tweets)):
                # Append their tweet
                tweet_list.append(specific_user_tweets[tweet_index])
                # Append their name
                user_list.append(specific_user)
                # Append the tweet number
                tweet_num_list.append(tweet_index)
            # Keeping a list of the number of tweets to expect from each user
            tweet_num_list_true.append(num_tweets)
            # Figuring out how much padding we need to do
            if num_tweets != 201:
                padding = (201 - num_tweets)
                # Padding
                for i in range(padding):
                    tweet_list.append('')
                    user_list.append(specific_user)
                    tweet_num_list.append(tweet_index + i + 1)
        # Converting the tweet list to an array
        tweet_array = np.asarray(tweet_list)

        # Cleaning all the tweets
        clean_tweet_array = [clean_tweet(tweet) for tweet in tweet_array]

        # Tokenizing all the tweets
        tweet_tokens = [tweet.split() for tweet in clean_tweet_array]

        dictionary = corpora.Dictionary(tweet_tokens)
        reloaded_dict = dictionary.load(self._dictionary_path)

        # Converting list of tweets (corpus) into a tweet term matrix using dictionary prepared above.
        tweet_term_matrix = [reloaded_dict.doc2bow(tweet) for tweet in tweet_tokens]

        # To reduce bottlenecks in the parallelized LDA we need a sharded (parallelized) corpus
        ShardedCorpus.serialize(self._sharded_corpus_dest, tweet_term_matrix,
                                shardsize=2048, dim=len(dictionary),
                                sparse_serialization=True, sparse_retrieval=True)

        lda = gensim.models.ldamulticore.LdaMulticore

        lda_model = lda.load(self._lda_model_path)

        tweets_df = pd.DataFrame({'name': user_list,
                                  'tweet': tweet_list,
                                  'tweet_num': tweet_num_list})
        tweets_df = tweets_df.pivot(index='name',
                                    columns='tweet_num',
                                    values='tweet')
        tweets_df = tweets_df.drop(columns=200)
        tweets_df = tweets_df.reset_index(drop=True)
        tweets_df['user'] = np.unique(user_list)

        for row in range(len(tweets_df)):
            # Getting topic scores for each tweet
            for tweet_num in range(200):
                current_tweet = tweets_df[tweet_num][row]
                if len(current_tweet) > 0:
                    current_lda_array = lda_model[(self._prep_new(current_tweet, reloaded_dict))]
                    formatted_lda_array = self._make_lda_consistent(current_lda_array)
                    formatted_lda_array = formatted_lda_array.reshape(50, 1)
                    if tweet_num == 0:
                        user_lda_matrix = formatted_lda_array
                    else:
                        user_lda_matrix = np.append(user_lda_matrix, formatted_lda_array, axis=1)
            # Getting topic variance / homogeneity - determined across all tweets
            average_lda_array = np.mean(user_lda_matrix, axis=1)
            total_variance = 0
            for matrix_row in range(user_lda_matrix.shape[1]):
                current_lda_row = user_lda_matrix[:, matrix_row]
                spread_from_average = np.sum(np.square(current_lda_row - average_lda_array))
                total_variance += spread_from_average
            scaled_variance = (total_variance / matrix_row)
            # Getting top topics - determined by largest average topic values
            one_hot_vector = np.zeros(50)
            sorted_lda_topics = np.argsort(average_lda_array)
            for topic_index in range(1, 11):
                actual_topic_num = sorted_lda_topics[-1 * topic_index]
                current_topic_score = average_lda_array[actual_topic_num]
                one_hot_vector[actual_topic_num] = current_topic_score
            reshaped_one_hot = one_hot_vector.reshape(1, 50)
            # Making the data frame
            user_name = tweets_df['user'][row]
            if row == 0:
                lda_df = pd.DataFrame(reshaped_one_hot)
                lda_df['topic_variance'] = scaled_variance
                lda_df['user'] = user_name
            else:
                temp_df = pd.DataFrame(reshaped_one_hot)
                temp_df['topic_variance'] = scaled_variance
                temp_df['user'] = user_name
                lda_df = pd.concat((lda_df, temp_df))

        return self._merge_df(user_tweets_df, lda_df)
