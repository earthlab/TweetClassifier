import os

import pandas as pd
import gensim
from gensim import corpora
import numpy as np
from gensim.corpora.sharded_corpus import ShardedCorpus
from src.utils import clean_tweet

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))


def save_lda_model():
    user_tweets_df = pd.read_csv(os.path.join(PROJ_DIR, 'data', 'training', 'training_data_without_lda_columns.csv'))
    user_tweets_df = user_tweets_df.rename(index=str, columns={'screen_name': 'screenname'})
    user_tweets_df.head()

    tweet_list = []
    user_list = []
    tweet_num_list = []
    tweet_num_list_true = []
    for user_index in range(len(user_tweets_df)):

        specific_user = user_tweets_df['screenname'][user_index]

        specific_user_tweets = user_tweets_df[user_tweets_df['screenname'] == specific_user]['condensed_tweets']
        specific_user_tweets = specific_user_tweets.values.item().split('EndOfTweet')

        num_tweets = len(specific_user_tweets)

        for tweet_index in range(len(specific_user_tweets)):
            tweet_list.append(specific_user_tweets[tweet_index])
            user_list.append(specific_user)
            tweet_num_list.append(tweet_index)

        tweet_num_list_true.append(num_tweets)
        if num_tweets != 201:
            padding = (201 - num_tweets)
            for i in range(padding):
                tweet_list.append('')
                user_list.append(specific_user)
                tweet_num_list.append(tweet_index + i + 1)

    tweet_array = np.asarray(tweet_list)

    clean_tweet_array = [clean_tweet(tweet) for tweet in tweet_array]
    tweet_tokens = [tweet.split() for tweet in clean_tweet_array]

    dictionary = corpora.Dictionary(tweet_tokens)
    dictionary.filter_extremes(no_below=2)
    dictionary.save(os.path.join(PROJ_DIR, 'data', 'models', 'dictionary.dict'))

    reloaded_dict = dictionary.load(os.path.join(PROJ_DIR, 'data', 'models', 'dictionary.dict'))

    tweet_term_matrix = [reloaded_dict.doc2bow(tweet) for tweet in tweet_tokens]

    corpus_dir = os.path.join(PROJ_DIR, 'data', 'models', 'lda_corpus')
    os.makedirs(corpus_dir, exist_ok=True)

    ShardedCorpus.serialize(os.path.join(corpus_dir, 'corpus.shdat'), tweet_term_matrix,
                            shardsize=2048, dim=len(dictionary),
                            sparse_serialization=True, sparse_retrieval=True)

    # Creating the object for LDA model using gensim library
    lda = gensim.models.ldamulticore.LdaMulticore

    # Running and Trainign LDA model on the document term matrix.
    lda_model = lda(tweet_term_matrix, num_topics=50,
                    id2word=dictionary, passes=50, workers=72)

    lda_model.print_topics(50)
    lda_model.save(os.path.join(PROJ_DIR, 'data', 'models', 'LDA_model'))
