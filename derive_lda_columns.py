import os
import pandas as pd
import string
import re
import gensim
from gensim import corpora
import numpy as np
from gensim.corpora.sharded_corpus import ShardedCorpus
import multiprocessing

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


input_csv = os.path.join(DATA_DIR, 'truncated.csv')
screenname_column = 'screen_name'
dictionary_path = os.path.join(DATA_DIR, 'dictionary.dict')
sharded_corpus_dest = os.path.join(DATA_DIR, 'corpus.shdat')
lda_model_path = os.path.join(DATA_DIR, 'LDA_model')
output_csv = 'PROTO_lda.csv'

user_tweets_df = pd.read_csv(input_csv)

# Removing duplicate users and resetting dataframe indices
user_tweets_df = user_tweets_df.drop_duplicates(subset=screenname_column)
user_tweets_df = user_tweets_df.reset_index(drop=True)

# Getting each of all user's tweets as their own 'document' for topic model results
# Empty lists to build on
tweet_list = []
user_list = []
tweet_num_list = []
tweet_num_list_TRUE = []
# The loop over all users
for user_index in range(len(user_tweets_df)):
    # Getting the user's name
    specific_user = user_tweets_df[screenname_column][user_index]
    # Getting their condensed tweets
    specific_user_tweets = user_tweets_df[user_tweets_df[screenname_column] == specific_user]['condensed_tweets']
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
    tweet_num_list_TRUE.append(num_tweets)
    # Figuring out how much padding we need to do
    if (num_tweets != 201):
        padding = (201 - num_tweets)
        # Padding
        for i in range(padding):
            tweet_list.append('')
            user_list.append(specific_user)
            tweet_num_list.append(tweet_index + i + 1)
# Converting the tweet list to an array
tweet_array = np.asarray(tweet_list)


def clean_tweet(tweet):
    """Function to perform standard tweet processing. Includes removing URLs, my
       end-of-line character, punctuation, lower casing, and recombining the rt
       character. Inputs a str, outputs a str"""
    # For later - to remove punctuation
    # Setting rule to replace any punctuation chain with equal amount of white space
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    # Remove end-of-line character
    phase1 = re.sub('EndOfTweet', ' ', tweet)
    # Remove URLs
    phase2 = re.sub(r'http\S+', '', phase1)
    # Remove punctuation
    phase3 = phase2.translate(replace_punctuation)
    # Seperate individual characters entities based on capitalization
    phase4 = re.sub("(\w)([A-Z])", r"\1 \2", phase3)
    # Make all characters lower case
    phase5 = phase4.lower()
    # Recombining the retweet indicator
    phase6 = re.sub("r t ", " rt ", phase5)
    # Removing stop words - very common, useless words ('the', 'a', etc)
    phase7 = gensim.parsing.preprocessing.remove_stopwords(phase6)
    return(phase7)


# Cleaning all the tweets
clean_tweet_array = [clean_tweet(tweet) for tweet in tweet_array]
# Tokenizing all the tweets
tweet_tokens = [clean_tweet.split() for clean_tweet in clean_tweet_array]


dictionary = corpora.Dictionary(tweet_tokens)
reloaded_dict = dictionary.load(dictionary_path)



# Converting list of tweets (corpus) into a tweet term matrix using dictionary prepared above.
tweet_term_matrix = [reloaded_dict.doc2bow(tweet) for tweet in tweet_tokens]



# To reduce bottlenecks in the parallelized LDA we need a sharded (parallelized) corpus
ShardedCorpus.serialize(sharded_corpus_dest, tweet_term_matrix,
                        shardsize = 2048, dim = len(dictionary),
                        sparse_serialization = True, sparse_retrieval = True)
sharded_corpus = ShardedCorpus.load(sharded_corpus_dest)


Lda = gensim.models.ldamulticore.LdaMulticore
# Reloadeding the Multi-CPU LDA model, requiring user input
ldamodel = Lda.load(lda_model_path)