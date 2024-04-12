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


# Defining function to get consistent LDA topic arrays
def make_lda_consistent(lda_output):
    current_lda_scores = []
    topics_considered = []
    for i in range(len(lda_output)):
        topics_considered.append(lda_output[i][0])
    for topics in range(50):
        if (topics in topics_considered):
            topic_index = topics_considered.index(topics)
            current_lda_scores.append(lda_output[topic_index][1])
        else:
            current_lda_scores.append(0)
    current_lda_scores_array = np.asarray(current_lda_scores)
    return(current_lda_scores_array)

tweets_df  = pd.DataFrame({'name':user_list,
                           'tweet':tweet_list,
                           'tweet_num':tweet_num_list})
tweets_df = tweets_df.pivot(index = 'name',
                            columns = 'tweet_num',
                            values = 'tweet')
tweets_df = tweets_df.drop(columns = 200)
tweets_df = tweets_df.reset_index(drop = True)
tweets_df['user'] = np.unique(user_list)

def prep_new(tweet, dict_to_use):
    phase1 = clean_tweet(tweet).split()
    phase2 = dict_to_use.doc2bow(phase1)
    return(phase2)

# The loop which....
for row in range(len(tweets_df)):
    # Gettings topic scores for each tweet
    for tweet_num in range(200):
        current_tweet = tweets_df[tweet_num][row]
        if (len(current_tweet) > 0):
            current_lda_array = ldamodel[(prep_new(current_tweet,
                                                   reloaded_dict))]
            formatted_lda_array = make_lda_consistent(current_lda_array)
            formatted_lda_array = formatted_lda_array.reshape(50, 1)
            if (tweet_num == 0):
                user_lda_matrix = formatted_lda_array
            else:
                user_lda_matrix = np.append(user_lda_matrix, formatted_lda_array, axis = 1)
    # Getting topic variance / homogeneity - determined across all tweets
    average_lda_array = np.mean(user_lda_matrix, axis = 1)
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
        actual_topic_num = sorted_lda_topics[-1*topic_index]
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

lda_df.to_csv(output_csv)