import os
import string
import re
import urllib
import zipfile
from operator import itemgetter
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
# data science
import pandas as pd
import numpy as np
# nlp
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
# machine learning
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler
# from livelossplot import PlotLosses
# earthpy
# plotting
from joblib import load

PROJ_PATH = os.path.dirname(__file__)
DATA_DIR = os.path.join(PROJ_PATH, 'data')
MODEL_DIR = os.path.join(PROJ_PATH, 'model_inputs')


def binary_regex_indicator(search_string, column, df):
    """
    Returns a list with 0's and 1's that indicate
        whether 'search_string' occurs in 'column' from
        a dataframe
        """
    res = [1 * bool(re.search(search_string, i, re.IGNORECASE))
           for i in df[column].values]
    return res


def image_loader(image_name, loader):
    image = Image.open(image_name)
    image = Variable(loader(image), requires_grad=False)
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)
    return image


def get_images_and_labels(df, loader, pics_dir):
    """Based on properly indexed-rows in a dataframe, a function that pulls
       profile pictures memory and labels from the column"""

    # Creating an empty images set that we'll insert the images into
    df_len = len(df)
    print(df_len)
    images = torch.zeros((df_len, 3, 48, 48))
    print(images.shape)

    # Since we're using random indices in the loop, using a growing count
    # For proper insertion index
    count = 0

    # Getting images - lot of QA required
    for file_index in df.index:

        jpg_file = pics_dir + str(file_index - len(df)) + ".jpg"

        # Assigning a black-image if the image is not available
        try:
            current_image = image_loader(jpg_file, loader)
        except FileNotFoundError as err:
            current_image = Variable(torch.zeros(1, 3, 48, 48))

        # If, for whatever reason, the image isn't a 48x48 square
        # This works because, for whatever reason, they're always bigger than 48
        # if not 48
        if (current_image.size()[2:4] != torch.Size([48, 48])):
            current_image = current_image[:, :, :48, :48]

        # Making gray-scale images 3-channel'd
        # It stays gray-scale but it can be handled like a 1x1x48x48
        if (current_image.size() == torch.Size([1, 1, 48, 48])):
            current_image = torch.cat([current_image,
                                       current_image,
                                       current_image], dim=1)

        # Some images have a saturated (all 1) 4th channel
        # It just drops the 4th channel
        if (current_image.size() == torch.Size([1, 4, 48, 48])):
            current_image = current_image[:, :3, :, :]

        # Some b&w image with a transparent background are 2-channel
        # It just uses one of the 2 channels twice, making 3
        if (current_image.size() == torch.Size([1, 2, 48, 48])):
            current_image = torch.cat([current_image,
                                       current_image[:, 0:1, :, :]],
                                      dim=1)

        # Inserting into the once-empty images set and growing the count
        images[count, :, :, :] = current_image
        count += 1

    return images


all_chars = string.ascii_letters + '_.\"~ -/!:()|$%^&*+=[]{}<>?' + string.digits
n_char = len(all_chars)


def charToIndex(char, all_chars=all_chars):
    # finds the index of char in the list of all chars
    index = all_chars.find(char)
    other_idx = len(all_chars) + 1
    if index == -1:
        # character not in list
        index = other_idx
    return index


def charToTensor(char):
    tensor = torch.zeros(1, n_char + 1)  # plus one for "other" chars
    tensor[0][charToIndex(char)] = 1
    return tensor


def nameToTensor(name):
    # assume a minibatch size of 1
    tensor = torch.zeros(len(name), 1, n_char + 2)  # plus 2 for "other" chars
    for i, char in enumerate(name):
        tensor[i][0][charToIndex(char)] = 1
    return tensor


def wordToTensor(word, model, dim=300):
    """
    Converts a word to a dense vector representation

    Catch key error exceptions and return a vector of 0's
     if word is not in vocabulary
    """
    # clean up text
    word = word.lower()
    word = re.sub('#', '', word)
    word = re.sub('@', '', word)
    try:
        np_array = model[word]
    except KeyError:  ########## GET RID OF THIS EVENTUALLY
        np_array = np.zeros(dim)
    tensor = torch.from_numpy(np_array)
    return tensor


def descToTensor(description, model, dim=300):
    """
    Converts a user description to tensor
        w/ dimensions (n_word X 1 (minibatch) X embedding_dim)
    """
    description_split = description.split()
    n_words = len(description_split)
    if n_words == 0:
        # no description, return zeros
        return torch.zeros(1, 1, dim)
    tensor = torch.zeros(n_words, 1, dim)
    for i, word in enumerate(description_split):
        tensor[i][0] = wordToTensor(word, model)
    return tensor


def pad_text_sequences(df, type_of_text, model):
    """PyTorch really doesn't like having a batch of different sizes, and since
       this is still early-prototyping, we're going to bad those different sizes
       with zeros to a common size (the max). In the future, I may implement that
       miserable pseudopadding that knows to stop at the actual padding"""

    text_list = []

    for index in df.index:
        if type_of_text in ['screen_name', 'u_name']:
            x_text = Variable(nameToTensor(df[type_of_text][index]),
                              requires_grad=False)
            text_list.append(x_text)
        elif type_of_text == 'u_description':
            x_text = Variable(descToTensor(df[type_of_text][index], model),
                              requires_grad=False)
            text_list.append(x_text)
        else:
            print('Error: False argument, must be either (1) u_screen_name, (2) u_name, or (3) u_description')

    seq_lengths = [element.shape[0] for element in text_list]

    real_and_sorted_indices = np.flip(np.argsort(np.asarray(seq_lengths)),
                                      axis=0)

    sorted_sequences = itemgetter(*real_and_sorted_indices)(text_list)

    padded_sorted_sequences = torch.nn.utils.rnn.pad_sequence(sorted_sequences).squeeze()

    padded_unsorted_sequences = torch.zeros_like(padded_sorted_sequences)

    for sorted_index in range(padded_sorted_sequences.shape[1]):
        padded_unsorted_sequences[:, real_and_sorted_indices[sorted_index], :] = padded_sorted_sequences[:,
                                                                                 sorted_index, :]

    return (padded_unsorted_sequences)


def get_numeric(df, data_type, standardizer, need_retweet_counts, retweet_counts):
    """A function to extract the numeric features from a dataframe and
       center and scale them for proper optimization"""

    # Don't remember why I did this
    numeric_features = df.copy()

    # Defining the columns of interest
    numeric_columns = ['u_followers_count', 'following',
                       'max_retweets', 'avg_retweets',
                       'max_favorites', 'avg_favorites',
                       'u_statuses_count', 'account_age',
                       'approx_entropy_r100', 'approx_entropy_r1000',
                       'approx_entropy_r2000', 'approx_entropy_r500',
                       'approx_entropy_r5000', 'are_retweets_percentage',
                       'avg_tweets_day',
                       'default_prof_image', 'default_theme_background',
                       'entropy', 'favorites_by_user',
                       'favorites_by_user_per_day',
                       'geoenabled',
                       'get_retweeted_percentage', 'listed_count',
                       'max_tweet_delta', 'max_tweets_day',
                       'max_tweets_hour', 'mean_tweet_delta',
                       'min_tweet_delta', 'std_tweet_delta',
                       'topic_variance',
                       'name_has_fire', 'name_has_gov',
                       'name_has_news', 'name_has_firefighter',
                       'name_has_emergency', 'name_has_wildland',
                       'name_has_wildfire', 'name_has_county',
                       'name_has_disaster', 'name_has_management',
                       'name_has_paramedic', 'name_has_right',
                       'name_has_maga', 'name_has_journalist',
                       'name_has_reporter', 'name_has_editor',
                       'name_has_photographer', 'name_has_newspaper',
                       'name_has_producer', 'name_has_anchor',
                       'name_has_photojournalist', 'name_has_tv',
                       'name_has_host', 'name_has_fm',
                       'name_has_morning', 'name_has_media',
                       'name_has_jobs', 'name_has_careers',
                       'name_has_job', 'name_has_career',
                       'name_has_romance', 'name_has_captain',
                       'name_has_firefighters', 'name_has_official',
                       'name_has_operations', 'name_has_prevention',
                       'name_has_government', 'name_has_responder',
                       'name_has_housing', 'name_has_station',
                       'name_has_correspondent', 'name_has_jewelry',
                       'name_has_trends', 'name_has_pio',
                       'name_has_ic', 'name_has_eoc',
                       'name_has_office', 'name_has_bureau',
                       'name_has_police', 'name_has_pd',
                       'name_has_department', 'name_has_city',
                       'name_has_state', 'name_has_mayor',
                       'name_has_governor', 'name_has_vost',
                       'name_has_smem', 'name_has_trump',
                       'screen_name_has_fire', 'screen_name_has_gov',
                       'screen_name_has_news', 'screen_name_has_firefighter',
                       'screen_name_has_wildland', 'screen_name_has_wildfire',
                       'screen_name_has_county', 'screen_name_has_disaster',
                       'screen_name_has_management', 'screen_name_has_paramedic',
                       'screen_name_has_right', 'screen_name_has_maga',
                       'screen_name_has_journalist', 'screen_name_has_reporter',
                       'screen_name_has_editor', 'screen_name_has_photographer',
                       'screen_name_has_newspaper', 'screen_name_has_producer',
                       'screen_name_has_anchor', 'screen_name_has_photojournalist',
                       'screen_name_has_tv', 'screen_name_has_host',
                       'screen_name_has_fm', 'screen_name_has_morning',
                       'screen_name_has_media', 'screen_name_has_jobs',
                       'screen_name_has_careers', 'screen_name_has_job',
                       'screen_name_has_career', 'screen_name_has_romance',
                       'screen_name_has_captain', 'screen_name_has_firefighters',
                       'screen_name_has_official', 'screen_name_has_operations',
                       'screen_name_has_prevention', 'screen_name_has_government',
                       'screen_name_has_responder', 'screen_name_has_housing',
                       'screen_name_has_station', 'screen_name_has_correspondent',
                       'screen_name_has_jewelry', 'screen_name_has_trends',
                       'screen_name_has_pio', 'screen_name_has_emergency',
                       'screen_name_has_ic', 'screen_name_has_eoc',
                       'screen_name_has_office', 'screen_name_has_bureau',
                       'screen_name_has_police', 'screen_name_has_pd',
                       'screen_name_has_department', 'screen_name_has_city',
                       'screen_name_has_state', 'screen_name_has_mayor',
                       'screen_name_has_governor', 'screen_name_has_vost',
                       'screen_name_has_smem', 'screen_name_has_trump']

    # Adding the topic-model topic columns and adding them to the list
    topic_columns = [str(i) for i in range(0, 50)]
    numeric_columns.extend(topic_columns)

    # Subsetting the dataframe with those columns
    # And converting to numpy array
    numeric_features = numeric_features[numeric_columns]
    numeric_features = numeric_features.values

    if need_retweet_counts == True:
        numeric_features = np.concatenate((numeric_features, retweet_counts), axis=1)

    # Centering and scaling the data
    # If it's the train set, fitting a new data transformer and using it
    if data_type == 'train':
        standardizer = MinMaxScaler()
        scaled_numeric_features = standardizer.fit_transform(numeric_features)
    # If it's validation data, only use the predefined data transformer
    elif data_type == 'validation':
        scaled_numeric_features = standardizer.transform(numeric_features)
    else:
        print('Incorrect data_type argument, please use (1) train or (2) validation')

    # Creating a variable of specified type - FloatTensor
    scaled_numeric_features = Variable(torch.from_numpy(scaled_numeric_features).type(torch.FloatTensor),
                                       requires_grad=False)

    # Output
    # If its the train set, we also want the data transformer that we trained
    if data_type == 'train':
        return scaled_numeric_features, standardizer
    # If its the validation, we use a predefined and unchanging data transformer
    elif data_type == 'validation':
        return scaled_numeric_features
    else:
        print('Did that first error really not throw?: Incorrect data_type argument' \
              ', please use (1) train or (2) validation')


def clean_tweet(tweet, want_retweet_count):
    """Function to perform standard tweet processing. Includes removing URLs, my
       end-of-line character, punctuation, lower casing, and recombining the rt
       character. Inputs a str, outputs a str"""

    if want_retweet_count == True:
        retweet_count = tweet.count(' RT ')

    # For later - to remove punctuation
    # Setting rule to replace any punctuation chain with equal amount of white space
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Remove end-of-line character and str-ified NaN
    phase1 = re.sub('EndOfTweet', ' ', tweet)
    phase1 = re.sub('EndOfDescr', ' ', tweet)
    # Remove URLs
    phase2 = re.sub(r'http\S+', '', phase1)
    # Remove punctuation
    phase3 = phase2.translate(replace_punctuation)
    # Seperate individual characters entities based on capitalization
    phase4 = re.sub("(\w)([A-Z])", r"\1 \2", phase3)
    # Make all characters lower case
    phase5 = phase4.lower()
    # Recombining the retweet indicator
    phase6 = re.sub("r t", " rt ", phase5)
    # Removing stop words - very common, useless words ('the', 'a', etc)
    phase7 = remove_stopwords(phase6)

    if want_retweet_count == True:
        return (phase7, retweet_count)
    else:
        return (phase7)


def get_and_split_ngram_features(natural_text, set_type, vect, want_retweet_count):
    """Returns sparse matrices determined by either counts or tf-idf
       weightings. Inputs the dataframe, method, and ngram_range, then
       outputs the vectorized train and text matrices"""

    # Getting all the text
    natural_text = np.reshape(natural_text, [len(natural_text), ])

    natural_text_list = []
    retweet_count_list = []

    # Deciding if we want retweet counts
    # Then getting the text as cleaned numpy array and getting retweet counts - if wanted
    if want_retweet_count == True:
        for text in natural_text:
            natural_text_item, retweet_count = clean_tweet(text, want_retweet_count)
            natural_text_list.append(natural_text_item)
            retweet_count_list.append(retweet_count)
        natural_text = np.asarray(natural_text_list)
    else:
        natural_text = np.asarray([clean_tweet(text, want_retweet_count) for text in natural_text])

    # Defining and fitting the one-hot encodering if training
    if set_type == 'train':
        vect = CountVectorizer(binary=True)
        vect = vect.fit(natural_text)
    # Getting the encoded text if not training
    else:
        one_hot_text = vect.transform(natural_text)

    # Returning the encoder if training
    if set_type == 'train':
        return (vect)
    # Returning the text (and counts) if not training
    else:
        if want_retweet_count == True:
            return (one_hot_text, retweet_count_list)
        else:
            return (one_hot_text)


text_vectorizer_file = os.path.join(MODEL_DIR, 'text_vectorizer.joblib')
text_vectorizer = load(text_vectorizer_file)


class Ensemble(nn.Module):
    # Architecture
    def __init__(self):
        super(Ensemble, self).__init__()

        self.tweet_layer_ensemble = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 3, bias=False),
            nn.Tanh())

    # Forward pass method
    def forward(self, image, numbers, screen_name,
                user_name, description, tweets,
                quoted_tweets, quoted_descr, retweet_descr):
        tweets_output = self.tweet_layer_ensemble(tweets)

        return (tweets_output)


def main():
    number_scaler_file = os.path.join(MODEL_DIR, 'number_scaler.joblib')
    zero_to_one_scaler_file = os.path.join(MODEL_DIR, 'zero_to_one_scaler.joblib')
    ensemble_model_file_v2_1 = os.path.join(MODEL_DIR, 'trained-model-uclassv2-1.pt')
    ensemble_model_file_v2_2 = os.path.join(MODEL_DIR, 'trained-model-uclassv2-2.pt')
    output_file = os.path.join(DATA_DIR, 'test_results_tweets_unimodal.csv')

    user_tweet_df = pd.read_csv(os.path.join(DATA_DIR, 'PROTO_merged.csv'))

    user_tweet_df['u_classv2_1'] = user_tweet_df['u_classv2_1'].replace({'feedbased': 'feed based'})

    user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'em expertise': 'em'})
    user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'em related': 'em'})
    user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'polarizing': 'distribution'})

    user_tweet_df['u_name'] = user_tweet_df['u_name'].astype(str)

    user_tweet_df = user_tweet_df.assign(
        name_has_fire=binary_regex_indicator('fire', 'u_name', user_tweet_df),
        name_has_gov=binary_regex_indicator('gov', 'u_name', user_tweet_df),
        name_has_news=binary_regex_indicator('news', 'u_name', user_tweet_df),
        name_has_firefighter=binary_regex_indicator('firefighter', 'u_name', user_tweet_df),
        name_has_emergency=binary_regex_indicator('emergency', 'u_name', user_tweet_df),
        name_has_wildland=binary_regex_indicator('wildland', 'u_name', user_tweet_df),
        name_has_wildfire=binary_regex_indicator('wildfire', 'u_name', user_tweet_df),
        name_has_county=binary_regex_indicator('county', 'u_name', user_tweet_df),
        name_has_disaster=binary_regex_indicator('disaster', 'u_name', user_tweet_df),
        name_has_management=binary_regex_indicator('management', 'u_name', user_tweet_df),
        name_has_paramedic=binary_regex_indicator('paramedic', 'u_name', user_tweet_df),
        name_has_right=binary_regex_indicator('right', 'u_name', user_tweet_df),
        name_has_maga=binary_regex_indicator('maga', 'u_name', user_tweet_df),
        name_has_journalist=binary_regex_indicator('journalist', 'u_name', user_tweet_df),
        name_has_reporter=binary_regex_indicator('reporter', 'u_name', user_tweet_df),
        name_has_editor=binary_regex_indicator('editor', 'u_name', user_tweet_df),
        name_has_photographer=binary_regex_indicator('photographer', 'u_name', user_tweet_df),
        name_has_newspaper=binary_regex_indicator('newspaper', 'u_name', user_tweet_df),
        name_has_producer=binary_regex_indicator('producer', 'u_name', user_tweet_df),
        name_has_anchor=binary_regex_indicator('anchor', 'u_name', user_tweet_df),
        name_has_photojournalist=binary_regex_indicator('photojournalist', 'u_name', user_tweet_df),
        name_has_tv=binary_regex_indicator('tv', 'u_name', user_tweet_df),
        name_has_host=binary_regex_indicator('host', 'u_name', user_tweet_df),
        name_has_fm=binary_regex_indicator('fm', 'u_name', user_tweet_df),
        name_has_morning=binary_regex_indicator('morning', 'u_name', user_tweet_df),
        name_has_media=binary_regex_indicator('media', 'u_name', user_tweet_df),
        name_has_jobs=binary_regex_indicator('jobs', 'u_name', user_tweet_df),
        name_has_careers=binary_regex_indicator('careers', 'u_name', user_tweet_df),
        name_has_job=binary_regex_indicator('job', 'u_name', user_tweet_df),
        name_has_career=binary_regex_indicator('career', 'u_name', user_tweet_df),
        name_has_romance=binary_regex_indicator('romance', 'u_name', user_tweet_df),
        name_has_captain=binary_regex_indicator('captain', 'u_name', user_tweet_df),
        name_has_firefighters=binary_regex_indicator('firefighters', 'u_name', user_tweet_df),
        name_has_official=binary_regex_indicator('official', 'u_name', user_tweet_df),
        name_has_operations=binary_regex_indicator('operations', 'u_name', user_tweet_df),
        name_has_prevention=binary_regex_indicator('prevention', 'u_name', user_tweet_df),
        name_has_government=binary_regex_indicator('government', 'u_name', user_tweet_df),
        name_has_responder=binary_regex_indicator('responder', 'u_name', user_tweet_df),
        name_has_housing=binary_regex_indicator('housing', 'u_name', user_tweet_df),
        name_has_station=binary_regex_indicator('station', 'u_name', user_tweet_df),
        name_has_correspondent=binary_regex_indicator('correspondent', 'u_name', user_tweet_df),
        name_has_jewelry=binary_regex_indicator('jewelry', 'u_name', user_tweet_df),
        name_has_trends=binary_regex_indicator('trends', 'u_name', user_tweet_df),
        name_has_pio=binary_regex_indicator('pio', 'u_name', user_tweet_df),
        name_has_ic=binary_regex_indicator('ic', 'u_name', user_tweet_df),
        name_has_eoc=binary_regex_indicator('eoc', 'u_name', user_tweet_df),
        name_has_office=binary_regex_indicator('office', 'u_name', user_tweet_df),
        name_has_bureau=binary_regex_indicator('bureau', 'u_name', user_tweet_df),
        name_has_police=binary_regex_indicator('police', 'u_name', user_tweet_df),
        name_has_pd=binary_regex_indicator('pd', 'u_name', user_tweet_df),
        name_has_department=binary_regex_indicator('department', 'u_name', user_tweet_df),
        name_has_city=binary_regex_indicator('city', 'u_name', user_tweet_df),
        name_has_state=binary_regex_indicator('state', 'u_name', user_tweet_df),
        name_has_mayor=binary_regex_indicator('mayor', 'u_name', user_tweet_df),
        name_has_governor=binary_regex_indicator('governor', 'u_name', user_tweet_df),
        name_has_vost=binary_regex_indicator('vost', 'u_name', user_tweet_df),
        name_has_smem=binary_regex_indicator('smem', 'u_name', user_tweet_df),
        name_has_trump=binary_regex_indicator('trump', 'u_name', user_tweet_df),
        name_has_politics=binary_regex_indicator('politics', 'u_name', user_tweet_df),
        name_has_uniteblue=binary_regex_indicator('uniteblue', 'u_name', user_tweet_df),
        name_has_retired=binary_regex_indicator('retired', 'u_name', user_tweet_df),
        name_has_revolution=binary_regex_indicator('revolution', 'u_name', user_tweet_df),
        name_has_ftw=binary_regex_indicator('ftw', 'u_name', user_tweet_df),
        name_has_difference=binary_regex_indicator('difference', 'u_name', user_tweet_df),
        name_has_patriot=binary_regex_indicator('patriot', 'u_name', user_tweet_df),
        name_has_best=binary_regex_indicator('best', 'u_name', user_tweet_df),
        name_has_interested=binary_regex_indicator('interested', 'u_name', user_tweet_df),
        name_has_understand=binary_regex_indicator('understand', 'u_name', user_tweet_df),
        name_has_clean=binary_regex_indicator('clean', 'u_name', user_tweet_df),
        name_has_global=binary_regex_indicator('global', 'u_name', user_tweet_df),
        name_has_must=binary_regex_indicator('must', 'u_name', user_tweet_df),
        name_has_book=binary_regex_indicator('book', 'u_name', user_tweet_df),
        name_has_transportation=binary_regex_indicator('transportation', 'u_name', user_tweet_df),
        name_has_defense=binary_regex_indicator('defense', 'u_name', user_tweet_df),
        name_has_warrior=binary_regex_indicator('warrior', 'u_name', user_tweet_df),
        name_has_christian=binary_regex_indicator('christian', 'u_name', user_tweet_df),
        name_has_tweet=binary_regex_indicator('tweet', 'u_name', user_tweet_df),
        name_has_first=binary_regex_indicator('first', 'u_name', user_tweet_df),
        screen_name_has_fire=binary_regex_indicator('fire', 'u_name', user_tweet_df),
        screen_name_has_gov=binary_regex_indicator('gov', 'u_name', user_tweet_df),
        screen_name_has_news=binary_regex_indicator('news', 'u_name', user_tweet_df),
        screen_name_has_firefighter=binary_regex_indicator('firefighter', 'u_name', user_tweet_df),
        screen_name_has_emergency=binary_regex_indicator('emergency', 'u_name', user_tweet_df),
        screen_name_has_wildland=binary_regex_indicator('wildland', 'u_name', user_tweet_df),
        screen_name_has_wildfire=binary_regex_indicator('wildfire', 'u_name', user_tweet_df),
        screen_name_has_county=binary_regex_indicator('county', 'u_name', user_tweet_df),
        screen_name_has_disaster=binary_regex_indicator('disaster', 'u_name', user_tweet_df),
        screen_name_has_management=binary_regex_indicator('management', 'u_name', user_tweet_df),
        screen_name_has_paramedic=binary_regex_indicator('paramedic', 'u_name', user_tweet_df),
        screen_name_has_right=binary_regex_indicator('right', 'u_name', user_tweet_df),
        screen_name_has_maga=binary_regex_indicator('maga', 'u_name', user_tweet_df),
        screen_name_has_journalist=binary_regex_indicator('journalist', 'u_name', user_tweet_df),
        screen_name_has_reporter=binary_regex_indicator('reporter', 'u_name', user_tweet_df),
        screen_name_has_editor=binary_regex_indicator('editor', 'u_name', user_tweet_df),
        screen_name_has_photographer=binary_regex_indicator('photographer', 'u_name', user_tweet_df),
        screen_name_has_newspaper=binary_regex_indicator('newspaper', 'u_name', user_tweet_df),
        screen_name_has_producer=binary_regex_indicator('producer', 'u_name', user_tweet_df),
        screen_name_has_anchor=binary_regex_indicator('anchor', 'u_name', user_tweet_df),
        screen_name_has_photojournalist=binary_regex_indicator('photojournalist', 'u_name', user_tweet_df),
        screen_name_has_tv=binary_regex_indicator('tv', 'u_name', user_tweet_df),
        screen_name_has_host=binary_regex_indicator('host', 'u_name', user_tweet_df),
        screen_name_has_fm=binary_regex_indicator('fm', 'u_name', user_tweet_df),
        screen_name_has_morning=binary_regex_indicator('morning', 'u_name', user_tweet_df),
        screen_name_has_media=binary_regex_indicator('media', 'u_name', user_tweet_df),
        screen_name_has_jobs=binary_regex_indicator('jobs', 'u_name', user_tweet_df),
        screen_name_has_careers=binary_regex_indicator('careers', 'u_name', user_tweet_df),
        screen_name_has_job=binary_regex_indicator('job', 'u_name', user_tweet_df),
        screen_name_has_career=binary_regex_indicator('career', 'u_name', user_tweet_df),
        screen_name_has_romance=binary_regex_indicator('romance', 'u_name', user_tweet_df),
        screen_name_has_captain=binary_regex_indicator('captain', 'u_name', user_tweet_df),
        screen_name_has_firefighters=binary_regex_indicator('firefighters', 'u_name', user_tweet_df),
        screen_name_has_official=binary_regex_indicator('official', 'u_name', user_tweet_df),
        screen_name_has_operations=binary_regex_indicator('operations', 'u_name', user_tweet_df),
        screen_name_has_prevention=binary_regex_indicator('prevention', 'u_name', user_tweet_df),
        screen_name_has_government=binary_regex_indicator('government', 'u_name', user_tweet_df),
        screen_name_has_responder=binary_regex_indicator('responder', 'u_name', user_tweet_df),
        screen_name_has_housing=binary_regex_indicator('housing', 'u_name', user_tweet_df),
        screen_name_has_station=binary_regex_indicator('station', 'u_name', user_tweet_df),
        screen_name_has_correspondent=binary_regex_indicator('correspondent', 'u_name', user_tweet_df),
        screen_name_has_jewelry=binary_regex_indicator('jewelry', 'u_name', user_tweet_df),
        screen_name_has_trends=binary_regex_indicator('trends', 'u_name', user_tweet_df),
        screen_name_has_pio=binary_regex_indicator('pio', 'u_name', user_tweet_df),
        screen_name_has_ic=binary_regex_indicator('ic', 'u_name', user_tweet_df),
        screen_name_has_eoc=binary_regex_indicator('eoc', 'u_name', user_tweet_df),
        screen_name_has_office=binary_regex_indicator('office', 'u_name', user_tweet_df),
        screen_name_has_bureau=binary_regex_indicator('bureau', 'u_name', user_tweet_df),
        screen_name_has_police=binary_regex_indicator('police', 'u_name', user_tweet_df),
        screen_name_has_pd=binary_regex_indicator('pd', 'u_name', user_tweet_df),
        screen_name_has_department=binary_regex_indicator('department', 'u_name', user_tweet_df),
        screen_name_has_city=binary_regex_indicator('city', 'u_name', user_tweet_df),
        screen_name_has_state=binary_regex_indicator('state', 'u_name', user_tweet_df),
        screen_name_has_mayor=binary_regex_indicator('mayor', 'u_name', user_tweet_df),
        screen_name_has_governor=binary_regex_indicator('governor', 'u_name', user_tweet_df),
        screen_name_has_smem=binary_regex_indicator('smem', 'u_name', user_tweet_df),
        screen_name_has_vost=binary_regex_indicator('vost', 'u_name', user_tweet_df),
        screen_name_has_trump=binary_regex_indicator('trump', 'u_name', user_tweet_df),
        screen_name_has_politics=binary_regex_indicator('politics', 'u_name', user_tweet_df),
        screen_name_has_uniteblue=binary_regex_indicator('uniteblue', 'u_name', user_tweet_df),
        screen_name_has_retired=binary_regex_indicator('retired', 'u_name', user_tweet_df),
        screen_name_has_revolution=binary_regex_indicator('revolution', 'u_name', user_tweet_df),
        screen_name_has_ftw=binary_regex_indicator('ftw', 'u_name', user_tweet_df),
        screen_name_has_difference=binary_regex_indicator('difference', 'u_name', user_tweet_df),
        screen_name_has_patriot=binary_regex_indicator('patriot', 'u_name', user_tweet_df),
        screen_name_has_best=binary_regex_indicator('best', 'u_name', user_tweet_df),
        screen_name_has_interested=binary_regex_indicator('interested', 'u_name', user_tweet_df),
        screen_name_has_understand=binary_regex_indicator('understand', 'u_name', user_tweet_df),
        screen_name_has_clean=binary_regex_indicator('clean', 'u_name', user_tweet_df),
        screen_name_has_global=binary_regex_indicator('global', 'u_name', user_tweet_df),
        screen_name_has_must=binary_regex_indicator('must', 'u_name', user_tweet_df),
        screen_name_has_book=binary_regex_indicator('book', 'u_name', user_tweet_df),
        screen_name_has_transportation=binary_regex_indicator('transporation', 'u_name', user_tweet_df),
        screen_name_has_defense=binary_regex_indicator('defense', 'u_name', user_tweet_df),
        screen_name_has_warrior=binary_regex_indicator('warrior', 'u_name', user_tweet_df),
        screen_name_has_christian=binary_regex_indicator('christian', 'u_name', user_tweet_df),
        screen_name_has_tweet=binary_regex_indicator('tweet', 'u_name', user_tweet_df),
        screen_name_has_first=binary_regex_indicator('first', 'u_name', user_tweet_df))

    numeric_columns = ['u_followers_count', 'following',
                       'max_retweets', 'avg_retweets',
                       'max_favorites', 'avg_favorites',
                       'u_statuses_count', 'account_age',
                       'approx_entropy_r100', 'approx_entropy_r1000',
                       'approx_entropy_r2000', 'approx_entropy_r500',
                       'approx_entropy_r5000', 'are_retweets_percentage',
                       'avg_tweets_day',
                       'default_prof_image',
                       'entropy',
                       'get_retweeted_percentage', 'listed_count',
                       'max_tweet_delta', 'max_tweets_day',
                       'max_tweets_hour', 'mean_tweet_delta',
                       'min_tweet_delta', 'std_tweet_delta',
                       # 'content_score', 'friend_score',
                       # 'network_score', 'sentiment_score',
                       # 'temporal_score', 'user_score',
                       # 'english_score',
                       'topic_variance',
                       'name_has_fire', 'name_has_gov',
                       'name_has_news', 'name_has_firefighter',
                       'name_has_emergency', 'name_has_wildland',
                       'name_has_wildfire', 'name_has_county',
                       'name_has_disaster', 'name_has_management',
                       'name_has_paramedic', 'name_has_right',
                       'name_has_maga', 'name_has_journalist',
                       'name_has_reporter', 'name_has_editor',
                       'name_has_photographer', 'name_has_newspaper',
                       'name_has_producer', 'name_has_anchor',
                       'name_has_photojournalist', 'name_has_tv',
                       'name_has_host', 'name_has_fm',
                       'name_has_morning', 'name_has_media',
                       'name_has_jobs', 'name_has_careers',
                       'name_has_job', 'name_has_career',
                       'name_has_romance', 'name_has_captain',
                       'name_has_firefighters', 'name_has_official',
                       'name_has_operations', 'name_has_prevention',
                       'name_has_government', 'name_has_responder',
                       'name_has_housing', 'name_has_station',
                       'name_has_correspondent', 'name_has_jewelry',
                       'name_has_trends', 'name_has_pio',
                       'name_has_ic', 'name_has_eoc',
                       'name_has_office', 'name_has_bureau',
                       'name_has_police', 'name_has_pd',
                       'name_has_department', 'name_has_city',
                       'name_has_state', 'name_has_mayor',
                       'name_has_governor', 'name_has_vost',
                       'name_has_smem', 'name_has_trump', 'name_has_politics',
                       'name_has_uniteblue', 'name_has_retired', 'name_has_revolution',
                       'name_has_ftw', 'name_has_difference', 'name_has_trends',
                       'name_has_patriot', 'name_has_best', 'name_has_interested',
                       'name_has_understand', 'name_has_clean', 'name_has_global',
                       'name_has_must', 'name_has_book', 'name_has_transportation',
                       'name_has_defense', 'name_has_warrior', 'name_has_christian',
                       'name_has_tweet', 'name_has_first',
                       'screen_name_has_fire', 'screen_name_has_gov',
                       'screen_name_has_news', 'screen_name_has_firefighter',
                       'screen_name_has_wildland', 'screen_name_has_wildfire',
                       'screen_name_has_county', 'screen_name_has_disaster',
                       'screen_name_has_management', 'screen_name_has_paramedic',
                       'screen_name_has_right', 'screen_name_has_maga',
                       'screen_name_has_journalist', 'screen_name_has_reporter',
                       'screen_name_has_editor', 'screen_name_has_photographer',
                       'screen_name_has_newspaper', 'screen_name_has_producer',
                       'screen_name_has_anchor', 'screen_name_has_photojournalist',
                       'screen_name_has_tv', 'screen_name_has_host',
                       'screen_name_has_fm', 'screen_name_has_morning',
                       'screen_name_has_media', 'screen_name_has_jobs',
                       'screen_name_has_careers', 'screen_name_has_job',
                       'screen_name_has_career', 'screen_name_has_romance',
                       'screen_name_has_captain', 'screen_name_has_firefighters',
                       'screen_name_has_official', 'screen_name_has_operations',
                       'screen_name_has_prevention', 'screen_name_has_government',
                       'screen_name_has_responder', 'screen_name_has_housing',
                       'screen_name_has_station', 'screen_name_has_correspondent',
                       'screen_name_has_jewelry', 'screen_name_has_trends',
                       'screen_name_has_pio', 'screen_name_has_emergency',
                       'screen_name_has_ic', 'screen_name_has_eoc',
                       'screen_name_has_office', 'screen_name_has_bureau',
                       'screen_name_has_police', 'screen_name_has_pd',
                       'screen_name_has_department', 'screen_name_has_city',
                       'screen_name_has_state', 'screen_name_has_mayor',
                       'screen_name_has_governor', 'screen_name_has_vost',
                       'screen_name_has_smem', 'screen_name_has_trump', 'screen_name_has_politics',
                       'screen_name_has_uniteblue', 'screen_name_has_retired', 'screen_name_has_revolution',
                       'screen_name_has_ftw', 'screen_name_has_difference', 'screen_name_has_trends',
                       'screen_name_has_patriot', 'screen_name_has_best', 'screen_name_has_interested',
                       'screen_name_has_understand', 'screen_name_has_clean', 'screen_name_has_global',
                       'screen_name_has_must', 'screen_name_has_book', 'screen_name_has_transportation',
                       'screen_name_has_defense', 'screen_name_has_warrior', 'screen_name_has_christian',
                       'screen_name_has_tweet', 'screen_name_has_first']

    user_tweet_df = user_tweet_df.dropna(subset=numeric_columns)

    imsize = 48

    loader = transforms.Compose([
        # transforms.Scale(imsize),  # scale imported image (deprecated LAS 1/10/2023)
        transforms.Resize(imsize),
        transforms.ToTensor()])  # transform it into a torch tensor

    fasttext_prefix = os.path.join(DATA_DIR, 'bin')
    fasttext_bin_path = os.path.join(fasttext_prefix, 'wiki-news-300d-1M.vec')

    # create bin directory if necessary
    if not os.path.isdir(fasttext_prefix):
        os.mkdir(fasttext_prefix)
    assert os.path.isdir(fasttext_prefix)

    # download pre-trained vector if necessary
    # link from https://fasttext.cc/docs/en/supervised-models.html
    if not os.path.isfile(fasttext_bin_path):
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip',
                                   fasttext_bin_path + '.zip')
        zip_ref = zipfile.ZipFile(fasttext_bin_path + '.zip', 'r')
        zip_ref.extractall(fasttext_prefix)
        zip_ref.close()
    assert os.path.isfile(fasttext_bin_path)

    model = KeyedVectors.load_word2vec_format(fasttext_bin_path)

    user_tweet_df = user_tweet_df.assign(has_description=1. - 1 * pd.isnull(user_tweet_df['u_description']))

    user_tweet_df.u_description = user_tweet_df.u_description.fillna('')



    validation_tweet_word_counts, validation_retweet_counts = get_and_split_ngram_features(
        user_tweet_df['condensed_tweets'].fillna('').values,
        'not_train', text_vectorizer, True)
    validation_quoted_word_counts = get_and_split_ngram_features(user_tweet_df['quoted_tweets'].fillna('').values,
                                                                 'not_train', text_vectorizer, False)
    validation_retweeted_descr_counts = get_and_split_ngram_features(user_tweet_df['retweeted_descr'].fillna('').values,
                                                                     'not_train', text_vectorizer, False)
    validation_quoted_descr_counts = get_and_split_ngram_features(user_tweet_df['quoted_descr'].fillna('').values,
                                                                  'not_train', text_vectorizer, False)

    validation_tweet_word_counts = Variable(
        torch.from_numpy(np.asarray(validation_tweet_word_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
    validation_quoted_word_counts = Variable(
        torch.from_numpy(np.asarray(validation_quoted_word_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
    validation_retweeted_descr_counts = Variable(
        torch.from_numpy(np.asarray(validation_retweeted_descr_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
    validation_quoted_descr_counts = Variable(
        torch.from_numpy(np.asarray(validation_quoted_descr_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)

    validation_retweet_counts = np.asarray(validation_retweet_counts).reshape(len(user_tweet_df), 1)



    number_scaler = load(number_scaler_file)

    #validation_images = get_images_and_labels(user_tweet_df, loader, pics_dir)

    ### NEED TO HAVE NUMber SCALAR sAVED + LOADED
    validation_numbers = get_numeric(user_tweet_df, 'validation', number_scaler, True, validation_retweet_counts)

    user_tweet_df = user_tweet_df.reset_index(drop=True)

    validation_screennames = pad_text_sequences(user_tweet_df, 'screen_name', model)

    validation_u_names = pad_text_sequences(user_tweet_df, 'u_name', model)

    validation_u_descriptions = pad_text_sequences(user_tweet_df, 'u_description', model)

    zero_to_one_scaler = load(zero_to_one_scaler_file)

    validation_u_descriptions_reshape = validation_u_descriptions.view(-1, 300)
    validation_descriptions_as_numpy = validation_u_descriptions_reshape.data.numpy()

    # Scaling and reshaping the validation set
    validation_u_descriptions_scaled = zero_to_one_scaler.transform(validation_descriptions_as_numpy)
    validation_u_descriptions_scaled = Variable(torch.from_numpy(validation_u_descriptions_scaled),
                                                requires_grad=False)
    validation_u_descriptions_scaled = validation_u_descriptions_scaled.view(validation_u_descriptions.shape[0],
                                                                             validation_u_descriptions.shape[1],
                                                                             300)

    class validation_Dataset(torch.utils.data.Dataset):
        'Characterizes a dataset for PyTorch'

        def __init__(self, list_IDs):
            'Initialization'
            self.list_IDs = list_IDs

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.list_IDs)

        def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            ID = self.list_IDs[index]

            # Getting all the different features
            # x_image = validation_images[ID]
            x_numeric = validation_numbers[ID]
            x_sn = validation_screennames[:, ID, :]
            x_un = validation_u_names[:, ID, :]
            x_descr = validation_u_descriptions[:, ID, :]
            x_tweet_words = validation_tweet_word_counts[ID]
            x_quoted_tweet_words = validation_quoted_word_counts[ID]
            x_quoted_descr_words = validation_quoted_descr_counts[ID]
            x_retweet_descr_words = validation_retweeted_descr_counts[ID]

            # Storing user name
            names = user_tweet_df['screen_name'][ID]

            # TODO: Replace x_image
            return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                    x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                    ID, names)

    validation_set = validation_Dataset(range(len(user_tweet_df)))

    batch_size = 512

    validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ensemble_model_v2_1 = torch.load(ensemble_model_file_v2_1)
    ensemble_model_v2_2 = torch.load(ensemble_model_file_v2_2)

    print('loaded models')

    predicted_label_list_y = []
    true_label_list_y = []
    ID_list = []
    name_list = []

    loader = validation_loader
    predictions_matrix_y = np.zeros((len(user_tweet_df), 3))
    ordered_labels_y = np.zeros((len(user_tweet_df), 1))

    for data in loader:

        # Get the data from the loader
        two, three, four, five, six, seven, eight, nine, ID, names = data

        # Move it to the GPUs
        # one = one.to(device)
        two = two.to(device)
        three = three.to(device)
        four = four.to(device)
        five = five.to(device)
        six = six.to(device)
        seven = seven.to(device)
        eight = eight.to(device)
        nine = nine.to(device)

        # Run it through the model
        prediction = ensemble_model_v2_1(two, three,
                                         four, five, six,
                                         seven, eight, nine)

        # Convert these probabilities to the label prediction
        prediction_array = prediction.cpu().data.numpy()
        predicted_label = np.argmax(prediction_array, axis=1).tolist()
        predicted_label_list_y.extend(predicted_label)

        # Storing IDs for data set inspection
        ID_list_temp = ID.cpu().data.numpy().tolist()
        ID_list.extend(ID_list_temp)

        # Storing names
        name_list.extend(names)

        id_count = 0
        for i in ID_list_temp:
            predictions_matrix_y[i, :] = prediction_array[id_count]
            ordered_labels_y[i, :] = predicted_label[id_count]
            id_count += 1

    hat_df1 = pd.DataFrame({'predicted_y1_redone': predicted_label_list_y})
    hat_df1['screen_name'] = name_list
    hat_df1['predicted_y1_redone'] = hat_df1['predicted_y1_redone'].replace({0: 'feed based',
                                                                             1: 'individual',
                                                                             2: 'organization'})
    hat_df1.head()

    predicted_label_list_y = []
    true_label_list_y = []
    ID_list = []
    name_list = []

    loader = validation_loader
    predictions_matrix_y = np.zeros((len(user_tweet_df), 5))
    ordered_labels_y = np.zeros((len(user_tweet_df), 1))

    for data in loader:

        # Get the data from the loader
        one, two, three, four, five, six, seven, eight, nine, ID, names = data

        # Move it to the GPUs
        one = one.to(device)
        two = two.to(device)
        three = three.to(device)
        four = four.to(device)
        five = five.to(device)
        six = six.to(device)
        seven = seven.to(device)
        eight = eight.to(device)
        nine = nine.to(device)

        # Run it through the model
        prediction = ensemble_model_v2_2(one, two, three,
                                         four, five, six,
                                         seven, eight, nine)

        # Convert these probabilities to the label prediction
        prediction_array = prediction.cpu().data.numpy()
        predicted_label = np.argmax(prediction_array, axis=1).tolist()
        predicted_label_list_y.extend(predicted_label)

        # Storing IDs for data set inspection
        ID_list_temp = ID.cpu().data.numpy().tolist()
        ID_list.extend(ID_list_temp)

        # Storing names
        name_list.extend(names)

        id_count = 0
        for i in ID_list_temp:
            predictions_matrix_y[i, :] = prediction_array[id_count]
            ordered_labels_y[i, :] = predicted_label[id_count]
            id_count += 1


if __name__ == '__main__':
    main()