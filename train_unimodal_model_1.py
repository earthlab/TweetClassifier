import pandas as pd
import numpy as np
import os
import urllib
import zipfile
from gensim.models import KeyedVectors
from gensim.parsing.preprocessing import remove_stopwords
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
from sklearn import metrics
from livelossplot import PlotLosses
from sklearn.utils import resample
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from operator import itemgetter
import time
from sklearn.preprocessing import label_binarize
from itertools import cycle
from scipy import interp
import warnings
from joblib import dump, load

warnings.filterwarnings("ignore", category=UserWarning)

PROJ_DIR = os.path.dirname(__file__)


class TrainDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, train_labels_to_specify, train_numbers, train_screennames, train_u_names,
                 train_u_descriptions, train_tweet_word_counts, train_quoted_word_counts, train_quoted_descr_counts,
                 train_retweeted_descr_counts):
        'Initialization'
        self.list_IDs = list_IDs
        self.train_labels = train_labels_to_specify
        self.train_numbers = train_numbers
        self.train_screennames = train_screennames
        self.train_u_names = train_u_names
        self.train_u_descriptions = train_u_descriptions
        self.train_tweet_word_counts = train_tweet_word_counts
        self.train_quoted_word_counts = train_quoted_word_counts
        self.train_quoted_descr_counts = train_quoted_descr_counts
        self.train_retweeted_descr_counts = train_retweeted_descr_counts

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Getting all the different features
        x_numeric = self.train_numbers[ID]
        x_sn = self.train_screennames[:, ID, :]
        x_un = self.train_u_names[:, ID, :]
        x_descr = self.train_u_descriptions[:, ID, :]
        x_tweet_words = self.train_tweet_word_counts[ID]
        x_quoted_tweet_words = self.train_quoted_word_counts[ID]
        x_quoted_descr_words = self.train_quoted_descr_counts[ID]
        x_retweet_descr_words = self.train_retweeted_descr_counts[ID]

        # Getting the labels
        y = self.train_labels[ID]

        return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                y, ID)


class ValidationDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list_IDs, validation_labels_to_specify, validation_numbers, validation_screennames,
                 validation_u_names, validation_u_description, validation_tweet_word_counts,
                 validation_quoted_word_counts, validation_quoted_descr_counts, validation_retweeted_descr_count):
        'Initialization'
        self.list_IDs = list_IDs
        self.validation_labels = validation_labels_to_specify
        self.validation_labels_to_specify = validation_labels_to_specify
        self.validation_numbers = validation_numbers
        self.validation_screennames = validation_screennames
        self.validation_u_names = validation_u_names
        self.validation_u_descriptions = validation_u_description
        self.validation_tweet_word_counts = validation_tweet_word_counts
        self.validation_quoted_word_counts = validation_quoted_word_counts
        self.validation_quoted_descr_counts = validation_quoted_descr_counts
        self.validation_retweeted_descr_counts = validation_retweeted_descr_count

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        x_numeric = self.validation_numbers[ID]
        x_sn = self.validation_screennames[:, ID, :]
        x_un = self.validation_u_names[:, ID, :]
        x_descr = self.validation_u_descriptions[:, ID, :]
        x_tweet_words = self.validation_tweet_word_counts[ID]
        x_quoted_tweet_words = self.validation_quoted_word_counts[ID]
        x_quoted_descr_words = self.validation_quoted_descr_counts[ID]
        x_retweet_descr_words = self.validation_retweeted_descr_counts[ID]

        # Getting the labels
        y = self.validation_labels[ID]

        return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                y, ID)


class Ensemble(nn.Module):
    # Architecture
    def __init__(self, numeric_features, text_vectorizer):
        super(Ensemble, self).__init__()

        print(numeric_features.shape, 'numeric_shape')

        # Part2. Numerical component
        self.numeric_layer = nn.Sequential(
            nn.Linear(numeric_features.shape[1], 3, bias=False),
            nn.Tanh())

        # Part3. User name component
        #        PyTorch LSTM already has a tanh
        self.uname_lstm_layer = nn.LSTM(90, 3, num_layers=1, bias=False)

        # Part4. Screen name component
        self.screenname_lstm_layer = nn.LSTM(90, 3, num_layers=1, bias=False)

        # Part5. Description component
        self.descr_lstm_layer = nn.LSTM(300, 3, num_layers=1, bias=False)

        # Part6. User (re)tweet component - for ensembling
        self.tweet_layer_ensemble = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 3, bias=False),
            nn.Tanh())

        # Part7. Quoted tweet component
        self.quoted_tweet_layer = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 3, bias=False),
            nn.Tanh())

        # Part8. Quoted description component
        self.quoted_descr_layer = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 3, bias=False),
            nn.Tanh())

        # Part9. Retweeted description component
        self.retweet_descr_layer = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 3, bias=False),
            nn.Tanh())

        # Part10. Ensemble
        self.ensemble_layer = nn.Sequential(
            nn.Linear(24, 3, bias=False),
            nn.Tanh())

        # Part11. User (re)tweet component - for "skip connection"
        self.tweet_layer_skip = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 3, bias=True),
            nn.Sigmoid())

        # Forward pass method
        # Forward pass method

    def forward(self, numbers, screen_name,
                user_name, description, tweets,
                quoted_tweets, quoted_descr, retweet_descr):
        # Part2. Numeric forward pass for ensembling
        numeric_layer_output_ensemble = self.numeric_layer(numbers)

        # Part3. User name through the char LSTM
        user_name_output, _ = self.uname_lstm_layer(user_name)
        # Only take the last cell state
        user_name_output = user_name_output[:, -1, :]

        # Part4. Screen name through the char LSTM
        screen_name_output, _ = self.screenname_lstm_layer(screen_name)
        # Only take the last cell state
        screen_name_output = screen_name_output[:, -1, :]

        # Part5. Description forward pass
        # Description embedding through the word LSTM
        description_output, _ = self.descr_lstm_layer(description)
        # Only take the last cell state
        description_output = description_output[:, -1, :]

        # Part6. Tweets forward pass
        tweets_output = self.tweet_layer_ensemble(tweets)

        # Part7. Quoted tweets forward pass
        quoted_tweet_output = self.quoted_tweet_layer(quoted_tweets)

        # Part8. Quoted descriptions forward pass
        quoted_descr_output = self.quoted_descr_layer(quoted_descr)

        # Part9. Retweeted descriptions forward pass
        retweet_descr_output = self.retweet_descr_layer(retweet_descr)

        # Part10. Ensemble
        # Concatenating the submodel outputs together
        combined_input = torch.cat((numeric_layer_output_ensemble,
                                    screen_name_output,
                                    user_name_output,
                                    description_output,
                                    tweets_output,
                                    quoted_tweet_output,
                                    quoted_descr_output,
                                    retweet_descr_output),
                                   dim=1)
        # Forward-passing them through the simple FC layer
        supplemental_output = self.ensemble_layer(combined_input)

        # Part11. Skipped forward pass of the numerical component
        primary_output = self.tweet_layer_skip(tweets)

        # Adding that 6-submodel-derived output to the primary
        # tweet submodel output
        final_output = torch.add(primary_output, supplemental_output)
        return final_output


all_chars = string.ascii_letters + '_.\"~ -/!:()|$%^&*+=[]{}<>?' + string.digits
n_char = len(all_chars)

fasttext_prefix = '../bin'
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


def wordToTensor(word, dim=300):
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


def descToTensor(description, dim=300):
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
        tensor[i][0] = wordToTensor(word)
    return tensor


def produce_eval(ensemble_model, data_set, train_loader, train_df, validation_loader, cv_df, device):
    predicted_label_list_y = []
    true_label_list_y = []
    ID_list = []

    if data_set == 'train':
        loader = train_loader
        predictions_matrix_y = np.zeros((len(train_df), 3))
        ordered_labels_y = np.zeros((len(train_df), 1))
    elif data_set == 'cv':
        loader = validation_loader
        predictions_matrix_y = np.zeros((len(cv_df), 3))
        ordered_labels_y = np.zeros((len(cv_df), 1))
    else:
        print('Invalid data set. Please use (1) train or (2) cv')

    for data in loader:

        # Get the data from the loader
        one, two, three, four, five, six, seven, eight, nine, ID = data

        # Move it to the GPUs
        one = one.to(device)
        two = two.to(device)
        three = three.to(device)
        four = four.to(device)
        five = five.to(device)
        six = six.to(device)
        seven = seven.to(device)
        eight = eight.to(device)

        # Getting labels
        true_y = nine.to(device)

        # Run it through the model
        prediction = ensemble_model(one, two, three,
                                    four, five, six,
                                    seven, eight)

        # Convert these probabilities to the label prediction
        prediction_array = prediction.cpu().data.numpy()
        predicted_label = np.argmax(prediction_array, axis=1).tolist()
        predicted_label_list_y.extend(predicted_label)

        # Storing IDs for data set inspection
        ID_list_temp = ID.cpu().data.numpy().tolist()
        ID_list.extend(ID_list_temp)

        # Get these-shuffled true labels for evaluation
        true_label_list_y.extend(true_y.cpu().data.numpy().tolist())

        id_count = 0
        for i in ID_list_temp:
            predictions_matrix_y[i, :] = prediction_array[id_count]
            ordered_labels_y[i, :] = predicted_label[id_count]
            id_count += 1

    print('##### Evaluation for the... ' + data_set + ' set:')
    print('### Y1')
    print('\nConfusion Matrix')
    confusion_matrix = metrics.confusion_matrix(true_label_list_y,
                                                predicted_label_list_y)
    print('\nAccuracy')
    accuracy = metrics.accuracy_score(true_label_list_y,
                                      predicted_label_list_y)

    print('\nClass-specifics')
    class_specifics = metrics.classification_report(true_label_list_y,
                                                    predicted_label_list_y)

    return (predicted_label_list_y, ID_list, predictions_matrix_y, ordered_labels_y, confusion_matrix, accuracy,
            class_specifics)


def pad_text_sequences(df, type_of_text):
    """PyTorch really doesn't like having a batch of different sizes, and since
       this is still early-prototyping, we're going to bad those different sizes
       with zeros to a common size (the max). In the future, I may implement that
       miserable pseudopadding that knows to stop at the actual padding"""

    text_list = []

    for index in df.index:
        if type_of_text in ['screen_name', 'u_name']:
            x_text = Variable(nameToTensor(df[type_of_text][index]),
                              requires_grad=False)
        elif type_of_text == 'u_description':
            x_text = Variable(descToTensor(df[type_of_text][index]),
                              requires_grad=False)
        else:
            print('Error: False argument, must be either (1) u_screen_name, (2) u_name, or (3) u_description')
        text_list.append(x_text)

    seq_lengths = [element.shape[0] for element in text_list]

    real_and_sorted_indices = np.flip(np.argsort(np.asarray(seq_lengths)),
                                      axis=0)

    sorted_sequences = itemgetter(*real_and_sorted_indices)(text_list)

    padded_sorted_sequences = torch.nn.utils.rnn.pad_sequence(sorted_sequences).squeeze()

    padded_unsorted_sequences = torch.zeros_like(padded_sorted_sequences)

    for sorted_index in range(padded_sorted_sequences.shape[1]):
        padded_unsorted_sequences[:, real_and_sorted_indices[sorted_index], :] = padded_sorted_sequences[:,
                                                                                 sorted_index, :]

    return padded_unsorted_sequences


def grid_search(l2_value, numeric_features, text_vectorizer, train_loader, train_df, validation_loader, validation_df,
                device):
    # Initialize the model and move it to GPU
    ensemble = Ensemble(numeric_features, text_vectorizer)
    ensemble = nn.DataParallel(ensemble)
    ensemble.to(device)

    # Train the model w/ specified hyperparameter
    trained_model, train_loss, cv_loss = train_model(ensemble_model=ensemble,
                                                     l2_value=l2_value,
                                                     epochs=num_epochs, train_loader=train_loader,
                                                     validation_loader=validation_loader)

    # Produce evaluations of the trained model
    train_predictions_y, train_IDs, train_scores_y, train_ordered_labels_y, train_confusion_matrix, train_accuracy, train_class_specifics = produce_eval(
        trained_model, 'train', train_loader, train_df, validation_loader, validation_df, device)

    (validation_predictions_y, validation_IDs, validation_scores_y, validation_ordered_labels_y,
     validation_confusion_matrix, validation_accuracy, validation_class_specifics) = produce_eval(
        trained_model, 'cv', train_loader, train_df, validation_loader, validation_df, device)

    # Store the model and predictions
    return (trained_model, train_predictions_y, validation_predictions_y,
            train_IDs, validation_IDs, train_scores_y, validation_scores_y,
            train_ordered_labels_y, validation_ordered_labels_y, train_confusion_matrix, train_accuracy,
            train_class_specifics, validation_confusion_matrix, validation_accuracy, validation_class_specifics)


def get_test_train_split():
    # Read the CSV file into a dataframe
    user_tweet_df = pd.read_csv(os.path.join(PROJ_DIR, 'data', 'PROTO_merged_lda.csv'))

    # Drop any unnecessary columns
    # user_tweet_df = user_tweet_df.drop('unnecessary_column_name', axis=1)

    # Define the train/test split ratio
    test_size = 0.2  # 20% of the data will be used for testing

    # Split the dataframe into train and test sets
    user_tweet_df, validation_user_tweet_df = train_test_split(user_tweet_df, test_size=test_size, random_state=42)

    # Optionally, you can reset the index of the resulting dataframes
    user_tweet_df.reset_index(drop=True, inplace=True)
    validation_user_tweet_df.reset_index(drop=True, inplace=True)

    user_tweet_df['set'] = np.repeat('train', len(user_tweet_df))
    user_tweet_df = user_tweet_df.drop('Unnamed: 0', axis=1)

    validation_user_tweet_df['set'] = np.repeat('validation', len(validation_user_tweet_df))
    validation_user_tweet_df = validation_user_tweet_df.drop('Unnamed: 0', axis=1)

    for i in user_tweet_df.columns:
        assert i in validation_user_tweet_df.columns

    for j in validation_user_tweet_df.columns:
        assert j in user_tweet_df.columns

    user_tweet_df = user_tweet_df.append(validation_user_tweet_df, sort=True)
    user_tweet_df = user_tweet_df.reset_index(drop=True)

    user_tweet_df['u_classv2_1'] = user_tweet_df['u_classv2_1'].replace({'feedbased': 'feed based'})
    # user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'em expertise':'em'})
    # user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'em related':'em'})
    # user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'polarizing':'distribution'})
    # user_tweet_df = user_tweet_df[user_tweet_df['u_classv2_2'] != 'protected']

    pandas_Cat1 = pd.Categorical(user_tweet_df['u_classv2_1'])
    print(pandas_Cat1.categories)
    user_tweet_df['u_classv2_1'] = pandas_Cat1.codes
    # pandas_Cat2 = pd.Categorical(user_tweet_df['u_classv2_2'])
    # print(pandas_Cat2.categories)
    # user_tweet_df['u_classv2_2'] = pandas_Cat2.codes

    pd.set_option('display.max_colwidth', -1)
    list(user_tweet_df)

    user_tweet_df['u_name'] = user_tweet_df['u_name'].astype(str)

    def binary_regex_indicator(search_string, column, df):
        """
        Returns a list with 0's and 1's that indicate
            whether 'search_string' occurs in 'column' from
            a dataframe
            """
        res = [1 * bool(re.search(search_string, i, re.IGNORECASE))
               for i in df[column].values]
        return res

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
        name_has_interested=binary_regex_indicator('u_name', 'screen_name', user_tweet_df),
        name_has_understand=binary_regex_indicator('u_name', 'screen_name', user_tweet_df),
        name_has_clean=binary_regex_indicator('clean', 'u_name', user_tweet_df),
        name_has_global=binary_regex_indicator('global', 'u_name', user_tweet_df),
        name_has_must=binary_regex_indicator('must', 'u_name', user_tweet_df),
        name_has_book=binary_regex_indicator('book', 'u_name', user_tweet_df),
        name_has_transportation=binary_regex_indicator('u_name', 'screen_name', user_tweet_df),
        name_has_defense=binary_regex_indicator('defense', 'u_name', user_tweet_df),
        name_has_warrior=binary_regex_indicator('warrior', 'u_name', user_tweet_df),
        name_has_christian=binary_regex_indicator('christian', 'u_name', user_tweet_df),
        name_has_tweet=binary_regex_indicator('tweet', 'u_name', user_tweet_df),
        name_has_first=binary_regex_indicator('first', 'u_name', user_tweet_df),
        screen_name_has_fire=binary_regex_indicator('fire', 'screen_name', user_tweet_df),
        screen_name_has_gov=binary_regex_indicator('gov', 'screen_name', user_tweet_df),
        screen_name_has_news=binary_regex_indicator('news', 'screen_name', user_tweet_df),
        screen_name_has_firefighter=binary_regex_indicator('firefighter', 'screen_name', user_tweet_df),
        screen_name_has_emergency=binary_regex_indicator('emergency', 'screen_name', user_tweet_df),
        screen_name_has_wildland=binary_regex_indicator('wildland', 'screen_name', user_tweet_df),
        screen_name_has_wildfire=binary_regex_indicator('wildfire', 'screen_name', user_tweet_df),
        screen_name_has_county=binary_regex_indicator('county', 'screen_name', user_tweet_df),
        screen_name_has_disaster=binary_regex_indicator('disaster', 'screen_name', user_tweet_df),
        screen_name_has_management=binary_regex_indicator('management', 'screen_name', user_tweet_df),
        screen_name_has_paramedic=binary_regex_indicator('paramedic', 'screen_name', user_tweet_df),
        screen_name_has_right=binary_regex_indicator('right', 'screen_name', user_tweet_df),
        screen_name_has_maga=binary_regex_indicator('maga', 'screen_name', user_tweet_df),
        screen_name_has_journalist=binary_regex_indicator('journalist', 'screen_name', user_tweet_df),
        screen_name_has_reporter=binary_regex_indicator('reporter', 'screen_name', user_tweet_df),
        screen_name_has_editor=binary_regex_indicator('editor', 'screen_name', user_tweet_df),
        screen_name_has_photographer=binary_regex_indicator('photographer', 'screen_name', user_tweet_df),
        screen_name_has_newspaper=binary_regex_indicator('newspaper', 'screen_name', user_tweet_df),
        screen_name_has_producer=binary_regex_indicator('producer', 'screen_name', user_tweet_df),
        screen_name_has_anchor=binary_regex_indicator('anchor', 'screen_name', user_tweet_df),
        screen_name_has_photojournalist=binary_regex_indicator('photojournalist', 'screen_name', user_tweet_df),
        screen_name_has_tv=binary_regex_indicator('tv', 'screen_name', user_tweet_df),
        screen_name_has_host=binary_regex_indicator('host', 'screen_name', user_tweet_df),
        screen_name_has_fm=binary_regex_indicator('fm', 'screen_name', user_tweet_df),
        screen_name_has_morning=binary_regex_indicator('morning', 'screen_name', user_tweet_df),
        screen_name_has_media=binary_regex_indicator('media', 'screen_name', user_tweet_df),
        screen_name_has_jobs=binary_regex_indicator('jobs', 'screen_name', user_tweet_df),
        screen_name_has_careers=binary_regex_indicator('careers', 'screen_name', user_tweet_df),
        screen_name_has_job=binary_regex_indicator('job', 'screen_name', user_tweet_df),
        screen_name_has_career=binary_regex_indicator('career', 'screen_name', user_tweet_df),
        screen_name_has_romance=binary_regex_indicator('romance', 'screen_name', user_tweet_df),
        screen_name_has_captain=binary_regex_indicator('captain', 'screen_name', user_tweet_df),
        screen_name_has_firefighters=binary_regex_indicator('firefighters', 'screen_name', user_tweet_df),
        screen_name_has_official=binary_regex_indicator('official', 'screen_name', user_tweet_df),
        screen_name_has_operations=binary_regex_indicator('operations', 'screen_name', user_tweet_df),
        screen_name_has_prevention=binary_regex_indicator('prevention', 'screen_name', user_tweet_df),
        screen_name_has_government=binary_regex_indicator('government', 'screen_name', user_tweet_df),
        screen_name_has_responder=binary_regex_indicator('responder', 'screen_name', user_tweet_df),
        screen_name_has_housing=binary_regex_indicator('housing', 'screen_name', user_tweet_df),
        screen_name_has_station=binary_regex_indicator('station', 'screen_name', user_tweet_df),
        screen_name_has_correspondent=binary_regex_indicator('correspondent', 'screen_name', user_tweet_df),
        screen_name_has_jewelry=binary_regex_indicator('jewelry', 'screen_name', user_tweet_df),
        screen_name_has_trends=binary_regex_indicator('trends', 'screen_name', user_tweet_df),
        screen_name_has_pio=binary_regex_indicator('pio', 'screen_name', user_tweet_df),
        screen_name_has_ic=binary_regex_indicator('ic', 'screen_name', user_tweet_df),
        screen_name_has_eoc=binary_regex_indicator('eoc', 'screen_name', user_tweet_df),
        screen_name_has_office=binary_regex_indicator('office', 'screen_name', user_tweet_df),
        screen_name_has_bureau=binary_regex_indicator('bureau', 'screen_name', user_tweet_df),
        screen_name_has_police=binary_regex_indicator('police', 'screen_name', user_tweet_df),
        screen_name_has_pd=binary_regex_indicator('pd', 'screen_name', user_tweet_df),
        screen_name_has_department=binary_regex_indicator('department', 'screen_name', user_tweet_df),
        screen_name_has_city=binary_regex_indicator('city', 'screen_name', user_tweet_df),
        screen_name_has_state=binary_regex_indicator('state', 'screen_name', user_tweet_df),
        screen_name_has_mayor=binary_regex_indicator('mayor', 'screen_name', user_tweet_df),
        screen_name_has_governor=binary_regex_indicator('governor', 'screen_name', user_tweet_df),
        screen_name_has_smem=binary_regex_indicator('smem', 'screen_name', user_tweet_df),
        screen_name_has_vost=binary_regex_indicator('vost', 'screen_name', user_tweet_df),
        screen_name_has_trump=binary_regex_indicator('trump', 'screen_name', user_tweet_df),
        screen_name_has_politics=binary_regex_indicator('politics', 'screen_name', user_tweet_df),
        screen_name_has_uniteblue=binary_regex_indicator('uniteblue', 'screen_name', user_tweet_df),
        screen_name_has_retired=binary_regex_indicator('retired', 'screen_name', user_tweet_df),
        screen_name_has_revolution=binary_regex_indicator('revolution', 'screen_name', user_tweet_df),
        screen_name_has_ftw=binary_regex_indicator('ftw', 'screen_name', user_tweet_df),
        screen_name_has_difference=binary_regex_indicator('difference', 'screen_name', user_tweet_df),
        screen_name_has_patriot=binary_regex_indicator('patriot', 'screen_name', user_tweet_df),
        screen_name_has_best=binary_regex_indicator('best', 'screen_name', user_tweet_df),
        screen_name_has_interested=binary_regex_indicator('interested', 'screen_name', user_tweet_df),
        screen_name_has_understand=binary_regex_indicator('understand', 'screen_name', user_tweet_df),
        screen_name_has_clean=binary_regex_indicator('clean', 'screen_name', user_tweet_df),
        screen_name_has_global=binary_regex_indicator('global', 'screen_name', user_tweet_df),
        screen_name_has_must=binary_regex_indicator('must', 'screen_name', user_tweet_df),
        screen_name_has_book=binary_regex_indicator('book', 'screen_name', user_tweet_df),
        screen_name_has_transportation=binary_regex_indicator('transporation', 'screen_name', user_tweet_df),
        screen_name_has_defense=binary_regex_indicator('defense', 'screen_name', user_tweet_df),
        screen_name_has_warrior=binary_regex_indicator('warrior', 'screen_name', user_tweet_df),
        screen_name_has_christian=binary_regex_indicator('christian', 'screen_name', user_tweet_df),
        screen_name_has_tweet=binary_regex_indicator('tweet', 'screen_name', user_tweet_df),
        screen_name_has_first=binary_regex_indicator('first', 'screen_name', user_tweet_df))

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

    print(len(user_tweet_df))

    user_tweet_df = user_tweet_df.dropna(subset=numeric_columns)

    print(len(user_tweet_df))

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
            return (scaled_numeric_features, standardizer)
        # If its the validation, we use a predefined and unchanging data transformer
        elif data_type == 'validation':
            return (scaled_numeric_features)
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
            return (one_hot_text)

    user_tweet_df = user_tweet_df.assign(has_description=1. - 1 * pd.isnull(user_tweet_df['u_description']))

    user_tweet_df.u_description = user_tweet_df.u_description.fillna('')

    train_df = user_tweet_df[user_tweet_df['set'] == 'train']
    cv_df = user_tweet_df[user_tweet_df['set'] == 'validation']

    # Getting all the types of multi-word text data
    normal_tweets = train_df['condensed_tweets'].fillna('').astype(str).values
    quoted_tweets = train_df['quoted_tweets'].fillna('').astype(str).values
    normal_descr = train_df['u_description'].fillna('').astype(str).values
    retweet_descr = train_df['retweeted_descr'].fillna('').astype(str).values
    quoted_descr = train_df['quoted_descr'].fillna('').astype(str).values

    # Combining them into 1 array
    all_train_text = np.append(normal_tweets, (quoted_tweets, normal_descr, retweet_descr, quoted_descr))

    # Storing the full-vocabulary one-hot-encoder
    text_vectorizer = get_and_split_ngram_features(all_train_text, 'train', 'dontmatter', False)

    # Creating all the one-hot-encodings
    train_tweet_word_counts, train_retweet_counts = get_and_split_ngram_features(
        train_df['condensed_tweets'].fillna('').values,
        'not_train', text_vectorizer, True)
    train_quoted_word_counts = get_and_split_ngram_features(train_df['quoted_tweets'].fillna('').values,
                                                            'not_train', text_vectorizer, False)
    train_retweeted_descr_counts = get_and_split_ngram_features(train_df['retweeted_descr'].fillna('').values,
                                                                'not_train', text_vectorizer, False)
    train_quoted_descr_counts = get_and_split_ngram_features(train_df['quoted_descr'].fillna('').values,
                                                             'not_train', text_vectorizer, False)
    validation_tweet_word_counts, validation_retweet_counts = get_and_split_ngram_features(
        cv_df['condensed_tweets'].fillna('').values,
        'not_train', text_vectorizer, True)
    validation_quoted_word_counts = get_and_split_ngram_features(cv_df['quoted_tweets'].fillna('').values,
                                                                 'not_train', text_vectorizer, False)
    validation_retweeted_descr_counts = get_and_split_ngram_features(cv_df['retweeted_descr'].fillna('').values,
                                                                     'not_train', text_vectorizer, False)
    validation_quoted_descr_counts = get_and_split_ngram_features(cv_df['quoted_descr'].fillna('').values,
                                                                  'not_train', text_vectorizer, False)

    # Making the one-hot-encodings model-ready
    train_tweet_word_counts = Variable(
        torch.from_numpy(np.asarray(train_tweet_word_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
    train_quoted_word_counts = Variable(
        torch.from_numpy(np.asarray(train_quoted_word_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
    train_retweeted_descr_counts = Variable(
        torch.from_numpy(np.asarray(train_retweeted_descr_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
    train_quoted_descr_counts = Variable(
        torch.from_numpy(np.asarray(train_quoted_descr_counts.todense()).astype(float)).type(torch.FloatTensor),
        requires_grad=False)
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

    # Viewing what these input sizes will be
    len(text_vectorizer.vocabulary_)

    train_retweet_counts = np.asarray(train_retweet_counts).reshape(len(train_df), 1)
    validation_retweet_counts = np.asarray(validation_retweet_counts).reshape(len(cv_df), 1)

    random_sample = train_df.sample()
    getting_num_numeric_features, whatever = get_numeric(random_sample, 'train',
                                                         'dont matter', True,
                                                         train_retweet_counts[
                                                         random_sample.index[0]:(random_sample.index[0] + 1)])
    getting_num_numeric_features.shape[1]

    train_numbers, number_scaler = get_numeric(train_df, 'train', 'dont matter', True, train_retweet_counts)
    validation_numbers = get_numeric(cv_df, 'validation', number_scaler, True, validation_retweet_counts)

    train_df = train_df.reset_index(drop=True)

    majority_df = train_df[train_df['u_classv2_1'] == 1]

    minority_df0 = train_df[train_df['u_classv2_1'] == 0]
    minority_df2 = train_df[train_df['u_classv2_1'] == 2]

    from sklearn.utils import resample

    minority_df0_upsampled = resample(minority_df0,
                                      replace=True,
                                      n_samples=len(majority_df),
                                      random_state=123)
    minority_df2_upsampled = resample(minority_df2,
                                      replace=True,
                                      n_samples=len(majority_df),
                                      random_state=123)

    no_imbalance_df = pd.concat((majority_df,
                                 minority_df0_upsampled,
                                 minority_df2_upsampled))

    cv_df = cv_df.reset_index(drop=True)

    train_screennames = pad_text_sequences(train_df, 'screen_name')
    validation_screennames = pad_text_sequences(cv_df, 'screen_name')

    train_u_names = pad_text_sequences(train_df, 'u_name')
    validation_u_names = pad_text_sequences(cv_df, 'u_name')

    train_u_descriptions = pad_text_sequences(train_df, 'u_description')
    validation_u_descriptions = pad_text_sequences(cv_df, 'u_description')

    # Reshaping the sequences into something sklearn is compatible with
    train_u_descriptions_reshape = train_u_descriptions.view(-1, 300)
    train_descriptions_as_numpy = train_u_descriptions_reshape.data.numpy()
    validation_u_descriptions_reshape = validation_u_descriptions.view(-1, 300)
    validation_descriptions_as_numpy = validation_u_descriptions_reshape.data.numpy()

    # Setting up the feature scaler
    zero_to_one_scaler = MinMaxScaler()
    zero_to_one_scaler.fit(train_descriptions_as_numpy)

    # Scaling and reshaping the train set
    train_u_descriptions_scaled = zero_to_one_scaler.transform(train_descriptions_as_numpy)
    train_u_descriptions_scaled = Variable(torch.from_numpy(train_u_descriptions_scaled),
                                           requires_grad=False)
    train_u_descriptions_scaled = train_u_descriptions_scaled.view(train_u_descriptions.shape[0],
                                                                   train_u_descriptions.shape[1],
                                                                   300)

    # Scaling and reshaping the validation set
    validation_u_descriptions_scaled = zero_to_one_scaler.transform(validation_descriptions_as_numpy)
    validation_u_descriptions_scaled = Variable(torch.from_numpy(validation_u_descriptions_scaled),
                                                requires_grad=False)
    validation_u_descriptions_scaled = validation_u_descriptions_scaled.view(validation_u_descriptions.shape[0],
                                                                             validation_u_descriptions.shape[1],
                                                                             300)

    train_labels = Variable(torch.from_numpy(np.asarray(train_df.u_classv2_1, dtype=int)), requires_grad=False)
    validation_labels = Variable(torch.from_numpy(np.asarray(cv_df.u_classv2_1, dtype=int)), requires_grad=False)

    training_set = TrainDataset(no_imbalance_df.index.tolist(), train_labels, train_numbers, train_screennames,
                                train_u_names, train_u_descriptions, train_tweet_word_counts, train_quoted_word_counts,
                                train_quoted_descr_counts, train_retweeted_descr_counts)
    validation_set = ValidationDataset(cv_df.index.tolist(), validation_labels, validation_numbers,
                                       validation_screennames, validation_u_names, validation_u_descriptions,
                                       validation_tweet_word_counts, validation_quoted_word_counts,
                                       validation_quoted_descr_counts, validation_retweeted_descr_counts)

    batch_size = 512

    train_loader = torch.utils.data.DataLoader(dataset=training_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    random_sample = train_df.sample()
    getting_num_numeric_features, whatever = get_numeric(random_sample, 'train',
                                                         'dont matter', True,
                                                         train_retweet_counts[
                                                         random_sample.index[0]:(random_sample.index[0] + 1)])

    return train_loader, train_df, validation_loader, cv_df, getting_num_numeric_features, text_vectorizer, zero_to_one_scaler, number_scaler


def train_model(ensemble_model, l2_value, epochs, train_loader, validation_loader):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ensemble_model.parameters(), weight_decay=l2_value)

    loss_list = []
    cv_loss_list = []
    cv_ave_loss_list = []

    for i in range(epochs):

        for data in train_loader:
            # Get the data from the loader
            one, two, three, four, five, six, seven, eight, nine, _ = data

            # Move it to the GPUs
            one = one.to(device)
            two = two.to(device)
            three = three.to(device)
            four = four.to(device)
            five = five.to(device)
            six = six.to(device)
            seven = seven.to(device)
            eight = eight.to(device)

            # Getting labels
            true_y = nine.to(device)

            # Run it through the model
            prediction = ensemble_model(one, two, three,
                                        four, five, six,
                                        seven, eight)

            # Computing and storing losses
            loss = loss_function(prediction,
                                 true_y)
            loss_list.append(loss.item())

            # Back-prop and optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        for cv_data in validation_loader:
            # Get the data from the loader
            cv_one, cv_two, cv_three, cv_four, cv_five, cv_six, cv_seven, cv_eight, cv_nine, _ = cv_data

            # Move it to the GPUs
            cv_one = cv_one.to(device)
            cv_two = cv_two.to(device)
            cv_three = cv_three.to(device)
            cv_four = cv_four.to(device)
            cv_five = cv_five.to(device)
            cv_six = cv_six.to(device)
            cv_seven = cv_seven.to(device)
            cv_eight = cv_eight.to(device)

            # Getting labels
            cv_true_y = cv_nine.to(device)

            # Run it through the model
            cv_prediction = ensemble_model(cv_one, cv_two, cv_three,
                                           cv_four, cv_five, cv_six,
                                           cv_seven, cv_eight)

            # Computing and storing loss
            cv_loss = loss_function(cv_prediction,
                                    cv_true_y)
            cv_loss_list.append(cv_loss.item())

        cv_ave_loss_list.append(np.mean(cv_loss_list))

        if i >= 3:
            if cv_ave_loss_list[i] >= cv_ave_loss_list[i - 1] >= cv_ave_loss_list[i - 2] >= cv_ave_loss_list[i - 3]:
                print('Early stopping on epoch: %0.6f' % (i + 1))
                break

            # Occasional print
        if ((i + 1) % 5) == 0:
            print('Epoch %0.6f complete!' % (i + 1))

    # Plot Generation
    #    Setting up size and subplots
    f = plt.figure(figsize=(14, 5))
    train_fig = f.add_subplot(121)
    cv_fig = f.add_subplot(122)
    #    Plotting the train loss w/ iteration
    train_fig.plot(loss_list, c='black')
    train_fig.set_xlabel('Iteration')
    train_fig.set_title('Train Set')
    train_fig.set_ylabel('Cross Entropy Loss')
    #    Plotting the validation loss w/ iteration
    cv_fig.plot(cv_loss_list, c='red')
    cv_fig.set_xlabel('Iteration')
    cv_fig.set_title('Validation Set')
    cv_fig.set_ylabel('Cross Entropy Loss')

    f.savefig(os.path.join(PROJ_DIR, 'data', f'cross_entropy_{l2_value}.png'))

    # Generating outputs
    return ensemble_model, loss_list, cv_loss_list


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 10

train_loader, train_df, validation_loader, validation_df, numeric_features, text_vectorizer, zero_to_one_scaler, number_scaler = get_test_train_split()


dump(text_vectorizer, os.path.join(PROJ_DIR, 'data', 'text_vectorizer-v1.joblib'))
dump(zero_to_one_scaler, os.path.join(PROJ_DIR, 'data', 'zero_to_one_scaler_v1.joblib'))
dump(number_scaler, os.path.join(PROJ_DIR, 'data', 'number_scaler_v1.joblib'))

l2_values = [1e-4, 1e-3, 1e-2, 1e-1, 1]
for l2_value in l2_values:
    (train_ensemble1, train_pred1_y1, cv_pred1_y1, train_IDs1, cv_IDs1, train_scores1_y1, cv_scores1_y1,
     train_ordered_labels1_y1, cv_ordered_labels1_y1, train_confusion_matrix, train_accuracy, train_class_specifics,
     validation_confusion_matrix, validation_accuracy, validation_class_specifics) = grid_search(l2_value,
                                                                                                 numeric_features,
                                                                                                 text_vectorizer,
                                                                                                 train_loader, train_df,
                                                                                                 validation_loader,
                                                                                                 validation_df, device)

    torch.save(train_ensemble1, os.path.join(PROJ_DIR, 'data', f'trained-model-uclassv2-1-{l2_value}.pt'))

    with open(os.path.join(PROJ_DIR, f'uclassv2-1-{l2_value}-train-metrics.txt'), 'w+') as f:
        f.write('Confusion Matrix:\n')
        np.savetxt(f, train_confusion_matrix, fmt='%d')
        f.write('\nAccuracy: {:.2f}\n'.format(train_accuracy))
        f.write('\nClass-specifics:\n')
        f.write(train_class_specifics)

    with open(os.path.join(PROJ_DIR, f'uclassv2-1-{l2_value}-validate-metrics.txt'), 'w+') as f:
        f.write('Confusion Matrix:\n')
        np.savetxt(f, validation_confusion_matrix, fmt='%d')
        f.write('\nAccuracy: {:.2f}\n'.format(validation_accuracy))
        f.write('\nClass-specifics:\n')
        f.write(validation_class_specifics)
