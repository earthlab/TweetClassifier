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

from src.utils import clean_tweet

warnings.filterwarnings("ignore", category=UserWarning)

PROJ_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(PROJ_DIR, 'data', 'models')

NUMERIC_COLUMNS = ['u_followers_count', 'following',
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

NUMERIC_COLUMNS.extend([str(i) for i in range(0, 50)])


class TrainDataset(torch.utils.data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, list_ids, train_labels_to_specify, train_numbers, train_screennames, train_u_names,
                 train_u_descriptions, train_tweet_word_counts, train_quoted_word_counts, train_quoted_descr_counts,
                 train_retweeted_descr_counts):
        self.list_ids = list_ids
        self.train_labels = train_labels_to_specify
        self.train_numbers = train_numbers
        self.train_screen_names = train_screennames
        self.train_u_names = train_u_names
        self.train_u_descriptions = train_u_descriptions
        self.train_tweet_word_counts = train_tweet_word_counts
        self.train_quoted_word_counts = train_quoted_word_counts
        self.train_quoted_descr_counts = train_quoted_descr_counts
        self.train_retweeted_descr_counts = train_retweeted_descr_counts

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        # Select sample
        column_id = self.list_ids[index]

        # Getting all the different features
        x_numeric = self.train_numbers[column_id]
        x_sn = self.train_screen_names[:, column_id, :]
        x_un = self.train_u_names[:, column_id, :]
        x_descr = self.train_u_descriptions[:, column_id, :]
        x_tweet_words = self.train_tweet_word_counts[column_id]
        x_quoted_tweet_words = self.train_quoted_word_counts[column_id]
        x_quoted_descr_words = self.train_quoted_descr_counts[column_id]
        x_retweet_descr_words = self.train_retweeted_descr_counts[column_id]

        # Getting the labels
        y = self.train_labels[column_id]

        return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                y, column_id)


class ValidationDataset(torch.utils.data.Dataset):

    def __init__(self, list_ids, validation_labels_to_specify, validation_numbers, validation_screennames,
                 validation_u_names, validation_u_description, validation_tweet_word_counts,
                 validation_quoted_word_counts, validation_quoted_descr_counts, validation_retweeted_descr_count):
        self.list_labels = list_ids
        self.validation_labels = validation_labels_to_specify
        self.validation_labels_to_specify = validation_labels_to_specify
        self.validation_numbers = validation_numbers
        self.validation_screen_names = validation_screennames
        self.validation_u_names = validation_u_names
        self.validation_u_descriptions = validation_u_description
        self.validation_tweet_word_counts = validation_tweet_word_counts
        self.validation_quoted_word_counts = validation_quoted_word_counts
        self.validation_quoted_descr_counts = validation_quoted_descr_counts
        self.validation_retweeted_descr_counts = validation_retweeted_descr_count

    def __len__(self):
        return len(self.list_labels)

    def __getitem__(self, index):
        # Select sample
        label = self.list_labels[index]

        x_numeric = self.validation_numbers[label]
        x_sn = self.validation_screen_names[:, label, :]
        x_un = self.validation_u_names[:, label, :]
        x_descr = self.validation_u_descriptions[:, label, :]
        x_tweet_words = self.validation_tweet_word_counts[label]
        x_quoted_tweet_words = self.validation_quoted_word_counts[label]
        x_quoted_descr_words = self.validation_quoted_descr_counts[label]
        x_retweet_descr_words = self.validation_retweeted_descr_counts[label]

        # Getting the labels
        y = self.validation_labels[label]

        return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                y, label)


class TweetAuthorEnsemble(nn.Module):
    # Architecture
    def __init__(self, numeric_features, text_vectorizer):
        super().__init__()

        # Part2. Numerical component
        self.numeric_layer = nn.Sequential(
            nn.Linear(numeric_features.shape[1], 3, bias=False),
            nn.Tanh())

        # Part3. User name component PyTorch LSTM already has a tanh
        self.uname_lstm_layer = nn.LSTM(90, 3, num_layers=1, bias=False)

        # Part4. Screen name component
        self.screen_name_lstm_layer = nn.LSTM(90, 3, num_layers=1, bias=False)

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
        screen_name_output, _ = self.screen_name_lstm_layer(screen_name)
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


class TweetTypeEnsemble(nn.Module):
    def __init__(self, numeric_features, text_vectorizer):
        super().__init__()

        # Part2. Numerical component
        self.numeric_layer = nn.Sequential(
            nn.Linear(numeric_features.shape[1], 7, bias=False),
            nn.Tanh())

        # Part3. User name component
        #        PyTorch LSTM already has a tanh
        self.uname_lstm_layer = nn.LSTM(90, 7, num_layers=1, bias=False)

        # Part4. Screen name component
        self.screenname_lstm_layer = nn.LSTM(90, 7, num_layers=1, bias=False)

        # Part5. Description component
        self.descr_lstm_layer = nn.LSTM(300, 7, num_layers=1, bias=False)

        # Part6. User (re)tweet component - for ensembling
        self.tweet_layer_ensemble = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 7, bias=False),
            nn.Tanh())

        # Part7. Quoted tweet component
        self.quoted_tweet_layer = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 7, bias=False),
            nn.Tanh())

        # Part8. Quoted description component
        self.quoted_descr_layer = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 7, bias=False),
            nn.Tanh())

        # Part9. Retweeted description component
        self.retweet_descr_layer = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 7, bias=False),
            nn.Tanh())

        # Part10. Ensemble
        self.ensemble_layer = nn.Sequential(
            nn.Linear(56, 7, bias=False),
            nn.Tanh())

        # Part11. User (re)tweet component - for "skip connection"
        self.tweet_layer_skip = torch.nn.Sequential(
            nn.Linear(len(text_vectorizer.vocabulary_), 7, bias=True),
            nn.Sigmoid())

    # Forward pass method
    def forward(self, numbers, screen_name,
                user_name, description, tweets,
                quoted_tweets, quoted_descr, retweet_descr):
        # Part2. Numeric forward pass for ensembling
        numeric_layer_output_ensemble = self.numeric_layer(numbers)

        # Part3. User name through the char LSTM
        user_name_output, _ = self.uname_lstm_layer(user_name)
        #        Only take the last cell state
        user_name_output = user_name_output[:, -1, :]

        # Part4. Screen name through the char LSTM
        screen_name_output, _ = self.screenname_lstm_layer(screen_name)
        #        Only take the last cell state
        screen_name_output = screen_name_output[:, -1, :]

        # Part5. Description forward pass
        #        Description embedding through the word LSTM
        description_output, _ = self.descr_lstm_layer(description)
        #        Only take the last cell state
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
        #        10.1 - Creating what we'll add to the language
        #        Concat'ing the nine submodel outputs together
        combined_input = torch.cat((numeric_layer_output_ensemble,
                                    screen_name_output,
                                    user_name_output,
                                    description_output,
                                    tweets_output,
                                    quoted_tweet_output,
                                    quoted_descr_output,
                                    retweet_descr_output),
                                   dim=1)
        #        Forward-passing them through the simple FC layer
        supplemental_output = self.ensemble_layer(combined_input)

        # Part11. Skipped forward pass of the numerical component
        primary_output = self.tweet_layer_skip(tweets)

        #        Adding that 6-submodel-derived output to the primary
        #            tweet submodel output
        final_output = torch.add(primary_output, supplemental_output)
        return final_output


class Base:
    def __init__(self):
        self._all_chars = string.ascii_letters + '_.\"~ -/!:()|$%^&*+=[]{}<>?' + string.digits
        self._n_char = len(self._all_chars)
        self._fasttext_model = self._get_read_fasttext()

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._int_to_tweet_author = {0: 'feedbased', 1: 'individual', 2: 'organization'}
        self._tweet_author_to_int = {v: k for k, v in self._int_to_tweet_author.items()}

        self._int_to_tweet_type = {
            0: 'public sector',
            1: 'distribution',
            2: 'em',
            3: 'media',
            4: 'personalized',
            5: 'nonprofit',
            6: 'tribal'
        }
        self._tweet_type_to_int = {v: k for k, v in self._int_to_tweet_type.items()}

    @staticmethod
    def _get_read_fasttext():
        fasttext_dir = os.path.join(PROJ_DIR, 'data', 'models')
        fasttext_path = os.path.join(PROJ_DIR, 'data', 'models', 'wiki-news-300d-1M.vec')

        # download pre-trained vector if necessary
        # link from https://fasttext.cc/docs/en/supervised-models.html
        if not os.path.isfile(fasttext_path):
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec'
                                       '.zip', fasttext_path + '.zip')
            zip_ref = zipfile.ZipFile(fasttext_path + '.zip', 'r')
            zip_ref.extractall(fasttext_dir)
            zip_ref.close()
        assert os.path.isfile(fasttext_path)

        return KeyedVectors.load_word2vec_format(fasttext_path)

    def _char_to_index(self, char):
        # finds the index of char in the list of all chars
        index = self._all_chars.find(char)
        other_idx = self._n_char + 1
        if index == -1:
            # character not in list
            index = other_idx
        return index

    def _char_to_tensor(self, char):
        tensor = torch.zeros(1, self._n_char + 1)  # plus one for "other" chars
        tensor[0][self._char_to_index(char)] = 1
        return tensor

    def _name_to_tensor(self, name):
        # assume a minibatch size of 1
        tensor = torch.zeros(len(name), 1, self._n_char + 2)  # plus 2 for "other" chars
        for i, char in enumerate(name):
            tensor[i][0][self._char_to_index(char)] = 1
        return tensor

    def _word_to_tensor(self, word, dim=300):
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
            np_array = self._fasttext_model[word]
        except KeyError:
            np_array = np.zeros(dim)
        tensor = torch.from_numpy(np_array)
        return tensor

    def _desc_to_tensor(self, description, dim=300):
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
            tensor[i][0] = self._word_to_tensor(word)
        return tensor

    def _pad_text_sequences(self, df, type_of_text):
        """PyTorch really doesn't like having a batch of different sizes, and since
           this is still early-prototyping, we're going to bad those different sizes
           with zeros to a common size (the max). In the future, I may implement that
           miserable pseudopadding that knows to stop at the actual padding"""

        text_list = []

        for index in df.index:
            if type_of_text in ['screen_name', 'u_name']:
                x_text = Variable(self._name_to_tensor(df[type_of_text][index]), requires_grad=False)
            elif type_of_text == 'u_description':
                x_text = Variable(self._desc_to_tensor(df[type_of_text][index]), requires_grad=False)
            else:
                raise ValueError('Error: False argument, must be either (1) u_screen_name, (2) u_name, or'
                                 ' (3) u_description')
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

    @staticmethod
    def _get_numeric(df, data_type, standardizer, need_retweet_counts, retweet_counts):

        """A function to extract the numeric features from a dataframe and
           center and scale them for proper optimization"""

        # Don't remember why I did this
        numeric_features = df.copy()

        # Subsetting the dataframe with those columns
        # And converting to numpy array
        numeric_features = numeric_features[NUMERIC_COLUMNS]
        numeric_features = numeric_features.values

        if need_retweet_counts:
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
            print('Did that first error really not throw?: Incorrect data_type argument, please use (1) train or ' \
                  '(2) validation')

    @staticmethod
    def _get_retweet_count(natural_text):
        natural_text = np.reshape(natural_text, [len(natural_text), ])
        return [text.count(' RT ') for text in natural_text]

    @staticmethod
    def _clean_natural_text(natural_text):
        # Getting all the text
        natural_text = np.reshape(natural_text, [len(natural_text), ])
        natural_text = np.asarray([clean_tweet(text) for text in natural_text])

        return natural_text

    def _create_text_vectorizer(self, natural_text):
        clean_natural_text = self._clean_natural_text(natural_text)
        vect = CountVectorizer(binary=True)
        vect = vect.fit(clean_natural_text)
        return vect

    def _get_ngram_features(self, natural_text, vect):
        """Returns sparse matrices determined by either counts or tf-idf
           weightings. Inputs the dataframe, method, and ngram_range, then
           outputs the vectorized train and text matrices"""
        natural_text = self._clean_natural_text(natural_text)
        return Variable(
            torch.from_numpy(np.asarray(vect.transform(natural_text).todense()).astype(float)).type(torch.FloatTensor),
            requires_grad=False
        )

    @staticmethod
    def _binary_regex_indicator(search_string, column, df):
        """
        Returns a list with 0's and 1's that indicate
            whether 'search_string' occurs in 'column' from
            a dataframe
            """
        res = [1 * bool(re.search(search_string, i, re.IGNORECASE)) for i in df[column].values]
        return res

    @staticmethod
    def _resample_df(df, label_column) -> pd.DataFrame:
        # Find the majority class
        majority_class = df[label_column].value_counts().idxmax()

        # Create separate dataframes for minority classes
        majority_df = df[df[label_column] == majority_class]
        minority_dfs = [df[df[label_column] == cls] for cls in df[label_column].unique() if
                        cls != majority_class]

        # Determine target number of samples
        target_samples = len(majority_df)

        # Upsample minority classes to match the number of samples in the majority class
        minority_upsampled = [resample(df, replace=True, n_samples=target_samples, random_state=123) for df in
                              minority_dfs]

        # Concatenate the majority class dataframe with upsampled minority class dataframes
        return pd.concat([majority_df] + minority_upsampled)

    def _add_boolean_columns_to_df(self, user_tweet_df: pd.DataFrame):
        user_tweet_df['u_name'] = user_tweet_df['u_name'].astype(str)

        keywords = [
            'fire', 'gov', 'news', 'firefighter', 'emergency', 'wildland', 'wildfire', 'county', 'disaster',
            'management', 'paramedic', 'right', 'maga', 'journalist', 'reporter', 'editor', 'photographer', 'newspaper',
            'producer', 'anchor', 'photojournalist', 'tv', 'host', 'fm', 'morning', 'media', 'jobs', 'careers', 'job',
            'career', 'romance', 'captain', 'firefighters', 'official', 'operations', 'prevention', 'government',
            'responder', 'housing', 'station', 'correspondent', 'jewelry', 'trends', 'pio', 'ic', 'eoc', 'office',
            'bureau', 'police', 'pd', 'department', 'city', 'state', 'mayor', 'governor', 'vost', 'smem', 'trump',
            'politics', 'uniteblue', 'retired', 'revolution', 'ftw', 'difference', 'patriot', 'best', 'interested',
            'understand', 'clean', 'global', 'must', 'book', 'transportation', 'defense', 'warrior', 'christian',
            'tweet', 'first'
        ]

        for keyword in keywords:
            user_tweet_df[f'name_has_{keyword}'] = self._binary_regex_indicator(keyword, 'u_name', user_tweet_df)
            user_tweet_df[f'screen_name_has_{keyword}'] = self._binary_regex_indicator(keyword, 'screen_name',
                                                                                       user_tweet_df)

        return user_tweet_df

    def _create_one_hot_encodings(self, df: pd.DataFrame, text_vectorizer):
        retweet_counts = self._get_retweet_count(df['condensed_tweets'].fillna('').values)
        tweet_word_counts = self._get_ngram_features(df['condensed_tweets'].fillna('').values, text_vectorizer)
        train_quoted_word_counts = self._get_ngram_features(df['quoted_tweets'].fillna('').values, text_vectorizer)
        train_retweeted_descr_counts = self._get_ngram_features(df['retweeted_descr'].fillna('').values,
                                                                text_vectorizer)
        train_quoted_descr_counts = self._get_ngram_features(df['quoted_descr'].fillna('').values, text_vectorizer)

        return (retweet_counts, tweet_word_counts, train_quoted_word_counts, train_retweeted_descr_counts,
                train_quoted_descr_counts)


class TrainBase(Base):
    def __init__(self):
        super().__init__()

    def _produce_eval(self, model, data_set):
        predicted_label_list_y = []
        true_label_list_y = []
        label_list = []

        if data_set == 'train':
            loader = self._train_loader
            predictions_matrix_y = np.zeros((len(self._train_df), 3 if self._label_column == 'u_classv2_1' else 7))
            ordered_labels_y = np.zeros((len(self._train_df), 1))
        elif data_set == 'cv':
            loader = self._validation_loader
            predictions_matrix_y = np.zeros((len(self._validation_df), 3 if self._label_column == 'u_classv2_1' else 7))
            ordered_labels_y = np.zeros((len(self._validation_df), 1))
        else:
            raise ValueError('Invalid data set. Please use (1) train or (2) cv')

        for data in loader:

            # Get the data from the loader
            one, two, three, four, five, six, seven, eight, nine, label = data

            # Move it to the GPUs
            one = one.to(self._device)
            two = two.to(self._device)
            three = three.to(self._device)
            four = four.to(self._device)
            five = five.to(self._device)
            six = six.to(self._device)
            seven = seven.to(self._device)
            eight = eight.to(self._device)

            # Getting labels
            true_y = nine.to(self._device)

            # Run it through the model
            prediction = model(one, two, three, four, five, six, seven, eight)

            # Convert these probabilities to the label prediction
            prediction_array = prediction.cpu().data.numpy()
            predicted_label = np.argmax(prediction_array, axis=1).tolist()
            predicted_label_list_y.extend(predicted_label)

            # Storing IDs for data set inspection
            label_list_temp = label.cpu().data.numpy().tolist()
            label_list.extend(label_list_temp)

            # Get these-shuffled true labels for evaluation
            true_label_list_y.extend(true_y.cpu().data.numpy().tolist())

            id_count = 0
            for i in label_list_temp:
                predictions_matrix_y[i, :] = prediction_array[id_count]
                ordered_labels_y[i, :] = predicted_label[id_count]
                id_count += 1

        confusion_matrix = metrics.confusion_matrix(true_label_list_y, predicted_label_list_y)
        accuracy = metrics.accuracy_score(true_label_list_y, predicted_label_list_y)
        class_specifics = metrics.classification_report(true_label_list_y, predicted_label_list_y)

        return (predicted_label_list_y, label_list, predictions_matrix_y, ordered_labels_y, confusion_matrix, accuracy,
                class_specifics)

    def _get_test_train_split(self, user_tweet_df: pd.DataFrame, test_size: float = 0.2):
        # Split the dataframe into train and test sets
        user_tweet_df, validation_user_tweet_df = train_test_split(user_tweet_df, test_size=test_size,
                                                                   random_state=42)

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

        user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].replace({'em related': 'em'})
        user_tweet_df['u_classv2_1'] = user_tweet_df['u_classv2_1'].map(self._tweet_author_to_int)
        user_tweet_df['u_classv2_2'] = user_tweet_df['u_classv2_2'].map(self._tweet_type_to_int)
        pd.set_option('display.max_colwidth', -1)

        user_tweet_df = self._add_boolean_columns_to_df(user_tweet_df)

        user_tweet_df = user_tweet_df.dropna(subset=NUMERIC_COLUMNS)
        user_tweet_df = user_tweet_df.assign(has_description=1. - 1 * pd.isnull(user_tweet_df['u_description']))
        user_tweet_df.u_description = user_tweet_df.u_description.fillna('')

        train_df = user_tweet_df[user_tweet_df['set'] == 'train']
        train_df.reset_index(inplace=True)
        cv_df = user_tweet_df[user_tweet_df['set'] == 'validation']
        cv_df.reset_index(inplace=True)

        # Getting all the types of multi-word text data
        normal_tweets = train_df['condensed_tweets'].fillna('').astype(str).values
        quoted_tweets = train_df['quoted_tweets'].fillna('').astype(str).values
        normal_descr = train_df['u_description'].fillna('').astype(str).values
        retweet_descr = train_df['retweeted_descr'].fillna('').astype(str).values
        quoted_descr = train_df['quoted_descr'].fillna('').astype(str).values

        # Combining them into 1 array
        all_train_text = np.append(normal_tweets, (quoted_tweets, normal_descr, retweet_descr, quoted_descr))

        # Storing the full-vocabulary one-hot-encoder
        text_vectorizer = self._create_text_vectorizer(all_train_text)

        (train_retweet_counts, train_tweet_word_counts, train_quoted_word_counts, train_retweeted_descr_counts,
         train_quoted_descr_counts) = self._create_one_hot_encodings(train_df, text_vectorizer)

        (validation_retweet_counts, validation_tweet_word_counts, validation_quoted_word_counts,
         validation_retweeted_descr_counts,
         validation_quoted_descr_counts) = self._create_one_hot_encodings(cv_df, text_vectorizer)

        train_retweet_counts = np.asarray(train_retweet_counts).reshape(len(train_df), 1)
        validation_retweet_counts = np.asarray(validation_retweet_counts).reshape(len(cv_df), 1)

        train_numbers, number_scaler = self._get_numeric(train_df, 'train', 'dont matter',
                                                         True, train_retweet_counts)
        validation_numbers = self._get_numeric(cv_df, 'validation', number_scaler, True,
                                               validation_retweet_counts)

        train_df = train_df.reset_index(drop=True)

        no_imbalance_df = self._resample_df(train_df, self._label_column)

        train_screen_names = self._pad_text_sequences(train_df, 'screen_name')
        validation_screen_names = self._pad_text_sequences(cv_df, 'screen_name')

        train_u_names = self._pad_text_sequences(train_df, 'u_name')
        validation_u_names = self._pad_text_sequences(cv_df, 'u_name')

        train_u_descriptions = self._pad_text_sequences(train_df, 'u_description')
        validation_u_descriptions = self._pad_text_sequences(cv_df, 'u_description')

        # Reshaping the sequences into something sklearn is compatible with
        train_u_descriptions_reshape = train_u_descriptions.view(-1, 300)
        train_descriptions_as_numpy = train_u_descriptions_reshape.data.numpy()

        # Setting up the feature scaler
        zero_to_one_scaler = MinMaxScaler()
        zero_to_one_scaler.fit(train_descriptions_as_numpy)

        train_labels = Variable(torch.from_numpy(np.asarray(train_df[self._label_column], dtype=int)),
                                requires_grad=False)
        validation_labels = Variable(torch.from_numpy(np.asarray(cv_df[self._label_column], dtype=int)),
                                     requires_grad=False)

        training_set = TrainDataset(no_imbalance_df.index.tolist(), train_labels, train_numbers, train_screen_names,
                                    train_u_names, train_u_descriptions, train_tweet_word_counts,
                                    train_quoted_word_counts,
                                    train_quoted_descr_counts, train_retweeted_descr_counts)

        validation_set = ValidationDataset(cv_df.index.tolist(), validation_labels, validation_numbers,
                                           validation_screen_names, validation_u_names, validation_u_descriptions,
                                           validation_tweet_word_counts, validation_quoted_word_counts,
                                           validation_quoted_descr_counts, validation_retweeted_descr_counts)

        batch_size = 512
        train_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)

        random_sample = train_df.sample()
        numeric_features_count, _ = self._get_numeric(random_sample, 'train', 'na',
                                                      True,
                                                      train_retweet_counts[
                                                      random_sample.index[0]:(random_sample.index[0] + 1)])

        return (train_loader, train_df, validation_loader, cv_df, numeric_features_count, text_vectorizer,
                zero_to_one_scaler, number_scaler)

    def _grid_search(self, model, l2_value: float, num_epochs: int):

        # Train the model w/ specified hyperparameter
        trained_model = self._train_model(model, l2_value, num_epochs)

        # Produce evaluations of the trained model
        (train_predictions_y, train_IDs, train_scores_y, train_ordered_labels_y, train_confusion_matrix, train_accuracy,
         train_class_specifics) = self._produce_eval(model, 'train')

        (validation_predictions_y, validation_IDs, validation_scores_y, validation_ordered_labels_y,
         validation_confusion_matrix, validation_accuracy, validation_class_specifics) = self._produce_eval(model, 'cv')

        # Store the model and predictions
        return (trained_model, train_confusion_matrix, train_accuracy, train_class_specifics,
                validation_confusion_matrix, validation_accuracy, validation_class_specifics)

    def _train_model(self, model, l2_value, epochs):
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=l2_value)

        loss_list = []
        cv_loss_list = []
        cv_ave_loss_list = []

        for i in range(epochs):

            for data in self._train_loader:
                # Get the data from the loader
                one, two, three, four, five, six, seven, eight, nine, _ = data

                # Move it to the GPUs
                one = one.to(self._device)
                two = two.to(self._device)
                three = three.to(self._device)
                four = four.to(self._device)
                five = five.to(self._device)
                six = six.to(self._device)
                seven = seven.to(self._device)
                eight = eight.to(self._device)

                # Getting labels
                true_y = nine.to(self._device)

                # Run it through the model
                prediction = model(one, two, three, four, five, six, seven, eight)

                # Computing and storing losses
                loss = loss_function(prediction,
                                     true_y)
                loss_list.append(loss.item())

                # Back-prop and optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for cv_data in self._validation_loader:
                # Get the data from the loader
                cv_one, cv_two, cv_three, cv_four, cv_five, cv_six, cv_seven, cv_eight, cv_nine, _ = cv_data

                # Move it to the GPUs
                cv_one = cv_one.to(self._device)
                cv_two = cv_two.to(self._device)
                cv_three = cv_three.to(self._device)
                cv_four = cv_four.to(self._device)
                cv_five = cv_five.to(self._device)
                cv_six = cv_six.to(self._device)
                cv_seven = cv_seven.to(self._device)
                cv_eight = cv_eight.to(self._device)

                # Getting labels
                cv_true_y = cv_nine.to(self._device)

                # Run it through the model
                cv_prediction = model(cv_one, cv_two, cv_three, cv_four, cv_five, cv_six, cv_seven, cv_eight)

                # Computing and storing loss
                cv_loss = loss_function(cv_prediction, cv_true_y)
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

        f.savefig(os.path.join(MODEL_DIR, f'{self._label_column}_{l2_value}', f'cross_entropy_{l2_value}.png'))

        return model

    def output_model(self, model, l2_value: float, num_epochs: int):
        model_dir = os.path.join(MODEL_DIR, f'{self._label_column}_{l2_value}')
        os.makedirs(model_dir, exist_ok=True)
        print(f'creating dir {model_dir}')

        dump(self._text_vectorizer, os.path.join(model_dir, f'text_vectorizer-{self._label_column}-{l2_value}.joblib'))
        dump(self._zero_to_one_scaler, os.path.join(model_dir,
                                                    f'zero_to_one_scaler_{self._label_column}_{l2_value}.joblib'))
        dump(self._number_scaler, os.path.join(model_dir, f'number_scaler_{self._label_column}_{l2_value}.joblib'))

        (
            trained_model, train_confusion_matrix, train_accuracy, train_class_specifics,
            validation_confusion_matrix, validation_accuracy, validation_class_specifics
        ) = self._grid_search(model, l2_value, num_epochs)

        torch.save(trained_model, os.path.join(model_dir, f'trained-model-{self._label_column}-{l2_value}.pt'))
        print(f"saved to {os.path.join(model_dir, f'trained-model-{self._label_column}-{l2_value}.pt')}")

        with open(os.path.join(model_dir, f'{self._label_column}-{l2_value}-train-metrics.txt'), 'w+') as f:
            f.write('Confusion Matrix:\n')
            np.savetxt(f, train_confusion_matrix, fmt='%d')
            f.write('\nAccuracy: {:.2f}\n'.format(train_accuracy))
            f.write('\nClass-specifics:\n')
            f.write(train_class_specifics)

        with open(os.path.join(model_dir, f'{self._label_column}-{l2_value}-validate-metrics.txt'), 'w+') as f:
            f.write('Confusion Matrix:\n')
            np.savetxt(f, validation_confusion_matrix, fmt='%d')
            f.write('\nAccuracy: {:.2f}\n'.format(validation_accuracy))
            f.write('\nClass-specifics:\n')
            f.write(validation_class_specifics)


class TrainTweetAuthorModel(TrainBase):
    def __init__(self):
        super().__init__()
        self._label_column = 'u_classv2_1'

        (self._train_loader, self._train_df, self._validation_loader, self._validation_df, self._numeric_features,
         self._text_vectorizer, self._zero_to_one_scaler, self._number_scaler) = self._get_test_train_split(
            pd.read_csv(os.path.join(PROJ_DIR, 'data', 'training', 'training_data_with_lda_columns.csv'))
        )

    def output_model(self, l2_value: float, num_epochs: int):
        ensemble = TweetAuthorEnsemble(self._numeric_features, self._text_vectorizer)
        ensemble = nn.DataParallel(ensemble)
        ensemble.to(self._device)

        super().output_model(ensemble, l2_value, num_epochs)


class TrainTweetTypeModel(TrainBase):
    def __init__(self):
        super().__init__()
        self._label_column = 'u_classv2_2'

        (self._train_loader, self._train_df, self._validation_loader, self._validation_df, self._numeric_features,
         self._text_vectorizer, self._zero_to_one_scaler, self._number_scaler) = self._get_test_train_split(
            pd.read_csv(os.path.join(PROJ_DIR, 'data', 'training', 'training_data_with_lda_columns.csv'))
        )

    def output_model(self, l2_value: float, num_epochs: int):
        ensemble = TweetTypeEnsemble(self._numeric_features, self._text_vectorizer)
        ensemble = nn.DataParallel(ensemble)
        ensemble.to(self._device)

        super().output_model(ensemble, l2_value, num_epochs)
