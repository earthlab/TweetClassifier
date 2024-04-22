import os
import string
import re
import urllib
import zipfile
from collections import OrderedDict
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
DATA_DIR = os.path.join(PROJ_PATH, '../data')
MODEL_DIR = os.path.join(PROJ_PATH, '../model_inputs')





def main():

    output_file = os.path.join(DATA_DIR, 'test_results_tweets_unimodal.csv')

    user_tweet_df = pd.read_csv(os.path.join(DATA_DIR, 'PROTO_merged.csv'))



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

            return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                    x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                    ID, names)

    validation_set = validation_Dataset(range(len(user_tweet_df)))

    batch_size = 512

    validation_loader = torch.utils.data.DataLoader(dataset=validation_set,
                                                    batch_size=batch_size,
                                                    shuffle=True)

    if torch.cuda.is_available():
        # Use the first available GPU
        device = torch.device("cuda:0")
        print("Using GPU:", torch.cuda.get_device_name(0))
    else:
        # Use CPU if GPUs are not available
        device = torch.device("cpu")
        print("Using CPU")

    ensemble_model_v2_1 = torch.load(ensemble_model_file_v2_1, map_location=device)
    ensemble_model_v2_2 = torch.load(ensemble_model_file_v2_2, map_location=device)

    print('loaded models')

    predicted_authors = []
    predicted_type = []
    name_list = []

    loader = validation_loader
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

        # Storing names
        name_list.extend(names)

    hat_df1 = pd.DataFrame({'predicted_y1_redone': predicted_label_list_y})
    hat_df1['screen_name'] = name_list
    hat_df1['predicted_y1_redone'] = hat_df1['predicted_y1_redone'].replace({0: 'feed based',
                                                                             1: 'individual',
                                                                             2: 'organization'})
    hat_df1.head()

    predicted_label_list_y = []
    name_list = []

    loader = validation_loader
    predictions_matrix_y = np.zeros((len(user_tweet_df), 5))
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
        prediction = ensemble_model_v2_2(two, three,
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

    hat_df2 = pd.DataFrame({'predicted_y2_redone': predicted_label_list_y})
    hat_df2['screen_name'] = name_list
    hat_df2['predicted_y2_redone'] = hat_df2['predicted_y2_redone'].replace({0: 'civic/public sector',
                                                                             1: 'distribution',
                                                                             2: 'em',
                                                                             3: 'media',
                                                                             4: 'personalized'})
    hat_df2.head()

    hat_df = pd.merge(hat_df1, hat_df2, on='screen_name')
    hat_df.head()

    total_df = pd.merge(hat_df, user_tweet_df, on='screen_name')
    total_df.head()

    total_df.to_csv(output_file)


if __name__ == '__main__':
    main()
