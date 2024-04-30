import os

from src.train_model import MODEL_DIR, NUMERIC_COLUMNS, Base

import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from joblib import load


class InferenceDataset(torch.utils.data.Dataset):

    def __init__(self, tweet_df, list_ids, inference_numbers, inference_screen_names,
                 inference_u_names, inference_u_description, inference_tweet_word_counts,
                 inference_quoted_word_counts, inference_quoted_descr_counts, inference_retweeted_descr_count):
        self.tweet_df = tweet_df
        self.list_labels = list_ids
        self.inference_numbers = inference_numbers
        self.inference_screen_names = inference_screen_names
        self.inference_u_names = inference_u_names
        self.inference_u_descriptions = inference_u_description
        self.inference_tweet_word_counts = inference_tweet_word_counts
        self.inference_quoted_word_counts = inference_quoted_word_counts
        self.inference_quoted_descr_counts = inference_quoted_descr_counts
        self.inference_retweeted_descr_counts = inference_retweeted_descr_count

    def __len__(self):
        return len(self.list_labels)

    def __getitem__(self, index):
        # Select sample
        label = self.list_labels[index]

        x_numeric = self.inference_numbers[label]
        x_sn = self.inference_screen_names[label]
        x_un = self.inference_u_names[label]
        x_descr = self.inference_u_descriptions[label]
        x_tweet_words = self.inference_tweet_word_counts[label]
        x_quoted_tweet_words = self.inference_quoted_word_counts[label]
        x_quoted_descr_words = self.inference_quoted_descr_counts[label]
        x_retweet_descr_words = self.inference_retweeted_descr_counts[label]

        names = self.tweet_df['screen_name'][label]

        return (x_numeric, x_sn, x_un, x_descr, x_tweet_words,
                x_quoted_tweet_words, x_quoted_descr_words, x_retweet_descr_words,
                label, names)


class InferenceBase(Base):
    def __init__(self):
        super().__init__()
        self._text_vectorizer = None
        self._number_scaler = None
        self._ensemble_model = None

    def _create_dataset(self, user_tweet_df: pd.DataFrame):
        # Optionally, you can reset the index of the resulting dataframes
        user_tweet_df.reset_index(drop=True, inplace=True)

        user_tweet_df = user_tweet_df.drop('Unnamed: 0', axis=1)
        user_tweet_df = user_tweet_df.reset_index(drop=True)

        user_tweet_df = self._add_boolean_columns_to_df(user_tweet_df)

        user_tweet_df = user_tweet_df.dropna(subset=NUMERIC_COLUMNS)
        user_tweet_df = user_tweet_df.assign(has_description=1. - 1 * pd.isnull(user_tweet_df['u_description']))
        user_tweet_df.u_description = user_tweet_df.u_description.fillna('')

        (inference_retweet_counts, inference_tweet_word_counts, inference_quoted_word_counts,
         inference_retweeted_descr_counts, inference_quoted_descr_counts) = self._create_one_hot_encodings(
            user_tweet_df, self._text_vectorizer)

        inference_retweet_counts = np.asarray(inference_retweet_counts).reshape(len(user_tweet_df), 1)

        inference_numbers = self._get_numeric(user_tweet_df, 'validation', self._number_scaler,
                                              True, inference_retweet_counts)

        user_tweet_df = user_tweet_df.reset_index(drop=True)

        # there is only one row of input so don't need to pad text sequences

        inference_set = InferenceDataset(
            user_tweet_df,
            range(len(user_tweet_df)),
            inference_numbers,
            Variable(self._name_to_tensor(user_tweet_df['screen_name'][0]), requires_grad=False),
            Variable(self._name_to_tensor(user_tweet_df['u_name'][0]), requires_grad=False),
            Variable(self._desc_to_tensor(user_tweet_df['u_description'][0]), requires_grad=False),
            inference_tweet_word_counts,
            inference_quoted_word_counts,
            inference_quoted_descr_counts,
            inference_retweeted_descr_counts
        )
        batch_size = 512
        inference_loader = torch.utils.data.DataLoader(dataset=inference_set, batch_size=batch_size, shuffle=True)

        return inference_loader

    def run_inference(self, user_tweet_df: pd.DataFrame):
        inference_loader = self._create_dataset(user_tweet_df)

        predictions = []
        for data in inference_loader:
            # Get the data from the loader
            two, three, four, five, six, seven, eight, nine, ID, names = data

            # Move it to the GPUs
            # one = one.to(device)
            two = two.to(self._device)
            three = three.to(self._device)
            four = four.to(self._device)
            five = five.to(self._device)
            six = six.to(self._device)
            seven = seven.to(self._device)
            eight = eight.to(self._device)
            nine = nine.to(self._device)

            # Run it through the model
            prediction = self._ensemble_model(two, three, four, five, six, seven, eight, nine)

            # Convert these probabilities to the label prediction
            predictions.extend(np.argmax(prediction.cpu().data.numpy(), axis=1).tolist())

        return predictions


class InferenceType(InferenceBase):
    def __init__(self):
        super().__init__()
        self._number_scaler = load(os.path.join(MODEL_DIR, 'number_scaler_u_classv2_1.joblib'))
        self._zero_to_one_number_scaler = load(os.path.join(MODEL_DIR, 'zero_to_one_scaler_u_classv2_1.joblib'))
        self._text_vectorizer = load(os.path.join(MODEL_DIR, 'text_vectorizer-u_classv2_1.joblib'))
        self._ensemble_model = torch.load(os.path.join(MODEL_DIR, 'trained-model-u_classv2_1.pt'),
                                          map_location=torch.device(self._device))
        if not torch.cuda.is_available():
            self._ensemble_model = torch.load(self._ensemble_model, map_location='cpu')

    def run_inference(self, user_tweet_df: pd.DataFrame):
        predicted_authors = super().run_inference(user_tweet_df)

        return [self._int_to_tweet_author[i] for i in predicted_authors]


class InferenceRole(InferenceBase):
    def __init__(self):
        super().__init__()
        self._number_scaler = load(os.path.join(MODEL_DIR, 'number_scaler_u_classv2_2.joblib'))
        self._zero_to_one_number_scaler = load(os.path.join(MODEL_DIR, 'zero_to_one_scaler_u_classv2_2.joblib'))
        self._text_vectorizer = load(os.path.join(MODEL_DIR, 'text_vectorizer-u_classv2_2.joblib'))
        self._ensemble_model = torch.load(os.path.join(MODEL_DIR, 'trained-model-u_classv2_2.pt'),
                                          map_location=self._device)
        if not torch.cuda.is_available():
            self._ensemble_model = torch.load(self._ensemble_model, map_location=torch.device('cpu'))

    def run_inference(self, user_tweet_df: pd.DataFrame):
        predicted_types = super().run_inference(user_tweet_df)

        return [self._int_to_tweet_type[i] for i in predicted_types]
