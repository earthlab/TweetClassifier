
import os
import earthpy as et
import pandas as pd
import re
import tweepy
import numpy as np
import datetime
from datetime import datetime, timezone
import scipy.stats


class TweetFilter(object):
    def __init__(self, client):
        self.client = client

        self.user_link_list = []
        self.user_profile_link_list = []
        self.user_name_list = []
        self.user_location_list = []
        self.user_bios_list = []
        self.status_count_list = []
        self.creation_date_list = []
        self.user_followers_list = []
        self.user_following_list = []
        self.max_tweet_retweets_list = []
        self.mean_tweet_retweets_list = []
        self.max_tweet_favorites_list = []
        self.mean_tweet_favorites_list = []
        self.user_tweets_list = []
        self.quoted_tweet_text_list = []
        self.linked_content_list = []
        self.retweeted_descriptions_list = []
        self.quoted_descriptions_list = []
        self.image_URL_list = []
        self.max_tweets_day_list = []
        self.max_tweets_hour_list = []
        self.mean_tweet_delta_list = []
        self.max_tweet_delta_list = []
        self.min_tweet_delta_list = []
        self.std_tweet_delta_list = []
        self.get_retweeted_percentage_list = []
        self.avg_retweets_list = []
        self.avg_favorites_list = []
        self.are_retweets_percentage_list = []
        self.max_retweets_list = []
        self.max_favorites_list = []
        self.account_age_list = []
        self.approx_entropy_r100_list = []
        self.approx_entropy_r500_list = []
        self.approx_entropy_r1000_list = []
        self.approx_entropy_r2000_list = []
        self.approx_entropy_r5000_list = []
        self.avg_tweets_day_list = []
        self.default_prof_image_list = []
        self.default_theme_background_list = []
        self.entropy_list = []
        self.favorites_by_user_list = []
        self.favorites_by_user_per_day_list = []
        self.geoenabled_list = []
        self.listed_count_list = []

    def remove_format_destroyers(self, string):
        string = re.sub('\n', '', string)
        string = re.sub('\r', '', string)
        string = re.sub('\t', '', string)
        return (string)

    def approx_en(self, U, m, r):

        def _maxdist(x_i, x_j):
            return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

        def _phi(m):
            x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
            C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
            return (N - m + 1.0) ** (-1) * sum(np.log(C))

        N = len(U)

        return abs(_phi(m + 1) - _phi(m))

    def etf_user_timeline_extract_apiv2(self, user_id, max_results: int):
        num_tweets = 0

        specific_user_favorites_list = []  # likes replaces favorites in API v2
        specific_user_max_favorites = 0
        specific_user_max_retweets = 0
        specific_user_quoted_descr = []
        specific_user_quoted_text = []
        specific_user_retweeted_count = 0  # retweeted tweet (not original)
        specific_user_retweet_count = 0  # number of tweets that get retweeted
        specific_user_retweets_list = []
        specific_user_total_og_favorites = 0
        specific_user_total_og_retweets = 0

        tweet_index = 0
        tweet_times = []
        user_link_strs = ''
        user_id_column = "user_id"

        try:
            paginator = tweepy.Paginator(
                self.client.get_users_tweets,  # method
                id=user_id,
                max_results=max_results,
                tweet_fields=['created_at',
                              'referenced_tweets',
                              'entities',
                              'public_metrics'],
                user_fields=['description',
                             'url',
                             'name',
                             'public_metrics',
                             'created_at',
                             'profile_image_url',
                             'location'],
                expansions=['author_id',
                            'referenced_tweets.id.author_id',
                            'referenced_tweets.id'],
                limit=2
            )
        except tweepy.errors.TweepyException as e:
            # return(num_tweets)
            print("error!")

        for page in paginator:
            # Get includes data for page(users and qtweet)
            tw_includes = page.includes
            if 'users' in page.includes:
                users = {u['id']: u for u in tw_includes['users']}
            if 'tweets' in page.includes:
                qtweet = {q['id']: q for q in tw_includes['tweets']}

            if page.data is not None:
                num_tweets += page.meta['result_count']
                tweets = page.data
                print(tweets)

            if num_tweets == 0:
                print("Tweets not found: {}".format(user_id))
                return

            for tweet in tweets:
                # Initialize tweet data variables
                twt_public_metrics = tweet.public_metrics
                twt_entities = tweet.entities

                num_urls = 0
                if twt_entities is not None:
                    if 'urls' in twt_entities:
                        twt_urls = twt_entities['urls']
                        num_urls = len(twt_urls)

                # reformat the tweet removing format destroyers
                twt_txt = self.remove_format_destroyers(tweet.text)
                twt_txt = twt_txt + ' EndOfTweet '

                # Get reference tweet id & type
                if tweet.referenced_tweets != None:
                    reftweet = {r['id']: r for r in tweet.referenced_tweets}
                    reftype = reftweet[list(reftweet.keys())[0]].type  # get it out of the first slot of the dictionary
                else:
                    reftype = ""

                # Check to see if this tweet is a retweet
                if reftype == "retweeted":
                    if twt_entities != None:
                        if 'mentions' in twt_entities.keys():
                            rt_id = int(twt_entities['mentions'][0]['id'])
                            if rt_id in users.keys():
                                retweeted_descr = self.remove_format_destroyers(users[rt_id].description)
                                retweeted_descr = retweeted_descr + ' EndOfDescr '
                            else:
                                retweeted_descr = ''
                        else:
                            retweeted_descr = ''
                    else:  # just have the tweetid not the user data
                        retweeted_descr = ''
                    # Increment retweeted count for user
                    specific_user_retweeted_count += 1
                else:
                    retweeted_descr = ''
                    specific_user_total_og_favorites += int(twt_public_metrics['like_count'])
                    specific_user_total_og_retweets += int(twt_public_metrics['retweet_count'])
                    if int(twt_public_metrics['retweet_count'] > 0):
                        specific_user_retweet_count += 1
                    specific_user_max_favorites = max(specific_user_max_favorites,
                                                      int(twt_public_metrics['like_count']))
                    specific_user_max_retweets = max(specific_user_max_retweets,
                                                     int(twt_public_metrics['retweet_count']))
                    specific_user_retweets_list.append(int(twt_public_metrics['retweet_count']))
                    specific_user_favorites_list.append(int(twt_public_metrics['like_count']))

                if reftype == "quoted":
                    # Get the user description and quoted tweet text and strip format destroyers]
                    refkey = list(reftweet.keys())[0]
                    if refkey in qtweet.keys():
                        quoted_tweet_text = self.remove_format_destroyers(qtweet[refkey].text)
                        quoted_tweet_text = quoted_tweet_text + ' EndOfTweet '
                        quoted_user_descr = self.remove_format_destroyers(users[qtweet[refkey].author_id].description)
                        quoted_user_descr = quoted_user_descr + ' EndOfDescr '
                    else:
                        quoted_tweet_text = ''
                        quoted_user_descr = ''
                else:
                    quoted_tweet_text = ''
                    quoted_user_descr = ''

                # Starting a growing string with the first tweet
                if tweet_index == 0:
                    tweet_strs = twt_txt
                    quoted_tweet_strs = quoted_tweet_text
                    quoted_user_descr_strs = quoted_user_descr
                    retweeted_user_descr_strs = retweeted_descr
                else:
                    tweet_strs = tweet_strs + twt_txt
                    quoted_tweet_strs = quoted_tweet_strs + quoted_tweet_text
                    quoted_user_descr_strs = quoted_user_descr_strs + quoted_user_descr
                    retweeted_user_descr_strs = retweeted_user_descr_strs + retweeted_descr

                # format links and add to twt_link_strs
                for index in range(num_urls):
                    current_link = twt_urls[index]['expanded_url']
                    current_link = current_link + ' EndOfLink '
                    if (current_link == 0) and (tweet_index == 0):
                        user_link_strs = current_link
                    else:
                        user_link_strs = user_link_strs + current_link

                tweet_times.append(tweet.created_at)  # used in Kylen's calculations below
                tweet_index += 1
                # BOTTOM OF PAGE/TWEETS LOOP

        # Further processing
        specific_user_total_og_tweets = num_tweets - specific_user_retweeted_count

        # Calculate statistics around original tweets
        if specific_user_total_og_tweets > 0:
            self.get_retweeted_percentage_list.append(specific_user_retweet_count / specific_user_total_og_tweets)
            self.avg_retweets_list.append(specific_user_total_og_retweets / specific_user_total_og_tweets)
            self.avg_favorites_list.append(specific_user_total_og_favorites / specific_user_total_og_tweets)
        else:
            self.get_retweeted_percentage_list.append(0)
            self.avg_retweets_list.append(0)
            self.avg_favorites_list.append(0)

        # Calculate retweet statistics
        self.are_retweets_percentage_list.append(specific_user_retweeted_count / num_tweets)
        self.max_retweets_list.append(specific_user_max_retweets)
        self.max_favorites_list.append(specific_user_max_favorites)

        # Add user information
        user_profile = users[int(user_id)]
        user_public_metrics = user_profile.public_metrics
        if 'url' in user_profile:
            self.user_link_list.append(user_profile.url)
        else:
            self.user_link_list.append('')
        self.user_profile_link_list.append('https://twitter.com/' + user_profile.username)
        self.user_name_list.append(self.remove_format_destroyers(user_profile.name))
        self.status_count_list.append(user_public_metrics['tweet_count'])
        self.creation_date_list.append(str(user_profile.created_at))

        account_age = (datetime.now(timezone.utc) - user_profile.created_at).days
        if account_age == 0 or account_age == None:
            account_age = 1
        self.account_age_list.append(account_age)

        avg_tweets_day = user_public_metrics['tweet_count'] / account_age
        self.avg_tweets_day_list.append(avg_tweets_day)

        # not supported in api v2 - subtituting False (null ~50% of time and True ~2% of time, mostly custom)
        self.default_prof_image_list.append(False)
        # not supported in api v2 - subtituting True (null ~50% of time and True ~28% of time, so default ~72%)
        self.default_theme_background_list.append(True)

        # not supported in api v2 - substituting False (null ~50% of time and True only 20% of time)
        self.geoenabled_list.append(False)
        self.listed_count_list.append(user_public_metrics['listed_count'])
        # not supported in api v2 - substituting median values from covid dataset 56k users
        self.favorites_by_user_list.append(27065)
        self.favorites_by_user_per_day_list.append(11.67)

        # This is actually a list of all tweets
        self.user_tweets_list.append(tweet_strs)
        self.user_bios_list.append(self.remove_format_destroyers(user_profile.description))
        if user_profile.location != None:
            self.user_location_list.append(self.remove_format_destroyers(user_profile.location))
        else:
            self.user_location_list.append(None)
        self.user_followers_list.append(user_public_metrics['followers_count'])
        self.user_following_list.append(user_public_metrics['following_count'])
        self.image_URL_list.append(user_profile.profile_image_url)

        # This if and elif represent the situation where a user ONLY retweets
        if len(specific_user_retweets_list) == 0:
            self.mean_tweet_retweets_list.append(0)
            self.max_tweet_retweets_list.append(0)
            self.mean_tweet_favorites_list.append(0)
            self.max_tweet_favorites_list.append(0)

        elif len(specific_user_favorites_list) == 0:
            self.mean_tweet_retweets_list.append(0)
            self.max_tweet_retweets_list.append(0)
            self.mean_tweet_favorites_list.append(0)
            self.max_tweet_favorites_list.append(0)
            # This represent when a user actually tweets original content
        else:
            self.mean_tweet_retweets_list.append(np.mean(np.asarray(specific_user_retweets_list)))
            self.max_tweet_retweets_list.append(np.max(np.asarray(specific_user_retweets_list)))
            self.mean_tweet_favorites_list.append(np.mean(np.asarray(specific_user_favorites_list)))
            self.max_tweet_favorites_list.append(np.max(np.asarray(specific_user_favorites_list)))

        self.linked_content_list.append(user_link_strs)

        # In the case of no retweeted users, supplying empty list
        if (len(retweeted_user_descr_strs) == 0):
            self.retweeted_descriptions_list.append('')
        else:
            self.retweeted_descriptions_list.append(retweeted_user_descr_strs)

        # In the case of no quoted tweets, supplying empty lists
        if (len(quoted_tweet_strs) == 0):
            self.quoted_tweet_text_list.append('')
        else:
            self.quoted_tweet_text_list.append(quoted_tweet_strs)

        if (len(quoted_user_descr_strs) == 0):
            self.quoted_descriptions_list.append('')
        else:
            self.quoted_descriptions_list.append(quoted_user_descr_strs)

        # Kylen Solvik's calculations using tweet_times list

        # Max tweets in a day
        days = np.array([[dt.year, dt.month, dt.day] for dt in tweet_times])
        days_counts = np.unique(days, axis=0, return_counts=True)
        ##### CHANGED THIS #####
        self.max_tweets_day_list.append(np.max(days_counts[1]))

        # Max tweets in an hour
        hours = np.array([[dt.year, dt.month, dt.day, dt.hour] for dt in tweet_times])
        hours_counts = np.unique(hours, axis=0, return_counts=True)
        ##### CHANGED THIS #####
        self.max_tweets_hour_list.append(np.max(hours_counts[1]))

        # Times between tweets
        tweet_deltas = []
        for i in range(len(tweet_times) - 1):
            tweet_deltas.append(tweet_times[i] - tweet_times[i + 1])
        tweet_deltas_s = np.array([td.seconds for td in tweet_deltas])
        if len(tweet_times) > 1:
            ##### CHANGED THIS #####
            mean_tweet_delta = np.mean(tweet_deltas_s)
            max_tweet_delta = np.max(tweet_deltas_s)
            min_tweet_delta = np.min(tweet_deltas_s)
            std_tweet_delta = np.std(tweet_deltas_s)
        else:
            ##### CHANGED THIS #####
            mean_tweet_delta = None
            max_tweet_delta = None
            min_tweet_delta = None
            std_tweet_delta = None
        self.mean_tweet_delta_list.append(mean_tweet_delta)
        self.max_tweet_delta_list.append(max_tweet_delta)
        self.min_tweet_delta_list.append(min_tweet_delta)
        self.std_tweet_delta_list.append(std_tweet_delta)

        ##### CHANGED THIS #####
        if len(tweet_deltas_s) > 2:
            # Approximate entropy with different r values
            approx_entropy_r100 = self.approx_en(tweet_deltas_s, 2, 100)
            approx_entropy_r500 = self.approx_en(tweet_deltas_s, 2, 500)
            approx_entropy_r1000 = self.approx_en(tweet_deltas_s, 2, 1000)
            approx_entropy_r2000 = self.approx_en(tweet_deltas_s, 2, 2000)
            approx_entropy_r5000 = self.approx_en(tweet_deltas_s, 2, 5000)
            # Simple entropy after binning
            delta_histo = np.histogram(tweet_deltas_s,
                                       np.arange(0, np.max(tweet_deltas_s) + 120, 120))
            entropy = scipy.stats.entropy(delta_histo[0])
        else:
            approx_entropy_r100 = None
            approx_entropy_r500 = None
            approx_entropy_r1000 = None
            approx_entropy_r2000 = None
            approx_entropy_r5000 = None
            entropy = None
        self.approx_entropy_r100_list.append(approx_entropy_r100)
        self.approx_entropy_r500_list.append(approx_entropy_r500)
        self.approx_entropy_r1000_list.append(approx_entropy_r1000)
        self.approx_entropy_r2000_list.append(approx_entropy_r2000)
        self.approx_entropy_r5000_list.append(approx_entropy_r5000)
        self.entropy_list.append(entropy)

        return self.etf_add_extract_columns(user_id)

    def etf_add_extract_columns(self, user_id: int):
        users_df = pd.DataFrame()
        users_df = users_df.assign(u_link=self.user_link_list)
        users_df = users_df.assign(u_profile_link=self.user_profile_link_list)
        users_df = users_df.assign(u_name=self.user_name_list)
        users_df = users_df.assign(u_location=self.user_location_list)
        users_df = users_df.assign(u_description=self.user_bios_list)
        users_df = users_df.assign(status_count=self.status_count_list)
        users_df = users_df.assign(creation_date=self.creation_date_list)
        users_df = users_df.assign(followers=self.user_followers_list)
        users_df = users_df.assign(following=self.user_following_list)
        users_df = users_df.assign(max_user_retweets=self.max_tweet_retweets_list)
        users_df = users_df.assign(mean_user_retweets=self.mean_tweet_retweets_list)
        users_df = users_df.assign(max_user_favorites=self.max_tweet_favorites_list)
        users_df = users_df.assign(mean_user_favorites=self.mean_tweet_favorites_list)
        users_df = users_df.assign(condensed_tweets=self.user_tweets_list)
        users_df = users_df.assign(quoted_tweets=self.quoted_tweet_text_list)
        users_df = users_df.assign(linked_content=self.linked_content_list)
        users_df = users_df.assign(retweeted_descr=self.retweeted_descriptions_list)
        users_df = users_df.assign(quoted_descr=self.quoted_descriptions_list)
        users_df = users_df.assign(profile_pic_URL=self.image_URL_list)
        users_df = users_df.assign(max_tweets_day=self.max_tweets_day_list)
        users_df = users_df.assign(max_tweets_hour=self.max_tweets_hour_list)
        users_df = users_df.assign(mean_tweet_delta=self.mean_tweet_delta_list)
        users_df = users_df.assign(max_tweet_delta=self.max_tweet_delta_list)
        users_df = users_df.assign(min_tweet_delta=self.min_tweet_delta_list)
        users_df = users_df.assign(std_tweet_delta=self.std_tweet_delta_list)
        users_df = users_df.assign(get_retweeted_percentage=self.get_retweeted_percentage_list)
        users_df = users_df.assign(avg_retweets=self.avg_retweets_list)
        users_df = users_df.assign(avg_favorites=self.avg_favorites_list)
        users_df = users_df.assign(are_retweets_percentage=self.are_retweets_percentage_list)
        users_df = users_df.assign(max_retweets=self.max_retweets_list)
        users_df = users_df.assign(max_favorites=self.max_favorites_list)
        users_df = users_df.assign(account_age=self.account_age_list)
        users_df = users_df.assign(approx_entropy_r100=self.approx_entropy_r100_list)
        users_df = users_df.assign(approx_entropy_r500=self.approx_entropy_r500_list)
        users_df = users_df.assign(approx_entropy_r1000=self.approx_entropy_r1000_list)
        users_df = users_df.assign(approx_entropy_r2000=self.approx_entropy_r2000_list)
        users_df = users_df.assign(approx_entropy_r5000=self.approx_entropy_r5000_list)
        users_df = users_df.assign(avg_tweets_day=self.avg_tweets_day_list)
        users_df = users_df.assign(default_prof_image=self.default_prof_image_list)
        users_df = users_df.assign(default_theme_background=self.default_theme_background_list)
        users_df = users_df.assign(entropy=self.entropy_list)
        users_df = users_df.assign(favorites_by_user=self.favorites_by_user_list)
        users_df = users_df.assign(favorites_by_user_per_day=self.favorites_by_user_per_day_list)
        users_df = users_df.assign(geoenabled=self.geoenabled_list)
        users_df = users_df.assign(listed_count=self.listed_count_list)
        # users_df.columns = users_df.columns.str.replace('username', 'screen_name')
        users_df = users_df.assign(URL=self.user_profile_link_list)
        users_df = users_df.assign(screen_name=self.user_name_list)
        users_df = users_df.assign(user_id=[user_id])

        users_df = users_df.reset_index(drop=True)
        return users_df

    # def etf_user_timeline_batch_extract_apiv2(self, users_input):
    #
    #     users_df = users_input.copy()
    #     timelineid = "user_id"  # username field
    #
    #     for user_index in range(len(users_input)):
    #         user_id = users_input.iloc[user_index][timelineid]
    #         username = users_input.iloc[user_index]["username"]
    #         print("i={} author_id = {} {}".format(user_index, user_id, username))
    #         users_df = self.etf_user_timeline_extract_apiv2(user_id, users_df)
    #     users_df = self.etf_add_extract_columns(users_df)
    #
    #     return users_df
