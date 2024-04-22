import re


def clean_tweet(tweet):
    """Function to perform standard tweet processing. Includes removing URLs, my
       end-of-line character, punctuation, lower casing, and recombining the rt
       character. Inputs a str, outputs a str"""

    # For later - to remove punctuation
    # Setting rule to replace any punctuation chain with equal amount of white space
    replace_punctuation = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

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

    return (phase7)
