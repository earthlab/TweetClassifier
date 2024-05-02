# Running the Docker Container
Below is the docker run command for running an instance of the image. The Docker container is a Flask app with two endpoints. A Flask app is useful because the classification models stay in memory during the lifetime of the Container, speeding up classification time.

The container will cache user classifications in the database and user tweets in csv files at volume mount location, specified by the -v flag. This way, data retrieved from the API can persist between containers.

You will also need an X API V2 bearer token for at least the Basic level of the API. Make sure to update the command below with this bearer token.

```bash=
sudo docker container run -v </absolute/path/to/mount/dir>:/app/persistent_data -d -p 5000:5000 -e X_BEARER_TOKEN=<your_x_api_v2_bearer_token> earthlabcu/tweet_classifier:latest

```

The container and its endpoints will be exposed on localhost:5000. The two endpoints, their query parameters, and example responses are:

## Endpoint 1: /classify_authors


Classify all the authors of the tweets returned from a tweet search.


### Query Params
* search_query (str) (**required**): String used to search for tweets
* start_time (str) (**required**): Beginning of date range to filter tweets by. Must be after current date minus 7 days. 'YYYY-MM-DDTHH:MM:SSZ' format.
* end_time (str) (**required**): End of date range to filter tweets by. Must be after current date minus 7 days. 'YYYY-MM-DDTHH:MM:SSZ' format.
* max_results (int): Maximum number of tweets to return for the tweet search.

Example
```python=
import requests

url = ('http://localhost:5000/classify_authors')
params = {
    'search_query': 'wildfire',
    'start_time': '2024-04-28T12:00:00Z'
    'end_time': '2024-04-28T13:00:00Z',
    'max_results': 10
}
response = requests.get(url, params=params)

print(response.status_code)
200

print(response.json())
{
    'classifications': [
        {'author_id': '22364225', 'contributor_role': 'em', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/TamoorZafar_', 'username': 'TamoorZafar_'},
        {'author_id': '1452856593024909316', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/chotahaazri', 'username': 'chotahaazri'},
        {'author_id': '2278932848', 'contributor_role': 'distribution', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/AlsarmiAmur', 'username': 'AlsarmiAmur'}, 
        ...
    ] 
}

```

This endpoint relies on the X API search_recent_tweets endpoint, thus start_date and end_date must be on or after the current date minus 7 days.


If you reach your X API rate limit, the authors will be cached for the search as unclassified. You will have to call the endpoint again when you are not rate limited to classify the remaining authors.

Example
```python=
import requests

url = 'http://localhost:5000/classify_authors'
params = {
     'search_query': 'extreme weather',
     'start_time': '2024-04-26T12:00:00Z',
     'end_time': '2024-04-27T14:00:00Z',
     'max_results': 20
}
response = requests.get(url, params=params)

print(response.status_code)
429

print(response.json())
{
    'classifications': [
        {'author_id': '1568858868742234115', 'contributor_role': 'em', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/NusratJahanJar2', 'username': 'NusratJahanJar2'},
        {'author_id': '1693295976562192384', 'contributor_role': 'em', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/BackpirchCrew', 'username': 'BackpirchCrew'}, {'author_id': '997871977351208961', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/shamashreshusur', 'username': 'shamashreshusur'},
        {'author_id': '348785359', 'contributor_role': 'em', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/TallCornJJ', 'username': 'TallCornJJ'},
        {'author_id': '929254227817517056', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/Gestvlt', 'username': 'Gestvlt'},
        {'author_id': '14471392', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/mytjake', 'username': 'mytjake'},
        {'author_id': '1249522069097734144', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/villaperpetua', 'username': 'villaperpetua'},
        {'author_id': '133718722', 'contributor_role': 'em', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/evaer101', 'username': 'evaer101'},
        {'author_id': '1641494385857638407', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/ultra_crepidity', 'username': 'ultra_crepidity'},
        {'author_id': '216659971', 'contributor_role': 'media', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/abigailamey', 'username': 'abigailamey'},
        {'author_id': '1265663843914510336', 'contributor_role': None, 'contributor_type': None, 'profile_url': 'https://twitter.com/awambmrgw', 'username': 'awambmrgw'},
        {'author_id': '1324336290159071232', 'contributor_role': None, 'contributor_type': None, 'profile_url': 'https://twitter.com/MatthewWielicki', 'username': 'MatthewWielicki'},
        {'author_id': '887766449577762817', 'contributor_role': None, 'contributor_type': None, 'profile_url': 'https://twitter.com/DaniellePendeg1', 'username': 'DaniellePendeg1'},
        {'author_id': '116895324', 'contributor_role': None, 'contributor_type': None, 'profile_url': 'https://twitter.com/newsjohnson', 'username': 'newsjohnson'},
        {'author_id': '702782965', 'contributor_role': None, 'contributor_type': None, 'profile_url': 'https://twitter.com/barryr5', 'username': 'barryr5'}],
    'error': 'Too Many Requests',
    'message': 'The get users tweets request amount has been exceeded. As many accounts as could be classified were, but you will need to call this later to classify the rest of them.'
}

```

After container startup, it may take ~5 minutes for the models to load in the app. Any requests made during this initialization time will hang.

## Endpoint 2: /get_all_authors

Return all of the authors in the database

Example
```python=
import requests

url = 'http://localhost:5000/get_all_authors'
response = requests.get(url)

print(response.status_code)
200

print(response.json())

{
    'classifications': [
        {'author_id': '22364225', 'contributor_role': 'em', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/TamoorZafar_', 'username': 'TamoorZafar_'},
        {'author_id': '1452856593024909316', 'contributor_role': 'personalized', 'contributor_type': 'individual', 'profile_url': 'https://twitter.com/chotahaazri', 'username': 'chotahaazri'}
    ]
}
```
