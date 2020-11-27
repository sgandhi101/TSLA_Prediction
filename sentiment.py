from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions, \
    SentimentOptions, CategoriesOptions

#Enter API key information to use the Watson API
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-11-16',
    iam_apikey="4AclcErHA5srDiyQzFCRIeihcGW0Ei_JgUDmDaVxwftj",
    url="https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/bd387bb8-806b-4d65-ac70-6e6f95791c40"
)

#Sentiment function that is fed in text
def sentiment(input_text):
    response = natural_language_understanding.analyze(
        text=input_text,
        features=Features(sentiment=SentimentOptions())).get_result()
    res = response.get('sentiment').get('document').get('score')
    return res