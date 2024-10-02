import streamlit as st
import joblib
from nltk.stem import PorterStemmer



st.logo(image='Media\\logo.png', icon_image='Media\\icon.png')


st.set_page_config(
    page_title='Predictor ',
    page_icon="Media\\page_icon.png",
    layout='wide'
)

stemmer = PorterStemmer()


def preprocess_text(text):
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)


def load_model_and_predict(new_tweets):
    model = joblib.load(r'Saved Model\nb_classifier.pkl')
    vectorizer = joblib.load(r'Saved Model\vectorizer.pkl')

    new_tweets_preprocessed = [preprocess_text(tweet) for tweet in new_tweets]
    new_tweets_transformed = vectorizer.transform(new_tweets_preprocessed)
    predictions = model.predict(new_tweets_transformed)

    results = []
    for prediction in predictions:
        if prediction == 1:
            results.append('Positive tweet')
        elif prediction == -1:
            results.append('Negative tweet')
        elif prediction == 0:
            results.append('Neutral tweet')
    return results


st.image("Media\\predictor banner.png")
new_tweet_input = st.text_area("Enter new tweets for classification (separate with a newline):")
if st.button('Predict'):
    new_tweets = new_tweet_input.split('\n')
    predictions = load_model_and_predict(new_tweets)
    st.dataframe({'prediction':predictions, 'Text': new_tweets})



