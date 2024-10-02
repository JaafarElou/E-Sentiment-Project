import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import io
import xlsxwriter
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import streamlit as st
from textblob import TextBlob



st.logo(image='Media\\logo.png', icon_image='Media\\icon.png')


st.set_page_config(
    page_title='Analyser ',
    page_icon="Media\\page_icon.png",
    layout='wide'
)

nltk.download('punkt')
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache_data
def load_data(file):
    ext = file.name.split('.')[-1]
    if ext == 'csv':
        df = pd.read_csv(file)
    elif ext in ['xls', 'xlsx']:
        df = pd.read_excel(file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
        return None
    return df


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt


@st.cache_data
def clean_tweets(df, text_column):
    df['clean_tweet'] = np.vectorize(remove_pattern)(df[text_column], "@[\w]*")
    df['clean_tweet'] = df['clean_tweet'].str.replace(r"[^a-zA-Z#]", " ", regex=True)
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]))
    tokenized_tweet = df['clean_tweet'].apply(lambda x: x.split())
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda sentence: [stemmer.stem(word) for word in sentence])
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    df['clean_tweet'] = tokenized_tweet
    return df


@st.cache_data
def generate_wordcloud(text, title):
    if text.strip():
        wordcloud = WordCloud(width=800, height=500, random_state=42, max_font_size=100).generate(text)
        buf = io.BytesIO()
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf
    else:
        return None


def hashtag_extract(tweets):
    hashtags = []
    for tweet in tweets:
        ht = re.findall(r"#(\w+)", tweet)
        hashtags.append(ht)
    return hashtags


@st.cache_data
def plot_top_hashtags(hashtags, title):
    freq = nltk.FreqDist(hashtags)
    d = pd.DataFrame({'Hashtag': list(freq.keys()), 'Count': list(freq.values())})
    d = d.nlargest(columns='Count', n=10)
    buf = io.BytesIO()
    plt.figure(figsize=(15, 9))
    sns.barplot(data=d, x='Hashtag', y='Count')
    plt.title(title)
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


st.image('Media\\analyser banner.png')

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xls", "xlsx"])

if 'preprocessed' not in st.session_state:
    st.session_state.preprocessed = False

if uploaded_file:
    df = load_data(uploaded_file)
    st.header('Dataset Preview')
    st.write(df.head(10))
    
    st.header("Dataset Informations")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    text_column = st.selectbox('Select the column containing text data', df.columns)

    if st.button('Preprocess Data'):
        
        st.header('Preprocessing Phase')

        df = clean_tweets(df, text_column)
        st.subheader('Cleaned Version Of Data - Preview')
        st.write("The process of cleaning include :\n- Removing Twitter handles (@), non-alphabetic characters (keeping hashtags) and words shorter than 4 characters\n- Tokenizing the cleaned tweets and stemming each word in the tokenized tweets\n- Re-joining the stemmed tokens into a single string for each tweet\n- Updating the DataFrame with the cleaned and processed tweets")
        st.write(df.head(10))
        
        df['clean_tweet'] = df['clean_tweet'].astype(str)
        df['polarity'] = df['clean_tweet'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df['clean_tweet'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        df['Label'] = df['polarity'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        final_df = df[[text_column, 'clean_tweet', 'polarity', 'subjectivity', 'Label']]

        
        st.subheader('Cleaned And Labeled Version Of Data - Preview')
        st.write("The 'Label' column was added based on the 'polarity' values:\n- 1 for positive polarity\n- 0 for neutral polarity\n- -1 for negative polarity.")
        st.write(final_df.head(10))

        
        st.sidebar.subheader("Download The Final Version Of Data")
        cleaned_data_csv = final_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Cleaned And Labeled Tweets CSV",
            data=cleaned_data_csv,
            file_name='cleaned_tweets.csv',
            mime='text/csv',
        )

        cleaned_data_excel = io.BytesIO()
        with pd.ExcelWriter(cleaned_data_excel, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Cleaned Tweets')
        cleaned_data_excel.seek(0)
        st.sidebar.download_button(
            label="Download Cleaned And Labeled Tweets Excel",
            data=cleaned_data_excel,
            file_name='cleaned_tweets.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )

        
        st.session_state.preprocessed = True
        st.session_state.df = df
        st.session_state.final_df = final_df


if st.session_state.preprocessed:
    df = st.session_state.df
    final_df = st.session_state.final_df

    if st.button('Visualize Data'):
        st.header('Visualizing Phase')
        st.subheader('Cleaned And Labeled Version Of Data - Preview')
        st.write(final_df.head(10))

        st.subheader('Word Cloud - All Tweets')
        all_words = " ".join([sentence for sentence in df['clean_tweet']])
        wc_all = generate_wordcloud(all_words, '')
        if wc_all:
            st.image(wc_all, use_column_width=True)

        st.subheader('Word Cloud - Positive Tweets')
        positive_words = " ".join([sentence for sentence in df['clean_tweet'][df['polarity'] > 0]])
        wc_positive = generate_wordcloud(positive_words, '')
        if wc_positive:
            st.image(wc_positive, use_column_width=True)

        st.subheader('Word Cloud - Negative Tweets')
        negative_words = " ".join([sentence for sentence in df['clean_tweet'][df['polarity'] < 0]])
        wc_negative = generate_wordcloud(negative_words, '')
        if wc_negative:
            st.image(wc_negative, use_column_width=True)

        st.subheader('Top Hashtags - Positive Tweets')
        ht_positive = hashtag_extract(df['clean_tweet'][df['polarity'] > 0])
        ht_positive = sum(ht_positive, [])
        plot_positive = plot_top_hashtags(ht_positive, '')
        if plot_positive:
            st.image(plot_positive, use_column_width=True)
        freq_pos = nltk.FreqDist(ht_positive)
        df_pos = pd.DataFrame({'Hashtag': list(freq_pos.keys()), 'Count': list(freq_pos.values())}).nlargest(columns='Count', n=10)
        st.write(df_pos.head())

        st.subheader('Top Hashtags - Negative Tweets')
        ht_negative = hashtag_extract(df['clean_tweet'][df['polarity'] < 0])
        ht_negative = sum(ht_negative, [])
        plot_negative = plot_top_hashtags(ht_negative, '')
        if plot_negative:
            st.image(plot_negative, use_column_width=True)
        freq_neg = nltk.FreqDist(ht_negative)
        df_neg = pd.DataFrame({'Hashtag': list(freq_neg.keys()), 'Count': list(freq_neg.values())}).nlargest(columns='Count', n=10)
        st.write(df_neg.head())

        st.subheader('Polarity & Subjectivity')
        st.write(df[['clean_tweet', 'polarity', 'subjectivity']].head(10))

        col1, col2 = st.columns(spec=[0.5,0.5])

        with col1:
            st.subheader('Distribution of Polarity')
            plt.figure(figsize=(10, 6))
            plt.hist(df['polarity'], bins=20, color='skyblue', edgecolor='black')
            plt.xlabel('Polarity')
            plt.ylabel('Frequency')
            plt.grid(True)
            buf_polarity = io.BytesIO()
            plt.savefig(buf_polarity, format='png')
            buf_polarity.seek(0)
            plt.close()
            st.image(buf_polarity, use_column_width=True)

            
            polarity_bins = pd.cut(df['polarity'], bins=5, labels=['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'])
            polarity_counts = polarity_bins.value_counts().sort_index()

        with col2 :
            st.subheader('Pie Chart of Polarity')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(polarity_counts, labels=polarity_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
            buf_pie_polarity = io.BytesIO()
            plt.savefig(buf_pie_polarity, format='png')
            buf_pie_polarity.seek(0)
            plt.close()
            st.image(buf_pie_polarity, use_column_width=True)

        col3, col4 = st.columns(spec=[0.5,0.5])

        with col3:
            st.subheader('Distribution of Subjectivity')
            plt.figure(figsize=(10, 6))
            plt.hist(df['subjectivity'], bins=20, color='lightgreen', edgecolor='black')
            plt.xlabel('Subjectivity')
            plt.ylabel('Frequency')
            plt.grid(True)
            buf_subjectivity = io.BytesIO()
            plt.savefig(buf_subjectivity, format='png')
            buf_subjectivity.seek(0)
            plt.close()
            st.image(buf_subjectivity, use_column_width=True)

            
            subjectivity_bins = pd.cut(df['subjectivity'], bins=5, labels=['Very Objective', 'Objective', 'Neutral', 'Subjective', 'Very Subjective'])
            subjectivity_counts = subjectivity_bins.value_counts().sort_index()

        with col4:
            st.subheader('Pie Chart of Subjectivity')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.pie(subjectivity_counts, labels=subjectivity_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
            buf_pie_subjectivity = io.BytesIO()
            plt.savefig(buf_pie_subjectivity, format='png')
            buf_pie_subjectivity.seek(0)
            plt.close()
            st.image(buf_pie_subjectivity, use_column_width=True)

        
        st.sidebar.subheader("Download The Final Version Of Data")
        cleaned_data_csv = final_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="Download Cleaned And Labeled Tweets CSV",
            data=cleaned_data_csv,
            file_name='cleaned_tweets.csv',
            mime='text/csv',
        )

        cleaned_data_excel = io.BytesIO()
        with pd.ExcelWriter(cleaned_data_excel, engine='xlsxwriter') as writer:
            final_df.to_excel(writer, index=False, sheet_name='Cleaned Tweets')
        cleaned_data_excel.seek(0)
        st.sidebar.download_button(
            label="Download Cleaned And Labeled Tweets Excel",
            data=cleaned_data_excel,
            file_name='cleaned_tweets.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )


        st.sidebar.subheader("Download The Visualizations")
        st.sidebar.download_button(label="Download Word Cloud - All Tweets", data=wc_all, file_name='wordcloud_all.png', mime='image/png')
        st.sidebar.download_button(label="Download Word Cloud - Positive Tweets", data=wc_positive, file_name='wordcloud_positive.png', mime='image/png')
        st.sidebar.download_button(label="Download Word Cloud - Negative Tweets", data=wc_negative, file_name='wordcloud_negative.png', mime='image/png')
        st.sidebar.download_button(label="Download Top Hashtags - Positive Tweets", data=plot_positive, file_name='top_hashtags_positive.png', mime='image/png')
        st.sidebar.download_button(label="Download Top Hashtags - Negative Tweets", data=plot_negative, file_name='top_hashtags_negative.png', mime='image/png')
        st.sidebar.download_button(label="Download Polarity Distribution", data=buf_polarity, file_name='polarity_distribution.png', mime='image/png')
        st.sidebar.download_button(label="Download Subjectivity Distribution", data=buf_subjectivity, file_name='subjectivity_distribution.png', mime='image/png')
        st.sidebar.download_button(label="Download Polarity Pie Chart", data=buf_pie_polarity, file_name='polarity_pie_chart.png', mime='image/png')
        st.sidebar.download_button(label="Download Subjectivity Pie Chart", data=buf_pie_subjectivity, file_name='subjectivity_pie_chart.png', mime='image/png')
