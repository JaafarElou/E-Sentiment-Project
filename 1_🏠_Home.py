import streamlit as st


st.logo(image='Media/logo.png', icon_image='Media/icon.png')

st.set_page_config(
    page_title='Home',
    page_icon="Media/page_icon.png"
)

st.image("Media/home banner.png")

st.header("Project Description")
st.write("""
Our sentiment analysis App is designed to help users analyse the sentiments conveyed in textual content quickly and accurately.\n
Starting from cleaning your dataset, Labeling it based on the polarity of the sentiments it representes in case you want to train your own model, to visualising the subjectivity and polarity of the data ending with predicting a text sentiment based on a pre-trained model, our application provides valuable insights into the emotional tone of your data.
""")

st.header("Example Results")

st.image("Media/examples.png")


st.header("What Users Can Expect")
st.write("""
- Accurate Sentiment Analysis: Our application utilizes state-of-the-art natural language processing algorithms to accurately determine the sentiment expressed in text, whether it's positive, negative, or neutral.

- Easy-to-Use Interface: With a simple and intuitive user interface, users can effortlessly input their text, initiate the analysis process, and receive clear and concise results.

- Fast Processing: Our application is optimized for speed, delivering near-instantaneous results even when analyzing large volumes of text.
""")


st.header("About the Team")
st.write("""
We are a group of students at "Data Science for Economics and Finance" Master's program. Our project, (Jo&Ka Web App), is a product of our enthusiasm for data analysis and our desire to apply our newly acquired skills to practical challenges. Despite being early in our academic journey, we are passionate about exploring the potential of sentiment analysis.

- Bouhdid Amal www.linkedin.com/in/amal-bouhdid
- El Ouakhchachi Jaafar www.linkedin.com/in/jaafar-el-ouakhchachi
- Meskine Khadija www.linkedin.com/in/khadija-meskine
- El Ouazdi Oussama www.linkedin.com/in/oussama-el-ouazdi
""")


st.header("Ressources")
st.write("""
Feel free to browse our githubs via these links :

- www.github.com/AmalBouhdid
- www.github.com/JaafarElou
- www.github.com/khadijameskine
- www.github.com/OssyElOuazdi
""")


st.markdown("""
<footer style='text-align: center;'>
    &copy; 2024 Jo&Ka Team. All rights reserved.
</footer>
""", unsafe_allow_html=True)
