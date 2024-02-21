import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# Function to preprocess data and perform topic modeling
def preprocess_and_topic_modeling(df):
    df.dropna(subset=["Title"], inplace=True)
    vectorizer = TfidfVectorizer(
        encoding="utf-8",
        lowercase=False,
        tokenizer=None,
        analyzer="word",
        stop_words=None,
        ngram_range=(1, 1),
        min_df=1,
        norm="l2",
        use_idf=True
    )

    train_data = vectorizer.fit_transform(df["Title"])
    lda_model = LatentDirichletAllocation(n_components=4)
    lda_model.fit(train_data)
    lda_components = lda_model.components_

    terms = vectorizer.get_feature_names_out()
    topic_words = []
    for index, component in enumerate(lda_components):
        zipped = zip(terms, component)
        top_terms_key = sorted(zipped, key=lambda t: t[1], reverse=True)[:20]
        top_terms_list = list(dict(top_terms_key).keys())
        topic_words.append(top_terms_list)

    return topic_words, lda_model

# Function to display WordCloud and Histogram
def display_wordcloud_and_histogram(topic_words, lda_model):
    list_colors = ["#17C37B", "#F92969", "#FACA0C", "#0D1117"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 20), dpi=160)

    for i, ax in enumerate(axes.flatten()):
        if i % 2 == 0:  # Word Clouds
            wc = WordCloud(
                background_color="white",
                max_words=20,
                max_font_size=80,
                colormap="tab10",
                color_func=lambda *args, **kwargs: list_colors[i//2],
                prefer_horizontal=1.0
            )
            wc.generate((" ").join(topic_words[i//2]))
            ax.imshow(wc)
            ax.set_title("Topic "+str(i//2), fontdict=dict(size=16))
            ax.axis("off")
        else:  # Histograms
            topic_idx = i//2
            topic_word_dist = lda_model.components_[topic_idx]
            sorted_word_idx = topic_word_dist.argsort()[::-1][:15]  # Select top 15 words
            sorted_word_freq = topic_word_dist[sorted_word_idx]

            ax.bar(range(len(sorted_word_freq)), sorted_word_freq, color=list_colors[i//2])
            ax.set_xlabel("Word Index")
            ax.set_ylabel("Frequency")
            ax.set_title("Word Frequency for Topic " + str(i//2))

    plt.subplots_adjust(hspace=0.5)
    st.pyplot(fig)

# Main function
def main():
    st.title("Amazon Reviews Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df_data = pd.read_csv(uploaded_file)
        topic_words, lda_model = preprocess_and_topic_modeling(df_data)
        display_wordcloud_and_histogram(topic_words, lda_model)

# Run the app
if __name__ == "__main__":
    main()
