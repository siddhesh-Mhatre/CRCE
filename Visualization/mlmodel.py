from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from pyTCTK import TextNet, WordNet
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import joblib
import matplotlib.pyplot as plt

def main():
    st.title("Amazon Reviews Analysis")

    # File uploader
    url = st.text_input("Enter Keyword or Reviews:")
    mlp_clf = joblib.load('mlp_classifier_model.pkl')
    df_new_data = pd.DataFrame(
    {
        "Review": [
            url
        ]
    }
    )
    stopwords_to_keep = [
    "doesn", "doesn't", "doesnt", "dont", "don't", "not", "wasn't", "wasnt",
    "aren", "aren't", "arent",  "couldn", "couldn't", "couldnt", "didn",
    "didn't", "didnt", "hadn", "hadn't", "hadnt",  "hasn", "hasn't", "hasnt",
    "haven't", "havent", "isn", "isn't", "isnt", "mightn",  "mightn't",
    "mightnt", "mustn", "mustn't", "mustnt", "needn", "needn't", "neednt",
    "shan", "shan't", "shant", "shouldn", "shouldn't", "shouldnt", "wasn",
    "wasn't",  "wasnt", "weren", "weren't", "werent", "won", "won't", "wont",
    "wouldn", "wouldn't", "wouldnt", "good", "bad", "worst", "wonderfull",
    "best", "better"
    ]
    stopwords_to_add = [
        "es", "que", "en", "la", "las", "le", "les", "lo", "los", "de", "no",
        "el", "al", "un", "una", "se", "sa", "su", "sus", "por", "con", "mi",
        "para", "todo", "gb", "laptop", "computer", "pc"
    ]
    df_new_data = WordNet(
    data=df_new_data,
    column="Review"
    ).remove_stopword(
        language="english",
        lowercase=False,
        remove_accents=False,
        add_stopwords=stopwords_to_add,
        remove_stopwords=stopwords_to_keep
    )
    df_new_data = WordNet(
    data=df_new_data,
    column="Review"
    ).lemmatize(
        language="english",
        lowercase=False,
        remove_accents=False
    )
    df_new_data = TextNet(
    data=df_new_data,
    column="Review"
    ).remove_punctuation()
    list_new_data = df_new_data["Review"].tolist()
    vectorizer = joblib.load('vectoriser.pkl')
    new_data_vec = vectorizer.transform(list_new_data)
    new_data_pred = mlp_clf.predict(new_data_vec)
    new_data_pred_proba = mlp_clf.predict_proba(new_data_vec)
    new_data_pred_proba = np.around(new_data_pred_proba, decimals=4)
    new_data_pred = pd.DataFrame({"Predicted Sentiment": new_data_pred})
    new_data_pred_proba = pd.DataFrame(new_data_pred_proba, columns=["Negative proba", "Neutral proba", "Positive proba"])
    y_pred_new = pd.concat([df_new_data["Review"], new_data_pred], axis=1)
    y_pred_new = pd.concat([y_pred_new, new_data_pred_proba], axis=1)
    new_data_pred_proba_transposed = new_data_pred_proba.transpose()

# Display the prediction probabilities as a bar chart
    st.subheader("Prediction Probabilities")
    probas = y_pred_new.iloc[0][['Negative proba', 'Neutral proba', 'Positive proba']]

# Plotting the pie chart
    fig, ax = plt.subplots()
    ax.pie(probas, labels=probas.index, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.legend(title="Sentiments", loc="upper right", bbox_to_anchor=(1, 0, 0.5, 1))

# Display the pie chart in Streamlit
    st.pyplot(fig)
    # print(y_pred_new)
        # topic_words, lda_model = preprocess_and_topic_modeling(df_data)
        # display_wordcloud_and_histogram(topic_words, lda_model)

# Run the app
if __name__ == "__main__":
    main()