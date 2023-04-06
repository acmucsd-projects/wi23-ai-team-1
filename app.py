import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


def run():
    st.set_page_config(
        page_title="Detecting Toxic Comments", page_icon=":guardsman:", layout="wide"
    )

    st.title("Detecting Toxic Comments")

    # Add a sidebar
    st.sidebar.subheader("Team Members")
    st.sidebar.write("Aniket Gupta")
    st.sidebar.write("Arnav Modi")
    st.sidebar.write("Jeffrey Lee")
    st.sidebar.write("Jimmy Ying")
    st.sidebar.write("Steven Shi")
    st.sidebar.write("Vivian Liu")
    st.sidebar.write("Mentor: Vincent Tu")

    st.sidebar.write("")
    st.sidebar.subheader("Motivation")
    st.sidebar.write(
        "Online platforms offer us unprecedented opportunities to communicate and share knowledge. However, harmful comments can create a hostile environment that leads to cyberbullying and discrimination. To address this issue, our team developed a machine learning model that can classify harmful online comments and alert moderators to take action. By automating this process, we aim to create a safer and more inclusive online community for everyone."
    )

    st.sidebar.write("")
    st.sidebar.subheader("References")
    st.sidebar.markdown(
        "- [The Banality of Online Toxicity](https://policyoptions.irpp.org/magazines/october-2021/the-banality-of-online-toxicity/)"
    )
    st.sidebar.markdown(
        "- [Tensorflow Neural Networks Text Classification Reference](https://www.tensorflow.org/text/tutorials/text_classification_rnn)"
    )
    st.sidebar.markdown(
        "- [Tensorflow Class Weighting](https://www.tensorflow.org/guide/keras/train_and_evaluate#using_sample_weighting_and_class_weighting)"
    )
    st.sidebar.markdown(
        "- [Overfitting and Underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)"
    )
    st.sidebar.markdown(
        "- [Visualizing Deep Learning Models](https://towardsdatascience.com/deep-learning-model-visualization-tools-which-is-best-83ecbe14fa7)"
    )

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    st.sidebar.markdown(
        "<a href='https://github.com/acmucsd-projects/wi23-ai-team-1'><img src='https://cdn.iconscout.com/icon/free/png-256/github-163-761603.png' style='width: 100px;'></a>",
        unsafe_allow_html=True,
    )

    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.write("")
        st.markdown(
            "![Alt Text](https://media.istockphoto.com/id/1278019531/vector/victim-of-cyber-bullying.jpg?s=612x612&w=0&k=20&c=DyMvMsOGJJ-Q54LFpGsiH86Yaabfu43LuvCv_vKVHj0=)"
        )
        st.write("Toxic comments can take many different forms, such as:")
        st.write(
            "- Comments that are **toxic**: containing rude, disrespectful, or insulting language"
        )
        st.write(
            "- Comments that are **severely toxic**: containing extremely offensive or abusive language"
        )
        st.write(
            "- Comments that are **obscene**: containing vulgar, profane, or sexually explicit language"
        )
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write(
            "- Comments that contain **threats**: expressing an intention to harm someone"
        )
        st.write(
            "- Comments that are **insulting**: containing language that is intended to offend or belittle someone"
        )
        st.write(
            "- Comments that express **identity hate**: containing language that is derogatory or discriminatory towards a particular group of people based on their race, religion, gender, or other personal characteristics"
        )
        st.write("")

    st.subheader("Interactive Toxic Comment Detector")

    # Add a text box for user input
    comment = st.text_input("Enter your comment here:")

    # Add a submit button
    submit_button = st.button("Submit")

    model = tf.keras.models.load_model("model")

    if submit_button:
        # Load the model

        # Create a dataframe from the user input
        # user_input = pd.DataFrame({'comment_text': [comment]})

        # Make predictions
        predictions = predict_toxicity(comment)

        # Create a dataframe with the predictions
        # df = pd.DataFrame({'Toxic': predictions[0][0]}, index=[0])
        [toxic, severe_toxic, obscene, threat, insult, identity_hate] = predictions[0]
        df = pd.DataFrame(
            {
                "Toxic": toxic,
                #'Severely Toxic': predictions[0][1],
                "Obscene": obscene,
                #'Threat': predictions[0][3],
                "Insult": insult,
                #'Identity Hate': predictions[0][5]
            },
            index=[0],
        )

        # Display the predictions
        st.write("")
        st.subheader("Predictions")
        st.write(df)

        # Display the comment
        st.write("")
        st.subheader("Comment")
        st.write(comment)

        # Display the toxicity score
        st.write("")
        st.subheader("Toxicity Score")
        st.write((toxic + obscene + insult) / 3)


@st.cache_resource
def predict_toxicity(phrase):
    # Load the model
    model = tf.keras.models.load_model("model")

    # Make predictions
    predictions = model.predict([phrase])

    return predictions


if __name__ == "__main__":
    run()
