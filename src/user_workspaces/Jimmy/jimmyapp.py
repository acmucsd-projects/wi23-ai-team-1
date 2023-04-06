import streamlit as st
import pandas as pd
import numpy as np
#import tensorflow as tf


def run():
    st.set_page_config(page_title='Detecting Toxic Comments',
                       page_icon=':guardsman:', layout='wide')

    st.title('Detecting Toxic Comments')

    # Add a sidebar
    st.sidebar.subheader('References')
    st.sidebar.markdown('- [The Banality of Online Toxicity](https://policyoptions.irpp.org/magazines/october-2021/the-banality-of-online-toxicity/)')
    st.sidebar.markdown('- [Tensorflow Neural Networks Text Classification Reference](https://www.tensorflow.org/text/tutorials/text_classification_rnn)')
    st.sidebar.markdown('- [Tensorflow Class Weighting](https://www.tensorflow.org/guide/keras/train_and_evaluate#using_sample_weighting_and_class_weighting)')
    st.sidebar.markdown('- [Overfitting and Underfitting](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit)')
    st.sidebar.markdown('- [Visualizing Deep Learning Models](https://towardsdatascience.com/deep-learning-model-visualization-tools-which-is-best-83ecbe14fa7)')

    st.sidebar.subheader('Building positive relationships')
    st.sidebar.write(
        'When we treat others with respect, kindness, and compassion, we are more likely to build positive and meaningful relationships with them.')

    st.sidebar.subheader('Creating a safe and supportive environment')
    st.sidebar.write(
        'Toxic comments can hurt others and create a negative and unsafe environment. Being respectful towards others helps create a safe and supportive environment where everyone can thrive.')

    st.sidebar.subheader('Fostering mutual understanding')
    st.sidebar.write(
        'Respectful behavior allows us to understand and appreciate different perspectives, backgrounds, and cultures, which helps us build bridges of mutual understanding.')

    st.sidebar.subheader('Promoting personal growth')
    st.sidebar.write(
        'When we treat others with respect, we are also promoting our own personal growth. Respect helps us develop self-awareness, empathy, and compassion, which are essential for personal growth and development.')

    st.write('')

    col1, col2 = st.columns(2)

    with col1:
        st.write('')
        st.markdown("![Alt Text](https://media.istockphoto.com/id/1278019531/vector/victim-of-cyber-bullying.jpg?s=612x612&w=0&k=20&c=DyMvMsOGJJ-Q54LFpGsiH86Yaabfu43LuvCv_vKVHj0=)")
        st.write('Toxic comments can take many different forms, such as:')
        st.write(
            '- Comments that are **toxic**: containing rude, disrespectful, or insulting language')
        st.write(
            '- Comments that are **severely toxic**: containing extremely offensive or abusive language')
        st.write(
            '- Comments that are **obscene**: containing vulgar, profane, or sexually explicit language')
    with col2:
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write('')
        st.write(
            '- Comments that contain **threats**: expressing an intention to harm someone')
        st.write(
            '- Comments that are **insulting**: containing language that is intended to offend or belittle someone')
        st.write(
            '- Comments that express **identity hate**: containing language that is derogatory or discriminatory towards a particular group of people based on their race, religion, gender, or other personal characteristics')
        st.write('')

    st.subheader('Interactive Toxic Comment Detector')

    # Add a text box for user input
    comment = st.text_input('Enter your comment here:')

    # Add a submit button
    submit_button = st.button('Submit')

    if submit_button:
        # Load the model
        #model = tf.keras.models.load_model('model')

        # Create a dataframe from the user input
        # user_input = pd.DataFrame({'comment_text': [comment]})

        # Make predictions
        predictions = model.predict([comment])

        # Create a dataframe with the predictions
       # df = pd.DataFrame({'Toxic': predictions[0][0]}, index=[0])
        df = pd.DataFrame({'Toxic': 1,
                           'Severely Toxic': 2,
                           'Obscene': 3,
                           'Threat': 4,
                           'Insult': 5,
                           'Identity Hate': 6}, index=[0])

        # Display the predictions
        st.write('')
        st.subheader('Predictions')
        st.write(df)

        # Display the comment
        st.write('')
        st.subheader('Comment')
        st.write(comment)

        # Display the toxicity score
        st.write('')
        st.subheader('Toxicity Score')
        st.write(np.sum(predictions))


if __name__ == "__main__":
    run()