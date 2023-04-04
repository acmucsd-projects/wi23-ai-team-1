import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Detecting Toxic Comments', page_icon=':guardsman:', layout='wide')

st.title('Detecting Toxic Comments')

# Add a sidebar
st.sidebar.subheader('Why is it important to be respectful towards others?')

st.sidebar.subheader('Building positive relationships')
st.sidebar.write('When we treat others with respect, kindness, and compassion, we are more likely to build positive and meaningful relationships with them.')

st.sidebar.subheader('Creating a safe and supportive environment')
st.sidebar.write('Toxic comments can hurt others and create a negative and unsafe environment. Being respectful towards others helps create a safe and supportive environment where everyone can thrive.')

st.sidebar.subheader('Fostering mutual understanding')
st.sidebar.write('Respectful behavior allows us to understand and appreciate different perspectives, backgrounds, and cultures, which helps us build bridges of mutual understanding.')

st.sidebar.subheader('Promoting personal growth')
st.sidebar.write('When we treat others with respect, we are also promoting our own personal growth. Respect helps us develop self-awareness, empathy, and compassion, which are essential for personal growth and development.')

st.write('')

col1, col2 = st.columns(2)
with col1:
    st.markdown('### Tags related to toxic comments')
    st.write('')
    st.write('Toxic comments can take many different forms, such as:')
    st.write('- Comments that are **toxic**: containing rude, disrespectful, or insulting language')
    st.write('- Comments that are **severely toxic**: containing extremely offensive or abusive language')
    st.write('- Comments that are **obscene**: containing vulgar, profane, or sexually explicit language')
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
    st.write('- Comments that contain **threats**: expressing an intention to harm someone')
    st.write('- Comments that are **insulting**: containing language that is intended to offend or belittle someone')
    st.write('- Comments that express **identity hate**: containing language that is derogatory or discriminatory towards a particular group of people based on their race, religion, gender, or other personal characteristics')
    st.write('')

st.subheader('Interactive Toxic Comment Detector')

# Add a text box for user input
comment = st.text_input('Enter your comment here:')

# Add a submit button
submit_button = st.button('Submit')
