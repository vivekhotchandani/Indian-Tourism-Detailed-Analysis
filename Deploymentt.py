import streamlit as st
import numpy as np
import pandas as pd
import joblib
from imblearn.over_sampling import SMOTE
from streamlit.components.v1 import components
from streamlit_option_menu import option_menu

with st.sidebar:        
            app = option_menu(
                menu_title='Main Menu',
                options=['About','Dashboard','Sentiment-Analysis','Contact'],
                icons=['info-circle-fill','chat-fill','chat-fill','person-circle'],
                menu_icon='chat-text-fill',
                default_index=1
                
            )
            
if app == "About":
    st.write("This Project Aims to give you a strong analysis of Tourists Visiting India and Sentiment Analysis of the Google Reviews.")
if app == "Dashboard":
    st.header(" ")    
if app == "Sentiment-Analysis":
    st.header("Google Reviews: Sentiment Analysis")
    Review = st.text_input("Enter a Review")
    # if st.button("Analyse"):
    #     st.write(")
    # URL of your default background image
    background_image_url = "https://raw.githubusercontent.com/vivekhotchandani/Indian-Tourism-Detailed-Analysis/main/pexels-chee-huey-wong-62348%20(1).png"

    # Set the background image using custom CSS
    def set_default_background_image():
        background_style = f"""
        <style>
        .stApp {{
            background-image: url('{background_image_url}');
            background-size: cover;
        }}
        </style>
        """
        st.markdown(background_style, unsafe_allow_html=True)

    # Set the default background image
    set_default_background_image()

    #load the csv file 
    # df=pd.read_csv("C:\Users\Symbiosis\Documents\DATASET-3RD SEM\Google_Reviews.xlsx")
    tdidf = joblib.load("tfidf11 (2).sav")
    X_train_smote = joblib.load("X_train_smote (1).pkl")
    y_train_smote = joblib.load("y_train_smote (1).pkl")

    # Load the trained model
    model= joblib.load("random_forest_model (1).pkl")
    # if st.button("Analsye") :
    #     a= tdidf.transform(Review)
    #     st.write(a)
    #     model.predict(a)
    if st.button("Analyze"):
        # Ensure user entered a review
        if Review:
            # Transform the user input using the loaded TfidfVectorizer
            transformed_review = tdidf.transform([Review])

            # Predict sentiment using the loaded model
            sentiment = model.predict(transformed_review)[0]

            # Display the sentiment prediction
            st.write("Sentiment: ", sentiment)
        else:
            st.warning("Please enter a review before analyzing.")
                
