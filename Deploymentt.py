import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import joblib
from imblearn.over_sampling import SMOTE
from streamlit.components.v1 import components
from streamlit_option_menu import option_menu

df_long = pd.read_csv(r"C:\Users\Symbiosis\Downloads\df_long.csv")
def forecast_arima_country(data, country):
    country_data = data[data['Country'] == country]
    
    if len(country_data) < 4:
        st.write(f"Insufficient data for {country}. Cannot forecast.")
        return
    
    model = sm.tsa.arima.ARIMA(country_data['Value'], order=(3, 0, 2))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=4)
    
    return forecast

with st.sidebar:        
    app = option_menu(
        menu_title='Main Menu',
        options=['About','Journey into the Future','Dashboard','Sentiment-Analysis','Contact'],
        icons=['info-circle-fill','calendar-week','map','chat-fill','person-circle'],
        menu_icon='chat-text-fill',
        default_index=0,
        key="unique_key_1"
    )


            
            
if app == "About":
    background_image_url = "https://img.freepik.com/free-photo/christmas-travel-concept-with-laptop_23-2149573078.jpg?w=1800&t=st=1698301195~exp=1698301795~hmac=ae0727b559539d2b68ebb96091659bef48f818f8d4bc520c476f36734cb918e1"

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

    st.markdown('## About the Project')
    st.header("This Project Aims to give:")
    st.markdown('* A strong analysis of Tourists Visiting India')
    st.markdown('* Sentiment Analysis of the Google Reviews')
    st.markdown('* Develop a comprehensive understanding of public sentiment towards these places')
    st.markdown('* Classifying the reviews into three categories: positive, negative, and neutral.')
    table_data = [
    ("1.Data Collection:", "Scraping Google Reviews: Gathered data by scraping Google reviews of places in various cities across India. This involved extracting textual reviews, star ratings, and other relevant metadata."),
    ("2.Data Preprocessing:", "Cleaned the raw data, removing irrelevant characters, HTML tags, and noise to prepare it for analysis"),
    ("3.Sentiment Analysis:", "Feature Extraction: Converted text data into features suitable for machine learning models, often using techniques like TF-IDF (Term Frequency-Inverse Document Frequency), Model Training: Trained the selected algorithm on the preprocessed data to classify reviews into positive, negative, and neutral sentiments."),
    ("4.ARIMA Modelling:","To predict the Country wise next quarter wise data for the Tourists coming yo India, applied Arima Modelling ,got the best fit(3,0,2) and used it to train the data and derive the results."),
    ("5.Future Scope:", "Future Work: Outlined potential future research directions, including exploring advanced sentiment analysis techniques, expanding the dataset to include more cities, or integrating social media data for a more comprehensive analysis.")

]

    # Create a list of dictionaries for the table
    table_rows = [{"Component": component, "Description": description} for component, description in table_data]

    # Display the table with headings
    st.table(table_rows)

if app =="Journey into the Future":
    selected_country = st.selectbox("Choose a country:", df_long['Country'].unique())
    predictions = forecast_arima_country(df_long, selected_country)
    
# Display results on the main page
    st.title(f"ARIMA Forecast for {selected_country}")
    if predictions is not None:
        st.write(f"Forecasted values for the next 4 quarters:")
        for i, val in enumerate(predictions):
            st.write(f"Quarter {i+1}: {val:.2f}")
    quarters = ["Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"]
    predicted_values = predictions

    data = pd.DataFrame({"Quarter": quarters, "Predicted Value": predicted_values})

    # Generate histogram
    st.write("### Histogram")
    fig, ax = plt.subplots()
    data.plot(kind='bar', x='Quarter', y='Predicted Value', ax=ax)
    ax.set_xlabel('Quarter')
    ax.set_ylabel('Predicted Value')
    st.pyplot(fig)
            
            
if app == "Dashboard":
    background_image_url = "https://raw.githubusercontent.com/vivekhotchandani/Indian-Tourism-Detailed-Analysis/main/Screenshot%202023-11-01%20095648.png"

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

    

       
if app == "Sentiment-Analysis":
    st.header("Google Reviews: Sentiment Analysis")
    Review = st.text_input("Enter a Review")
    
    # if st.button("Analyse"):
    #     st.write(")
    # URL of your default background image
    background_image_url = "https://img.freepik.com/free-photo/views-entrance-indian-temple_119635-5.jpg?w=1800&t=st=1698301087~exp=1698301687~hmac=e681216e857e27743f622d2a39946640079fb7434b983106b081088154cb368a"

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
    tdidf = joblib.load(r"C:\Users\Symbiosis\Downloads\tfidf11 (2).sav")
    X_train_smote = joblib.load(r"C:\Users\Symbiosis\Downloads\X_train_smote (1).pkl")
    y_train_smote = joblib.load(r"C:\Users\Symbiosis\Downloads\y_train_smote (1).pkl")

    # Load the trained model
    model= joblib.load(r"C:\Users\Symbiosis\Downloads\random_forest_model (1).pkl")
    # if st.button("Analsye") :
    #     a= tdidf.transform(Review)
    #     st.write(a)
    #     model.predict(a)
    import time
    if st.button("Analyze"):
        if Review:
            progress_bar = st.progress(0)

            for i in range(10):
                progress_bar.progress((i + 1)*10)
                time.sleep(0.05)

            # Perform sentiment analysis on the input review
            # (Assuming you have loaded the 'tdidf' and 'model' objects beforehand)
            transformed_review = tdidf.transform([Review])
            sentiment = model.predict(transformed_review)[0]

            # Display the sentiment prediction
            st.write("Sentiment: ", sentiment)
        else:
            st.warning("Please enter a review before analyzing.")
if app=='Contact':
    background_image_url = "https://img.freepik.com/free-photo/architecture-color-holy-beautiful-detail_1203-6106.jpg?w=1800&t=st=1698300607~exp=1698301207~hmac=2dcbb92f4eb1c69f462a57b0b6b5fe91cc47c571e1f26f1d962e7f206c4b83ff"

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
    st.write("<h1 style='color: black;'>Our Mentor</h1>", unsafe_allow_html=True)
    # st.markdown("<font color='black'>Dr. Harsh Dhiman</font>", unsafe_allow_html=True)
    # st.markdown('\n<a href="https://www.linkedin.com/in/harshdhiman/" target="_blank">LinkedIn Profile</a>', unsafe_allow_html=True)
    # Define the faculty members and their LinkedIn profiles


    faculty_data = [
        {"name": "Dr. Smita Mahajan", "linkedin": "https://www.linkedin.com/in/dr-smita-mahajan-b8a319128/"},
        {"name": "Dr. Pooja Kamat", "linkedin": "https://www.linkedin.com/in/pooja-kamat-19603158/"},
        {"name": "Dr. Harsh Dhiman", "linkedin": "https://www.linkedin.com/in/harshdhiman/"}
    ]

    # Display the faculty members side by side
    for faculty in faculty_data:
        st.markdown(f"<div style='display:inline-block; margin:10px;'><font color='black'>{faculty['name']}</font><br><a href='{faculty['linkedin']}' target='_blank'>LinkedIn Profile</a></div>", unsafe_allow_html=True)


    # Display the mentors side by side

    st.header("For Further Queries! Please contact")

    st.markdown("<font color='black'>* Vivek Hotchandani</font>", unsafe_allow_html=True)
    st.markdown('\n<a href="https://www.linkedin.com/in/vivek-hotchandani-853804249/" target="_blank">Visit My LinkedIn Profile</a>', unsafe_allow_html=True)

    st.markdown("<font color='black'>* Akshat Jain</font>", unsafe_allow_html=True)
    st.markdown('\n<a href="https://www.linkedin.com/in/akshat-jain-4b6721260/" target="_blank">Visit My LinkedIn Profile</a>', unsafe_allow_html=True)

    st.markdown("<font color='black'>* Vanshika Tyagi</font>", unsafe_allow_html=True)
    st.markdown('\n<a href="https://www.linkedin.com/in/vanshika-tyagi-745856263/" target="_blank">Visit My LinkedIn Profile</a>', unsafe_allow_html=True)

    st.markdown("<font color='black'>* Vanshika Tyagi</font>", unsafe_allow_html=True)
    st.markdown('\n<a href="https://www.linkedin.com/in/vanshika-tyagi-745856263/" target="_blank">Visit My LinkedIn Profile</a>', unsafe_allow_html=True)

    st.write("Special Credits")

    st.markdown("<font color='black'>* Sahil Goyal</font>", unsafe_allow_html=True)
    st.markdown('\n<a href="https://www.linkedin.com/in/sahil-goyal-1a731124a/" target="_blank">Visit My LinkedIn Profile</a>', unsafe_allow_html=True)
