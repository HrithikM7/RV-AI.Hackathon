import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array,array_to_img
from joblib import load
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from keras import backend as K
from PIL import Image
import pickle
from pathlib import Path
import streamlit_authenticator
import requests
import json
import pywhatkit
from datetime import datetime,timedelta

def get_user_data(api):
    response = requests.get(f"{api}")
    if response.status_code == 200:
        return response.json()

streamlit_style = """
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');
            html, body, [class*="css"]  {
            font-family: 'Georgia', sans-serif;
            }
            </style>
            """

st.markdown(streamlit_style, unsafe_allow_html=True)

#Authentication 

names = ['Hrithik Maddirala', 'Gadaputi Ashritha']
usernames = ['hrithikm2002' , 'gashritha']

fp = Path("C:\\Users\\Anil\\Downloads\\User_Interface\\hashed_passwords.pkl")

with fp.open("rb") as file : 
    hashed_passwords = pickle.load(file)

credentials = {
        "usernames":{
            usernames[0]:{
                "name":names[0],
                "password":hashed_passwords[0]
                         },
            usernames[1]:{
                "name":names[1],
                "password":hashed_passwords[1]
                         }            
                    }       
                }   

authenticator = streamlit_authenticator.Authenticate(credentials,"cookie", "abcdef", cookie_expiry_days= 1)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status == False:
    st.error("Username/password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status : 
    st.title("🍔 EMOTION-BASED FOOD RECOMMENDATION SYSTEM")
    menu = ["🏠 Home","🍞 Food","🌍 NGO Connectivity","🙍 Information"]

    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
    choice = st.sidebar.selectbox("MENU",menu)
    if choice == "🏠 Home":
        st.subheader("THE NEW AGE FOOD RECOMMENDATION")
        st.markdown("Rather than going through the trouble of recommending customers what to order, what about using an emotion-based food recommendation app which will recommend customers food based on how they are feeling? Welcome to the one-stop website for all your food needs!")
        image = Image.open('C:\\Users\\Anil\\Downloads\\User_Interface\\home_pic.jpg') 
        st.image(image,width=600)

    if choice == "🍞 Food":
        st.subheader("HELLO!")
        menu1 = ["Emotion-based Food Recommendation"] 
        ch = st.selectbox("Select an option",menu1)
        st.text("")
        if ch == "Emotion-based Food Recommendation":
            faceDetect=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            model1 = tf.keras.models.load_model("C:\\Users\\Anil\\Downloads\\User_Interface\\final_model_file.h5")
            model2 = tf.keras.models.load_model("C:\\Users\\Anil\\Downloads\\User_Interface\\gender_model.h5")
            model3 = tf.keras.models.load_model("C:\\Users\\Anil\\Downloads\\User_Interface\\age_model_.h5")
            ### load file
            uploaded_file = st.file_uploader("Choose an image file", type="jpg")
        
            map_dict_emotion = {0: 'Negative',
                    1: 'Neutral',
                    2: 'Positive'}
            map_dict_gender = {0: 'Male',
                    1: 'Male'}
            map_dict_age = {0: 'Between 0 and 20 years',
                    1: 'Between 20 and 45 years',
                    2: 'Above 45 years'}

            if uploaded_file is not None:
            # Convert the file to an opencv image.
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                print(file_bytes)
                opencv_image = cv2.imdecode(file_bytes, 1)
                print(opencv_image)
                opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                # resized = cv2.resize(opencv_image,(224,224))
                # Now do something with the image! For example, let's display it:
                st.image(opencv_image, channels="RGB")

                gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

                Generate_pred = st.button("Generate Prediction")    
                if Generate_pred:
                    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
                    x,y,w,h = faces[0]
                    # for x,y,w,h in faces[0][0]:
                    sub_face_img=gray[y:y+h, x:x+w]
                    resized=cv2.resize(sub_face_img,(48,48))
                    normalize=resized/255.0
                    reshaped=np.reshape(normalize, (1, 48, 48, 1))
                    result1=model1.predict(reshaped) #emotion
                    result2=model2.predict(reshaped) #gender
                    result3=model3.predict(reshaped) #age
                    label1=np.argmax(result1, axis=1)[0]
                    label2=np.argmax(result2, axis=1)[0]
                    label3=np.argmax(result3, axis=1)[0]
                    # prediction = model.predict(img_reshape).argmax()
                    st.title("Predicted emotion for the image is {}".format(map_dict_emotion[label1])) 
                    st.title("Predicted gender for the image is {}".format(map_dict_gender[label2]))
                    st.title("Predicted age for the image is {}".format(map_dict_age[label3]))   
                    st.text("")

    if choice=="🙍 Information":

        menu2 = ["Informative Plots", "FAQ's"] 
        ch = st.selectbox("Select an option",menu2)
        if ch=="Informative Plots":
            menu3 = ["Calories","Protein","Fat","Saturated Fat","Fiber","Carbohydrates"] 
            ch2 = st.selectbox("Select an option",menu3)
            st.text("")
            df=pd.read_csv("C:\\Users\\Anil\\Downloads\\User_Interface\\final_dataset.csv")
            
            if ch2 == "Calories":
                df1 = df.sort_values('Actual_Calories',ascending = False)[0:20]
                fig1=px.bar(df1,x='Food',y='Actual_Calories')
                st.plotly_chart(fig1)
            if ch2 == "Protein":
                df2 = df.sort_values('Actual_Protein',ascending = False)[3:22]
                fig2=px.bar(df2,x='Food',y='Actual_Protein')
                st.plotly_chart(fig2)
            if ch2 == "Fat":
                df3 = df.sort_values('Actual_Fat',ascending = False)[9:20]
                fig3=px.bar(df3,x='Food',y='Actual_Fat')
                st.plotly_chart(fig3)
            if ch2 == "Saturated Fat":
                df4 = df.sort_values('Actual_Sat.Fat',ascending = False)[3:14]
                fig4=px.bar(df4,x='Food',y='Actual_Sat.Fat')
                st.plotly_chart(fig4)
            if ch2 == "Fiber":
                df5 = df.sort_values('Actual_Fiber',ascending = False)[3:14]
                fig5=px.bar(df5,x='Food',y='Actual_Fiber')
                st.plotly_chart(fig5)
            if ch2 == "Carbohydrates":
                df6 = df.sort_values('Actual_Carbs',ascending = False)[5:16]
                fig6=px.bar(df6,x='Food',y='Actual_Carbs')
                st.plotly_chart(fig6)

    if choice=="🌍 NGO Connectivity":
        place = st.text_input("Enter the address of the restaurant")
        st.markdown(place)
        if place is not None:
            print(place)
            str1 = "https://atlas.microsoft.com/search/address/json?subscription-key=BBda83GkjJ--03W4DN-VqgzlkSfqxdfzZfbhAGL-Zyg&api-version=1.0&query="
            str2 = str1+place
            coordinates = get_user_data(str2) 
            latitude = coordinates['results'][0]['position']['lat']
            longitude = coordinates['results'][0]['position']['lon']
            # print(coordinates['results'][0]['position'])
            dicto = get_user_data("https://atlas.microsoft.com/search/poi/category/json?subscription-key=BBda83GkjJ--03W4DN-VqgzlkSfqxdfzZfbhAGL-Zyg&api-version=1.0&query=NON_GOVERNMENTAL_ORGANIZATION&limit=30&lat="+str(latitude)+"&lon="+str(longitude))
            l = []
            key = 'phone'
            for i in range(0,30):
                if key in dicto['results'][i]['poi']:
                #print(dict['results'][i]['poi']['name'])
                    l.append({dicto['results'][i]['poi']['name'] : [dicto['results'][i]['poi']['phone'], dicto['results'][i]['dist']]})

            for Ngo in l:
                k = []
                v = []
                d = []
                for Ngo in l:
                    k.append(list(Ngo.keys()))
                    v.append(list((Ngo.values()))[0][0])
                    d.append(list((Ngo.values()))[0][1])

            for i in range(min(len(l),3)):
                st.markdown(str(k[i][0])+" : "+str(v[i])+"   Distance : "+str(d[i])+" meters")

            now = datetime.now()+timedelta(minutes=1.2)
            #pywhatkit.sendwhatmsg('+91 9481634956',"Hi this is "+str('John'),now.hour,now.minute,15,True,3)
            pywhatkit.sendwhatmsg('+91 9481634956',"Hi. We have food remaining and would like to donate it today. Please respond to initiate further communication.",now.hour,now.minute,15,True,3)
    
    if choice=="FAQ's":

        if st.button('Is there any connection between food and mood?'):
            st.markdown('There have been many studies conducted which state that the food we eat influences our mental health.')
            
        if st.button('How do we use the recommendation system?'):
            st.markdown("Select the Food option on the sidebar. Upload a photo of the customer, and based on the customer's mood & other details like customer allergies,etc ,few food items will be recommended for them.")
        image_faq = Image.open("C:\\Users\\Anil\\Downloads\\User_Interface\\FAQ.jpg") 
        st.image(image_faq,width=650)  
               

        
