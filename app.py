import streamlit as st
import pickle 

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

st.title("SMS Spam Detection Model")
st.write("*This is a Machine Learning application to classify sms as spam or ham.*")
input_sms = st.text_input("Enter the SMS")

if st.button('Predict'):
    if input_sms:
        data = [input_sms]
        vectorized_data = tk.transform(data).toarray()
        result = model.predict(vectorized_data)
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.write("Please enter a sms to classify.")
