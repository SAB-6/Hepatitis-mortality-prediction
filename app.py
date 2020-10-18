import numpy as np
import joblib
import pandas as pd
import streamlit as st 


model = joblib.load("./model/model.pkl")

def predict_hepatitis():
    """
    Predicting hepatitis
    Specifies the features to make prediction under the parameters.
    ---
    parameters: 
      - name: age
        in: query
        type: number
        required: true
      - name: sex
        in: query
        type: number
        required: true
      - name: steroid
        in: query
        type: number
        required: true
      - name: antivirals
        in: query
        type: number
        required: true
      - name: fatigue'
        in: query
        type: number
        required: true
      - name: malaise
        in: query
        type: number
        required: true
      - name: anorexia
        in: query
        type: number
        required: true
      - name: liver_big
        in: query
        type: number
        required: true
      - name: liver_firm'
        in: query
        type: number
        required: true
      - name: spleen_palpable
        in: query
        type: number
        required: true
      - name: spiders
        in: query
        type: number
        required: true
      - name: ascites
        in: query
        type: number
        required: true
        - name: varices
        in: query
        type: number
        required: true
      - name: bilirubin
        in: query
        type: number
        required: true
      - name: alk_phosphate
        in: query
        type: number
        required: true
      - name: sgot
        in: query
        type: number
        required: true
      - name: albumin
        in: query
        type: number
        required: true
      - name: protime
        in: query
        type: number
        required: true
      - name: histology
        in: query
        type: number
        required: true
     """ 
    try:
        prediction= model.predict(np.array[[age, sex, steroid, antivirals, 
            fatigue, malaise, anorexia, liver_big, liver_firm, spleen_palpable,
            spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin,
            protime, histology]])
        return "Hello The answer is "+str(prediction[0])
    except:
        return "All variable should be numeric (male: 1, female 2)"           


def main():
    st.title("Hepatitis Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Hepatitis Predictor</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    age = st.text_input("Age","Type Here")
    sex = st.text_input("Sex","Type Here")
    steroid = st.text_input("Steroid","Type Here")
    antivirals = st.text_input("Antivirals","Type Here")
    fatigue = st.text_input("Fatigue","Type Here")
    malaise = st.text_input("Malaise","Type Here")
    anorexia = st.text_input("Anorexia","Type Here")
    liver_big = st.text_input("Liver_big","Type Here")
    liver_firm = st.text_input("Liver_firm","Type Here")
    spleen_palpable = st.text_input("Spleen_palpable","Type Here")
    spiders = st.text_input("Spiders","Type Here")
    ascites = st.text_input("Ascites","Type Here")
    varices = st.text_input("Varices","Type Here")
    bilirubin = st.text_input("Bilirubin","Type Here")
    alk_phosphate = st.text_input("Alk_phosphate","Type Here")
    sgot = st.text_input("Sgot","Type Here")
    albumin = st.text_input("Albumin","Type Here")
    protime = st.text_input("Protime","Type Here")
    histology = st.text_input("Histology","Type Here")
    
    output = ""
    if st.button("Predict"):
        output = predict_hepatitis(age, sex, steroid, antivirals, 
                    fatigue, malaise, anorexia, liver_big, liver_firm, spleen_palpable,
                    spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin,
                    protime, histology)
    st.success(f"The prediction is: {output}")
    if st.button("About"):
        st.text("Predich Hepatitis")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()