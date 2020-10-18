# Import required libraries
import joblib
import numpy as np
from flask import Flask, request
import flasgger
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

# load model
model = joblib.load('./model/model.pkl')

# web page that handles user's query and displays model results
@app.route('/')
def home():
    return "A web app for predicting hepatitis"


@app.route('/predict', methods=['Get'])
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
    age = request.args.get("age")
    sex = request.args.get("sex")
    steroid = request.args.get("steroid")
    antivirals = request.args.get("antivirals")
    fatigue =request.args.get("fatigue")
    malaise = request.args.get("malaise")
    anorexia = request.args.get("anorexia")
    liver_big = request.args.get("liver_big")
    liver_firm = request.args.get("liver_firm")
    spleen_palpable = request.args.get("spleen_palpable")
    spiders = request.args.get("spiders")
    ascites = request.args.get("ascites")
    varices=request.args.get("varices")
    bilirubin = request.args.get("bilirubin")
    alk_phosphate =request.args.get("alk_phosphate")
    sgot = request.args.get("sgot")
    albumin = request.args.get("albumin")
    protime =request.args.get("protime")
    histology = request.args.get("histology")
    
    try:
      prediction= model.predict(np.array[[age, sex, steroid, antivirals, 
        fatigue, malaise, anorexia, liver_big, liver_firm, spleen_palpable,
        spiders, ascites, varices, bilirubin, alk_phosphate, sgot, albumin,
        protime, histology]])
      return "The prediction is: "+str(prediction[0])
    except:
      return "All variable should be numeric (male: 1, female 2)"                   


def main():
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    main()