# Import required 
import joblib
from flask import Flask, render_template, request
import numpy as np

# load model
model = joblib.load('./model/model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(float(request.form['age']))
        sex = int(float(request.form['sex']))
        steroid = int(float(request.form['steroid']))
        antivirals = int(float(request.form['antivirals']))
        fatigue = int(float(request.form['fatigue']))
        malaise = int(float(request.form['malaise']))
        anorexia = int(float(request.form['anorexia']))
        liver_big = int(float(request.form['liver_big']))
        liver_firm = int(float(request.form['liver_firm']))
        spleen_palpable = int(float(request.form['spleen_palpable']))
        spiders = int(float(request.form['spiders']))
        ascites = int(float(request.form['ascites']))
        varices = int(float(request.form['varices']))
        bilirubin = float(request.form['bilirubin'])
        alk_phosphate = float(request.form['alk_phosphate'])
        sgot = int(float(request.form['sgot']))
        albumin = float(request.form['albumin'])
        protime = int(float(request.form['protime']))
        histology = int(float(request.form['histology']))  
            
        data = np.array([[age, sex, steroid, antivirals, fatigue,  malaise, anorexia, liver_big,
                            liver_firm, spleen_palpable, spiders, ascites,varices, bilirubin, alk_phosphate,sgot, 
                            albumin, protime, histology]])
        pred = model.predict(data)
        return render_template('results.html', prediction = pred)

if __name__ == '__main__':
	app.run(debug=True)
