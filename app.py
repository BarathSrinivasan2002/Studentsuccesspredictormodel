from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

app = Flask(__name__)

# # Dictionary mapping model names to their corresponding pickle files
# models = {
#     "Neural Network": "nn.pkl"
# }

# Load the pipeline
preprocessor = joblib.load("preprocessor.pkl")

cols = ['First_Term_Gpa', 'Second_Term_Gpa', 'First_Language', 'Funding', 'FastTrack','Residency','Gender','Prev_Education','Age_Group','English_Grade']

# route the app
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    try:
        print("Received form submission")
        
        # Retrieve form data
        data = request.json
        firstTermGpa = np.array([data['firstTermGpa']])
        print("firstTermGpa: {firstTermGpa}")
        secondTermGpa = np.array([data['secondTermGpa']])
        firstLanguage = np.array([data['firstLanguage']])
        funding = np.array([data['funding']])
        FastTrack = np.array([data['fastTrack']])
        Residency = np.array([data['residency']])
        print("Residency: {Residency}")

        Gender = np.array([data['gender']])
        Prev_Education = np.array([data['prevEducation']])
        Age_Group = np.array([data['ageGroup']])
        English_Grade = np.array([data['englishGrade']])
        
        

        model = joblib.load("nn.pkl")
        print("Received Data:", data)

        
        # Concatenate form data
        final = np.concatenate([firstTermGpa, secondTermGpa, firstLanguage, funding, FastTrack,Residency,Gender,Prev_Education,Age_Group,English_Grade])
        
        final = np.array(final)
        data = pd.DataFrame([final], columns=cols)
        print(data)
        data_trans = preprocessor.transform(data)
        data_reshaped = data_trans.reshape(1, -1)
        prediction = model.predict(data_reshaped)
        result = prediction[0]*100
        result = float(result)
        print(result)
        return jsonify({'score': result , 'message': 'Prediction complete!'})
    
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({'error': 'Prediction failed'}), 500
    

if __name__ == "__main__":
    app.run(debug=True, port=5000)