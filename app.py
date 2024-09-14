from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib  # If you're using a pre-trained model
from sklearn.impute import SimpleImputer

app = Flask(__name__)

imputer = SimpleImputer(strategy='mean')

# Load your pre-trained model (replace with your model path)
import os
import pandas as pd

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Create the path to the dataset
file_path = os.path.join(script_dir, 'model.pkl')

# Load the dataset
model = joblib.load(file_path)

# Function to preprocess input data
# Function to preprocess input data
def preprocess_input(data_):
    
    data = pd.DataFrame([data_])
    # Mapping input features to numeric values
    education_mapping = {1: 1, 2: 2, 3: 3, 4: 4}  # Already numerical in form
    gender_mapping = {1: 1, 2: 2, 3: 3}  # Already numerical
    race_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}  # Already numerical
    married_mapping = {1: 1, 2: 2, 3: 3}
    age_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}  # Already numerical
    familysize_mapping = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    
    Q1 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q4 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q8 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q11 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q16 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q22 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q29 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q34 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q38 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q40 = {1: 1, 2: 2, 3: 3, 4: 4}
    Q42 = {1: 1, 2: 2, 3: 3, 4: 4}
    
    TIPI1 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI2 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI3 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI4 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI5 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI6 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI7 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI8 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI9 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}
    TIPI10 = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7}

    # Encode the categorical variables to numerical
    data["Education"] = education_mapping.get(data_["Education"])
    data["Gender"] = gender_mapping.get(data_["Gender"])
    data["Race"] = race_mapping.get(data_["Race"])
    data["Married"] = married_mapping.get(data_["Married"])
    data["Age"] = age_mapping.get(data_["Age"])
    data["Familysize"] = familysize_mapping.get(data_["Familysize"])
    
    data["Q1"] = Q1.get(data_["Q1"])
    data["Q4"] = Q4.get(data_["Q4"])
    data["Q8"] = Q8.get(data_["Q8"])
    data["Q11"] = Q11.get(data_["Q11"])
    data["Q16"] = Q16.get(data_["Q16"])
    data["Q22"] = Q22.get(data_["Q22"])
    data["Q29"] = Q29.get(data_["Q29"])
    data["Q34"] = Q34.get(data_["Q34"])
    data["Q38"] = Q38.get(data_["Q38"])
    data["Q40"] = Q40.get(data_["Q40"])
    data["Q42"] = Q42.get(data_["Q42"])
    
    data["TIPI1"] = TIPI1.get(data_["TIPI1"])
    data["TIPI2"] = TIPI2.get(data_["TIPI2"])
    data["TIPI3"] = TIPI3.get(data_["TIPI3"])
    data["TIPI4"] = TIPI4.get(data_["TIPI4"])
    data["TIPI5"] = TIPI5.get(data_["TIPI5"])
    data["TIPI6"] = TIPI6.get(data_["TIPI6"])
    data["TIPI7"] = TIPI7.get(data_["TIPI7"])
    data["TIPI8"] = TIPI8.get(data_["TIPI8"])
    data["TIPI9"] = TIPI9.get(data_["TIPI9"])
    data["TIPI10"] = TIPI10.get(data_["TIPI10"])

    # Convert data to DataFrame (ensure it is numeric)
    imputed_data = imputer.fit_transform(data)
    
    

    return pd.DataFrame(imputed_data,columns=data.columns)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Extract data from the form
    data = {
        "Education": int(request.form["Education"]),
        "Gender": int(request.form["Gender"]),
        "Race": int(request.form["Race"]),
        "Married": int(request.form["Married"]),
        "Familysize": int(request.form["Familysize"]),
        "Age": int(request.form["Age"]),
        
        "Q1" : int(request.form["Q1"]),
        "Q4" : int(request.form["Q4"]),
        "Q8" : int(request.form["Q8"]),
        "Q11" : int(request.form["Q11"]),
        "Q16" : int(request.form["Q16"]),
        "Q22" : int(request.form["Q22"]),
        "Q29" : int(request.form["Q29"]),
        "Q34" : int(request.form["Q34"]),
        "Q38" : int(request.form["Q38"]),
        "Q40" : int(request.form["Q40"]),
        "Q42" : int(request.form["Q42"]),
        
        "TIPI1" : int(request.form["TIPI1"]),
        "TIPI2" : int(request.form["TIPI2"]),
        "TIPI3" : int(request.form["TIPI3"]),
        "TIPI4" : int(request.form["TIPI4"]),
        "TIPI5" : int(request.form["TIPI5"]),
        "TIPI6" : int(request.form["TIPI6"]),
        "TIPI7" : int(request.form["TIPI7"]),
        "TIPI8" : int(request.form["TIPI8"]),
        "TIPI9" : int(request.form["TIPI9"]),
        "TIPI10" : int(request.form["TIPI10"]),
        
    }
    processed_data = preprocess_input(data)
    # Preprocess the input data
    print(f"Shape of processed data: {processed_data.shape}")

    # Ensure it has 27 features
    if processed_data.shape[1] != 27:
        raise ValueError(f"Expected 27 features, but got {processed_data.shape[1]}")

    # Predict using the model
    prediction = model.predict(processed_data)

    return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
