import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.externals import joblib
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(-1,1) 
    loaded_model = joblib.load("model.pkl") 
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict()         
        to_predict_list = list(to_predict_list.values())        
        list2 = []
        list2.append(to_predict_list[0])
        list2.append(to_predict_list[1])
        list2.append(to_predict_list[3])
        list2.append(to_predict_list[6])
        list2.append(to_predict_list[7])
        list2.append(to_predict_list[8])
        le = preprocessing.LabelEncoder()
        list2 = le.fit_transform(list2)       
        print(list2)
        to_predict_list[0] = list2[0]
        to_predict_list[1] = list2[1]
        to_predict_list[3] = list2[2]
        to_predict_list[6] = list2[3]
        to_predict_list[7] = list2[4]
        to_predict_list[8] = list2[5]
        print(to_predict_list)
        to_predict_list = list(map(int, to_predict_list)) 
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)         
        if int(result)== 1: 
            prediction ='Yes! You have chances of getting Higher Education.'
        else: 
            prediction ='No! You do not have chances of getting Higher Education.'            
        return render_template("result.html", prediction = prediction)

if __name__ == "__main__":
    app.run(debug=True)