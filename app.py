from flask import Flask,render_template,request
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)


# Load the regression  model
filename = 'multiple_regression_model.pkl'
ml_model = pickle.load(open(filename, 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict",methods=['GET','POST'])
def predict():
    
    temp_array=list()

    if request.method=="POST":
        try:

            city = request.form['city']

            if city=="NewYork":
                temp_array=temp_array + [1,0,0]

            elif city=="Florida":
                temp_array=temp_array + [0,1,0]

            elif city=="California":
                temp_array=temp_array + [0,0,1]
            
                
            #NewYork=float(request.form['NewYork'])
            
            #Florida=float(request.form['Florida'])
            
            #California=float(request.form['California'])
            
            
            
            RnD_Spend=float(request.form['RnD_Spend'])
            
            Admin_Spend=float(request.form['Admin_Spend'])
            
            Market_Spend=float(request.form['Market_Spend'])

            temp_array = temp_array + [RnD_Spend,Admin_Spend,Market_Spend]

            #pred_args=[NewYork,Florida,California,RnD_Spend,Admin_Spend,Market_Spend]

            pred_args_arr=np.array([temp_array])

            pred_args_arr=pred_args_arr.reshape(1,-1)

            model_prediction =ml_model.predict(pred_args_arr)

            model_prediction =round(float(model_prediction),2)
            

        except valueError:

            return "Please check if the values are entered correctly"


            
    return render_template('predict.html',prediction=model_prediction)

if __name__=="__main__":
    app.run()
