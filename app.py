'''from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))



@app.route('/')
def hello_world():
    return render_template("LoanPrediction.html")'''

'''def conversion(int_features=None):
  if int_features[0]=="Male":
    int_features[0]=1
  else:
    int_features[0]=0

  if int_features[1]=="Yes":
    int_features[1]=1
  else:
    int_features[1]=0

  if int_features[3]=="Graduate":
    int_features[3]=1
  else:
    int_features[3]=0

  if int_features[4]=="Yes":
    int_features[4]=1
  else:
    int_features[4]=0

  if int_features[10]=="Urban":
    int_features[10]=1
  elif int_features[10]=="Semiurban":
    int_features[10]=2
  else:
    int_features[10]=3
  return int_features'''

'''@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    # cv=conversion(int_features)
    # final = [np.array(cv)]
    # final = np.asarray(final, dtype='int')
    final = [np.asarray(int_features)]
    print(int_features)
    print(final)
    output=model.predict(final)
    # output='{0:.{1}f}'.format(prediction[0][1])

    if output==1:
        return render_template('LoanPrediction.html',pred="Congratulations!!! Your Loan Will Be Approved!")
    else:
        return render_template('LoanPrediction.html',pred="Sorry! Your Loan Cannot Be Approved!")


if __name__ == '__main__':
    app.run(debug=True)'''


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('LoanPrediction2.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1:
        return render_template('LoanPrediction2.html', prediction_text="Congratulations!!! Your Loan Will Be Approved!")
    else:
        return render_template('LoanPrediction2.html', prediction_text="Sorry! Your Loan Cannot Be Approved!")

    # return render_template('LoanPrediction2.html', prediction_text='Your loan will be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)