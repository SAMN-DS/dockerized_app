"""
Load a model to predict churning
"""
import numpy as np
import pickle
from flask import Flask
from flask import request
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__)


@app.route('/predict_churn')
def predict_churn():
    clf = read_model('churn_model.pkl')
    is_male = request.args.get('is_male')
    num_inters = request.args.get('num_inters')
    late_on_payment = request.args.get('late_on_payment')
    age = request.args.get('age')
    years_in_contract = request.args.get('years_in_contract')

    d = {'is_male': [is_male], 'num_inters': [num_inters], 'late_on_payment': [late_on_payment], 'age': [age],
         'years_in_contract': [years_in_contract]}
    X = pd.DataFrame(d)
    y = clf.predict(X)

    return str(y[0])


def read_model(filename):
    with open(filename, 'rb') as f:
        lr = pickle.load(f)
    return lr


def main():
    app.run(host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
