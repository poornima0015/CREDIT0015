import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
import pickle

app = Flask(__name__)
app.config["DEBUG"]=True
api = Api(app)

a= pickle.load(open('decisiontree.pkl','rb'))
b= pickle.load(open('logisticreg.pkl','rb'))
c= pickle.load(open('randomforest.pkl','rb'))
d= pickle.load(open('knn.pkl','rb'))

class MakePrediction(Resource):
    @app.route('/predict', methods=['GET'])
    def get():
        
        avg_amt = request.args['Average Amount/transaction/day']
        trans_amt = request.args['Transaction_amount']
        is_dec = request.args['Is declined']
        total_dec = request.args['Total Number of declines/day']
        is_foreign = request.args['isForeignTransaction']
        is_highrisk = request.args['isHighRiskCountry']
        daily_chargeback = request.args['Daily_chargeback_avg_amt']
        six_chbk = request.args['6_month_avg_chbk_amt']
        six_freq= request.args['6-month_chbk_freq']

        avg_amt1=float(avg_amt)
        trans_amt1=float(trans_amt)
        is_dec1=float(is_dec)
        total_dec1=float(total_dec)
        is_foreign1=float(is_foreign)
        is_highrisk1=float(is_highrisk)
        daily_chargeback1=float(daily_chargeback)
        six_chbk1=float(six_chbk)
        six_freq1=float(six_freq)

        p = a.predict([[avg_amt,trans_amt,is_dec,total_dec,is_foreign,is_highrisk,daily_chargeback,six_chbk,six_freq]])[0]
         
        q = b.predict([[avg_amt1,trans_amt1,is_dec1,total_dec1,is_foreign1,is_highrisk1,daily_chargeback1,six_chbk1,six_freq1]])[0]
        r = c.predict([[avg_amt,trans_amt,is_dec,total_dec,is_foreign,is_highrisk,daily_chargeback,six_chbk,six_freq]])[0]
        s = d.predict([[avg_amt,trans_amt,is_dec,total_dec,is_foreign,is_highrisk,daily_chargeback,six_chbk,six_freq ]])[0]

        if p == 0:
            predicted_class = 'no'
        else:
            predicted_class = 'yes'
        if q == 0:
            predicted_class1 = 'no'
        else:
            predicted_class1 = 'yes'
        if r == 0:
            predicted_class2 = 'no'
        else:
            predicted_class2 = 'yes'
        if s == 0:
            predicted_class3 = 'no'
        else:
            predicted_class3 = 'yes'
        return jsonify({
            'Prediction for decision tree': predicted_class,
            'Prediction for logisticreg': predicted_class1,
            'Prediction for randomforest': predicted_class2,
            'Prediction for knn': predicted_class3
        })


if __name__ == '__main__':
    app.run()

