from flask import Flask
import flask_monitoringdashboard as dashboard
from model import *

app = Flask(__name__)
dashboard.bind(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/id/<int:customer_id>")
def get_customer_segment(customer_id):

    global scaler, customer_history_df, clusterer

    if customer_id < customer_history_df.CustomerID.min() or customer_id > customer_history_df.CustomerID.max():
        raise ValueError('CustomerID should be within admissible range') 

    log_features = ['amount_log',  'recency_log',  'frequency_log']
    X = customer_history_df[customer_history_df.CustomerID==customer_id][log_features]
    X_scaled = scaler.transform(X)
    return str(clusterer.predict(X_scaled)[0])