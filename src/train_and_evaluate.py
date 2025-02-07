import os
import yaml
import pandas as pd
import numpy as np
import argparse
from pkgutil import get_data
from get_data import get_data, read_param
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error , mean_squared_error
from sklearn.linear_model import ElasticNet
import joblib
import json
from urllib.parse import urlparse

def train_and_evaluate(config_path):
    # config = read_param(config_path)
    # #df = get_data(config_path)
    # train_data_path = config["split_data"]["train_data"]
    # test_data_path = config["split_data"]["test_data"]
    # raw_data_path = config["load_data"]["clean_data"]
    # split_data = config["split_data"]["test_size"]
    # random_state = config["base"]["random_state"]
    # df = pd.read_csv(raw_data_path, sep=",")
    # model_dir=config['model_path']


    # alpha=config['estimator']['ElasticNet']['params']['alpha']
    # l1_ratio=config['estimator']['ElasticNet']['params']['L1_ratio']

    # target=config['base']['target_col']
    # train=pd.read_csv('train_data_path')
    # test=pd.read_csv('test_data_path')



    config = read_param(config_path)
    train_data_path = config["split_data"]["train_data"]
    test_data_path = config["split_data"]["test_data"]
    raw_data_path = config["load_data"]["clean_data"]
    split_data = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]
    df = pd.read_csv(raw_data_path, sep=",")
    model_dir = config["model_path"]

    alpha = config["estimator"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimator"]["ElasticNet"]["params"]["L1_ratio"]

    target = config["base"]["target_col"]
    train = pd.read_csv("train_data_path")
    test = pd.read_csv("test_data_path")

    train_y=train[target]
    test_y=test[target]

    train_x=train.drop(target , axis=1)
    test_x=train.drop(target ,axis=1)

    lr=ElasticNet(alpha=alpha,l1_ratio=L1_ratio ,random_state=random_state)
    lr.fit(train_x,train_y)
    predicted_quality=lr.predict(test_x)

    (rmse,mae,r2) = eval_metrics(test_y,predicted_quality)
    print('Elastic model (alpha=%f , l1_ratio=%f):'%(alpha,l1_ratio))






if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yml")
    parsed_args= args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)