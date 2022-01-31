from typing import Tuple
import numpy as np


class Model:
    # Modify your model, default is a linear regression model with random weights
    ID_DICT = {"MINH LE NGUYEN": "", "BU_ID": "U02407412", "BU_EMAIL": "minhng99@bu.edu"}

    def __init__(self):
        self.theta = None

    def preprocess(self, X: np.array, y: np.array) -> Tuple[np.array, np.array]:
        ###############################################
        ####      add preprocessing code here      ####
        ###############################################
        return X, y

    def train(self, X_train: np.array, y_train: np.array):
        """
        Train model with training data
        """
        ###############################################
        ####   initialize and train your model     ####
        ###############################################
        #X_train = np.vstack((np.ones((X_train.shape[0],)), X_train.T)).T
        #self.theta = np.random.rand(X_train.shape[1], 1)
        L = 3
        #Normal Equations with Regularlization (X^T X + lambda*I)X^T y
        self.theta = np.linalg.inv(((X_train.T.dot(X_train)) + L * np.eye(X_train.shape[1]))).dot(X_train.T).dot(y_train)

    def predict(self, X_val: np.array) -> np.array:
        """
        Predict with model and given feature
        """
        ###############################################
        ####      add model prediction code here   ####
        ###############################################
        #X_val = np.vstack((np.ones((X_val.shape[0],)), X_val.T)).T
        return np.dot(X_val, self.theta)
