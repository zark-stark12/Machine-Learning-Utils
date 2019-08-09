from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.base import ClassifierMixin, BaseEstimator, RegressorMixin
import numpy as np
import logging
import sys
import os
from copy import deepcopy
from math import sqrt

def RMSE(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2 ).mean())

def MAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)).mean()

def MSE(y_true, y_pred):
    return np.square(y_true - y_pred).mean()

def _setup_logging():
  """Sets up logging."""
  root_logger = logging.getLogger()
  root_logger_previous_handlers = list(root_logger.handlers)
  for h in root_logger_previous_handlers:
    root_logger.removeHandler(h)
  root_logger.setLevel(logging.INFO)
  root_logger.propagate = False

  # Set tf logging to avoid duplicate logging. If the handlers are not removed,
  # then we will have duplicate logging

  # Redirect INFO logs to stdout
  stdout_handler = logging.StreamHandler(sys.stdout)
  stdout_handler.setLevel(logging.DEBUG)
  root_logger.addHandler(stdout_handler)

  # Suppress C++ level warnings.
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class SupervisedLearner(BaseEstimator,RegressorMixin,ClassifierMixin):
    """
    Supervised Learning Class that allows for conducting and building Regressors or Classifiers and applying scores for each of the models. The models come from a list that you may generate from
    instantiating the class or by adding the model to the class object itself through the class method 'add_model'. The class was built with Sci-kit learn in mind meaning all the functions and
    model fitting/training/predicting work with Sci-kit model objects.

    These Sci-kit model objects can be the models themselves e.g. Linear Regression or any hyper-paramter tuning object such as GridSearchCV
    """
    classifier_scorers = (f1_score, accuracy_score, precision_score, recall_score)
    regressor_scorers = (mean_squared_error, mean_absolute_error, r2_score, explained_variance_score)

    def __init__(self, model_list=[], logs=True):

        if logs:
            _setup_logging()
            logging.info("Initializing Modeler object")
            logging.info("...............................................")

        if len(model_list) > 0:

            est = [model._estimator_type for model in model_list]

            if est.count(est[0]) == len(est):
                names = [str(model).split('(')[0] if str(model).split('(')[0] != "GridSearchCV" else str(model.estimator).split('(')[0] for model in model_list]
                self.models = dict(zip(names,model_list))
                self.estimator_type = list(set(est))[0]
            else:
                vals = set(est)
                c1 = est.count(vals[0])
                c2 = est.count(vals[1])
                raise ValueError("Estimator Types do not match. Construct object using same estimator types. {} count: {}, {} count: {}".format(vals[0],c1,vals[1],c2))
        else:
            self.models = {}

    def add_model(self,model):
        if str(model).split('(')[0] == "GridSearchCV":
            name = str(model.estimator).split('(')[0]
        else:
            name = str(model).split('(')[0]

        if name not in self.models:
            self.models[name] = model
        else:
            raise KeyError("{} already exists.".format(name))

    def remove_model(self,name):
        try:
            self.models.pop(name)
        except KeyError:
            raise KeyError("Key value for model name: {}, does not exist.".format(name))

    def fit(self,X,Y):
        logging.info("Fitting data on ")
        logging.info("...............................................")
        self._models = deepcopy(self.models)
        for model in self._models.values():
            model.fit(X,Y)
        return self

    def predict(self, X):
        predictions = {}
        try:
            for name, model in self._models.items():
                predictions[name] = model.predict(X)
            return predictions
        except AttributeError as e:
            raise ValueError("Error encountered while running predict. Check to see if models have been fitted. Error message: {}".format(e))

    def score_models(self, y_true, y_pred):

        scores = {}

        if self.estimator_type == 'classifier':
            for scorer in self.classifier_scorers:
                scores[str(scorer.__name__)] = scorer(y_true,y_pred)
            return scores
        elif self.estimator_type == 'regressor':
            for scorer in self.regressor_scorers:
                scores[str(scorer.__name__)] = scorer(y_true,y_pred)
                if str(scorer.__name__ ) == 'mean_squared_error':
                    scores['RMSE'] = np.sqrt(scorer(y_true,y_pred))
            return scores
        else:
            raise ValueError("Unknown estimator type ({})".format(self.estimator_type))

class MetaLearner(SupervisedLearner):

    def __init__(self,meta_model, model_list=[],logs=True, meta_method='Mean'):
        super().__init__(model_list,logs)
        self.meta_method = meta_method
        self.meta_model = meta_model

    def meta_fit(self, X,Y):
        logging.info("Generating Model Predictions for Meta Learner")
        logging.info("...............................................")
        predictions = super().predict(X)
        n = len(list(predictions.values())[0])
        m = len(predictions)
        self._X_meta = np.zeros((n,m))
        combos = list()
        for i,preds in enumerate(predictions.values()):
            self._X_meta[:,i] = preds
            temp = list(combinations(range(m),i+1))
            combos.extend(temp)

        for combo in combos:
            self._X_meta = np.column_stack((self._X_meta, self._X_meta[:,combo].mean(axis=1)))

        self._meta_model = deepcopy(self.meta_model)
        self._meta_model.fit(self._X_meta,Y)

    def meta_predict(self, X):
        logging.info("Predicting outcomes for Meta Learner")
        logging.info("...............................................")
        predictions = super().predict(X)
        n = len(list(predictions.values())[0])
        m = len(predictions)
        _X = np.zeros((n, m))
        combos = list()
        for i, preds in enumerate(predictions.values()):
            _X[:, i] = preds
            temp = list(combinations(range(m), i + 1))
            combos.extend(temp)
        for combo in combos:
            _X= np.column_stack((_X, _X[:,combo].mean(axis=1)))

        return self._meta_model.predict(_X)
    



