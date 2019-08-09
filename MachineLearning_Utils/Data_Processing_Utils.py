from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
import category_encoders as ce
import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
from termcolor import colored
import threading
import itertools
import time


def log_waiting(func_name):
    done = False
    for c in itertools.cycle(['.','..','...']):
        if done:
            break
        sys.stdout.write('\n{} processing '.format(func_name) + c)
        sys.stdout.flush()
        time.sleep(2)

    sys.stdout.write('\n{} processing complete.'.format(func_name))

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

class DataProcessor:

    def __init__(self, X={'X_train':None,'X_val':None,'X_test':None},Y={'Y_train':None,'Y_val':None,'Y_test':None},logs=True, random_seed=0):

        logging.info("Initializing DataProcessor object")
        logging.info("...............................................")

        if type(X) != dict or type(Y) != dict:
            raise TypeError("Expected dict type for X/Y. Received {}/{}".format(type(X), type(Y)))

        try:
            self._X_train = X['X_train']
        except KeyError as ke:
            print(ke)
            raise KeyError
        try:
            self._X_val = X['X_val']
        except KeyError as ke:
            print(ke)
            raise KeyError
        try:
            self._X_test = X['X_test']
        except KeyError as ke:
            print(ke)
            raise KeyError
        try:
            self._Y_train = Y['Y_train']
        except KeyError as ke:
            print(ke)
            raise KeyError
        try:
            self._Y_val = Y['Y_val']
        except KeyError as ke:
            print(ke)
            raise KeyError
        try:
            self._Y_val = Y['Y_val']
        except KeyError as ke:
            print(ke)
            raise KeyError

        self._random_seed = random_seed

        if logs:
            _setup_logging()

    @classmethod
    def load_train_test(cls,train_filepath="",test_filepath="", target_variable="",split=(0.8,0.2), index_col = None, random_seed=0):
        np.random.seed(random_seed)

        _, file_ext = os.path.splitext(train_filepath)
        if file_ext == '.csv':
            logging.info("...............................................")
            logging.info("Loading CSV file")
            try:
                X = pd.read_csv(train_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])

            try:
                X_test = pd.read_csv(test_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        elif file_ext == '.json':
            logging.info("...............................................")
            logging.info("Loading json file")
            try:
                X = pd.read_json(train_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])

            try:
                X_test = pd.read_json(test_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])

        elif file_ext == '.parquet':
            logging.info("...............................................")
            logging.info("Loading parquet file")
            try:
                X = pd.read_parquet(train_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])

            try:
                X_test = pd.read_parquet(test_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])

        else:
            raise ValueError("Expected the following files: [csv, json, parquet]")

        Y = X[target_variable]
        try:
            Y_test = X_test[target_variable]
        except:
            Y_test = None

        X.drop(target_variable, inplace=True, axis=1)

        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=split[0], shuffle=True, random_state=random_seed)

        return cls(X={'X_train': X_train,'X_val': X_val, 'X_test': X_test},
                   Y={'Y_train': Y_train,'Y_val': Y_val, 'Y_test':Y_test})

    @classmethod
    def load_dataset(cls, fulldata_filepath="", split =(0.6, 0.2, 0.2), target_variable = "", index_col = None, random_seed=0):
        np.random.seed(random_seed)

        _, file_ext = os.path.splitext(fulldata_filepath)
        if file_ext == '.csv':
            logging.info("...............................................")
            logging.info("Loading CSV file")
            try:
                X = pd.read_csv(fulldata_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        elif file_ext == '.json':
            logging.info("...............................................")
            logging.info("Loading json file")
            try:
                X = pd.read_json(fulldata_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        elif file_ext == '.parquet':
            logging.info("...............................................")
            logging.info("Loading parquet file")
            try:
                X = pd.read_parquet(fulldata_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        else:
            raise ValueError("Expected the following files: [csv, json, parquet]")

        train_test_total = split[1] + split[2]

        Y = X[target_variable]
        X.drop(target_variable, inplace=True, axis=1)

        X_train, x_temp, Y_train, y_temp = train_test_split(X, Y, train_size=split[0], shuffle=True, random_state=random_seed)
        X_val, X_test, Y_val, Y_test = train_test_split(x_temp, y_temp, train_size=split[1]/train_test_total, shuffle=True, random_state=random_seed)

        return cls(X={'X_train': X_train, 'X_val': X_val, 'X_test': X_test},
                   Y={'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test})

    @classmethod
    def load_train_val_test(cls, train_filepath="",test_filepath="", val_filepath="", target_variable="",index_col=None,random_seed=0):
        np.random.seed(random_seed)

        def load_data(data, _type):
            if _type == '.csv':
                try:
                    X = pd.read_csv(data,index_col=index_col)
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                except Exception:  # handle other exceptions such as attribute errors
                    print("Unexpected error:", sys.exc_info()[0])
                try:
                    Y = X[target_variable]
                except:
                    Y = None
                X.drop(target_variable, axis = 1, inplace = True)
                return X, Y
            elif _type == '.json':
                try:
                    X = pd.read_json(data, index_col=index_col)
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                except Exception:  # handle other exceptions such as attribute errors
                    print("Unexpected error:", sys.exc_info()[0])
                try:
                    Y = X[target_variable]
                except:
                    Y = None
                X.drop(target_variable, axis=1, inplace=True)
                return X, Y
            elif _type == '.parquet':
                try:
                    X = pd.read_parquet(data, index_col=index_col)
                except IOError as e:
                    print("I/O error({0}): {1}".format(e.errno, e.strerror))
                except Exception:  # handle other exceptions such as attribute errors
                    print("Unexpected error:", sys.exc_info()[0])
                try:
                    Y = X[target_variable]
                except:
                    Y = None
                X.drop(target_variable, axis=1, inplace=True)
                return X, Y
            else:
                raise ValueError("Expected the following files: [csv, json, parquet]")

        _, file_ext = os.path.splitext(train_filepath)

        X_train, Y_train = load_data(train_filepath, file_ext)
        X_val, Y_val = load_data(val_filepath, file_ext)
        X_test, Y_test = load_data(test_filepath, file_ext)

        return cls(X={'X_train': X_train, 'X_val': X_val, 'X_test': X_test},
                   Y={'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test})

    def get_datasets(self):
        log1 = False
        log2 = False
        log3 = False

        try:
            log1 = True
            _dict = {'X_data':{'X_train': self._X_train_transformed,'X_val': self._X_val_transformed, 'X_test': self._X_test_transformed}, 'Y_data':{'Y_train': self._Y_train_transformed,'Y_val': self._Y_val_transformed}}
        except AttributeError:
            try:
                log2 = True
                _dict = {'X_data':{'X_train': self._X_train_transformed,'X_val': self._X_val_transformed, 'X_test': self._X_test_transformed}, 'Y_data':{'Y_train': self._Y_train,'Y_val': self._Y_val}}
            except AttributeError:
                log3 = True
                _dict = {'X_data':{'X_train': self._X_train,'X_val': self._X_val, 'X_test': self._X_test}, 'Y_data':{'Y_train': self._Y_train,'Y_val': self._Y_val}}

        if log1 and not log2:
            logging.info("...............................................")
            logging.info("Returning datasets fully transformed.")
            return _dict
        elif log2 and not log3:
            logging.info("...............................................")
            logging.info("Returning datasets with just X transformed.")
            return _dict
        elif log3:
            logging.info("...............................................")
            logging.info("Returning datasets none of which are transformed.")
            return _dict

    def remove_na_target(self):
        """
        Removes row observations that do not have a target variable or 'y' outcome within the datasets.
        :return: None
        """

        def apply_deletion(X,y):
            null_indices = y.index[y.isnull()]
            _y = y.drop(null_indices)
            _X = X.drop(null_indices)
            return _X, _y

        try:
            logging.info("...............................................")
            logging.info("Running "+str(DataProcessor.remove_na_target))
            self._X_train_transformed, self._Y_train_transformed = apply_deletion(self._X_train_transformed, self._Y_train)
            self._X_val_transformed, self._Y_val_transformed = apply_deletion(self._X_val_transformed, self._Y_val)
            self._X_test_transformed, self._Y_test_transformed = apply_deletion(self._X_test_transformed, self._Y_test)
        except AttributeError:
            logging.info("...............................................")
            logging.debug("Attribute error occurred when applying "+str(DataProcessor.remove_na_target))
            self._X_train_transformed, self._Y_train_transformed = apply_deletion(self._X_train,self._Y_train)
            self._X_val_transformed, self._Y_val_transformed = apply_deletion(self._X_val, self._Y_val)
            self._X_test_transformed, self._Y_test_transformed = apply_deletion(self._X_test, self._Y_test)

    def impute(self, method="mean", fill_value=None,cols=None):
        """
        :param method: acceptable impute methods: ['mean', 'median', 'most_frequent', 'constant', 'argmax', 'drop']
        :param fill_value: Constant fill value to fill in np.nan values with if method is 'constant'
        :param cols: list of columns to apply imputation. If left None, then imputation will be applied on dataframe.
        :return: dictionary of transformed feature dataframes
        """

        def apply_imputation(transformed_X, original_X):
            transformed_X.columns = original_X.columns
            transformed_X.index = original_X.index
            return transformed_X

        if not hasattr(self,'_X_train_transformed'):
            self._X_train_transformed = self._X_train
            self._X_val_transformed = self._X_val
            self._X_test_transformed = self._X_test

        if method in ['mean','median','most_frequent','constant']:
            if method == 'constant':
                imputer = SimpleImputer(missing_values=np.nan, strategy=method, fill_value=fill_value)
            else:
                imputer = SimpleImputer(missing_values=np.nan, strategy=method)
            numeric_columns = [col for col in self._X_train if (self._X_train[col].dtype == 'int64') or (self._X_train[col].dtype == 'float64')]
            try:
                if cols is None:

                    train_handler = apply_imputation(pd.DataFrame(imputer.fit_transform(self._X_train_transformed[numeric_columns])), self._X_train_transformed[numeric_columns])
                    val_handler = apply_imputation(pd.DataFrame(imputer.transform(self._X_val_transformed[numeric_columns])),self._X_val_transformed[numeric_columns])
                    test_handler = apply_imputation(pd.DataFrame(imputer.transform(self._X_test_transformed[numeric_columns])), self._X_test_transformed[numeric_columns])

                    self._X_train_transformed.drop(numeric_columns, axis = 1, inplace=True)
                    self._X_val_transformed.drop(numeric_columns, axis = 1, inplace=True)
                    self._X_test_transformed.drop(numeric_columns, axis = 1, inplace=True)

                    self._X_train_transformed = pd.concat([self._X_train_transformed,train_handler],axis=1)
                    self._X_val_transformed = pd.concat([self._X_val_transformed, val_handler], axis=1)
                    self._X_test_transformed = pd.concat([self._X_test_transformed, test_handler], axis=1)

                else:

                    self._X_train_transformed[cols] = apply_imputation(pd.DataFrame(imputer.fit_transform(self._X_train_transformed[cols])),self._X_train_transformed[cols])
                    self._X_val_transformed[cols] = apply_imputation(pd.DataFrame(imputer.transform(self._X_val_transformed[cols])), self._X_val_transformed[cols])
                    self._X_test_transformed[cols] = apply_imputation(pd.DataFrame(imputer.transform(self._X_test_transformed[cols])), self._X_test_transformed[cols])

            except AttributeError as e:
                print(e)
                if cols is None:

                    train_handler = apply_imputation(pd.DataFrame(imputer.fit_transform(self._X_train[numeric_columns])),self._X_train_transformed[numeric_columns])
                    val_handler = apply_imputation(pd.DataFrame(imputer.transform(self._X_val[numeric_columns])),self._X_val[numeric_columns])
                    test_handler = apply_imputation(pd.DataFrame(imputer.transform(self._X_test[numeric_columns])),self._X_test[numeric_columns])

                    self._X_train_transformed.drop(numeric_columns, axis=1, inplace=True).reset_index(inplace=True)
                    self._X_val_transformed.drop(numeric_columns, axis=1, inplace=True).reset_index(inplace=True)
                    self._X_test_transformed.drop(numeric_columns, axis=1, inplace=True).reset_index(inplace=True)

                    self._X_train_transformed = pd.concat([self._X_train, train_handler], axis=1)
                    self._X_val_transformed = pd.concat([self._X_val, val_handler], axis=1)
                    self._X_test_transformed = pd.concat([self._X_test, test_handler], axis=1)

                else:
                    self._X_train_transformed[cols] = apply_imputation(pd.DataFrame(imputer.fit_transform(self._X_train[cols])),self._X_train[cols])
                    self._X_val_transformed[cols] = apply_imputation(pd.DataFrame(imputer.transform(self._X_val[cols])), self._X_val[cols])
                    self._X_test_transformed[cols] = apply_imputation(pd.DataFrame(imputer.transform(self._X_test[cols])), self._X_test[cols])

            return {'X_train_transformed': self._X_train_transformed,'X_val_transformed': self._X_val_transformed, 'X_test_transformed': self._X_test_transformed}

        elif method == 'drop':
            null_cols = self._X_train.isnull().sum()
            col_names = list(null_cols[null_cols > 0].index)
            self._X_train.drop(col_names, axis=1, inplace=True)
            self._X_val.drop(col_names, axis=1, inplace=True)
            self._X_test.drop(col_names, axis=1, inplace=True)

        elif method == "argmax":
            """
            WIP: This function is to be built out to use conditional probability/bayes (potentially) to predict what would be the most likely value to input based on the other features.
            """
            pass
        else:
            raise ValueError("Invalid method imputation method called. Review DataProcessor instance method 'impute' for list of methods to call.")

    def label_encoding(self,encoding_type='one_hot',cardinality=None, column_names=None):
        """
        :param encoding_type: 'label', 'one_hot', 'backward_difference', 'binary', 'sum', 'helmert'
        :param cardinality: Threshold by which you want to encode label columns based on number of unique labels within column. E.g. if cardinality = 10, any column with less than 10 unique labels will have one hot encoding applied.
        :return: None

        It is recommended to apply one hot encoding prior to label encoding based on categories/cardinality you wish to have label encoded vs one hot encoded

        Otherwise if encodiing type is label_encoding then generic label encoding will be applied across
        """

        if column_names == None:
            pass
        elif type(column_names) == list or type(column_names) == tuple:
            pass
        else:
            raise ValueError("Expected either list or tuple object type for column_names parameter.")

        if encoding_type == 'one_hot':
            try:
                object_cols = [col for col in self._X_train_transformed if self._X_train_transformed[col].dtype == 'object']
            except AttributeError:
                object_cols = [col for col in self._X_train if self._X_train[col].dtype == 'object']

            if cardinality is None:
                raise ValueError("Expected numeric value for Cardinality for label encoding method 'one_hot'")
            cardinality_cols = [col for col in object_cols if self._X_train[col].nunique() < cardinality]
            encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

            for col in cardinality_cols:
                try:
                    self._X_train_transformed[col][self._X_train_transformed[col].isna()] = 'NaN'
                    self._X_val_transformed[col][self._X_val_transformed[col].isna()] = 'NaN'
                    self._X_test_transformed[col][self._X_test_transformed[col].isna()] = 'NaN'
                except AttributeError:
                    self._X_train_transformed[col][self._X_train[col].isna()] = 'NaN'
                    self._X_val_transformed[col][self._X_val[col].isna()] = 'NaN'
                    self._X_test_transformed[col][self._X_test[col].isna()] = 'NaN'

            try:
                train_handler = pd.DataFrame(encoder.fit_transform(self._X_train_transformed[cardinality_cols]))
                val_handler = pd.DataFrame(encoder.transform(self._X_val_transformed[cardinality_cols]))
                test_handler = pd.DataFrame(encoder.transform(self._X_test_transformed[cardinality_cols]))

                train_handler.index = self._X_train_transformed.index
                val_handler.index = self._X_val_transformed.index
                test_handler.index = self._X_test_transformed.index

                self._X_train_transformed.drop(cardinality_cols, inplace=True, axis=1)
                self._X_val_transformed.drop(cardinality_cols, inplace=True, axis=1)
                self._X_test_transformed.drop(cardinality_cols, inplace=True, axis=1)

                self._X_train_transformed = pd.concat([self._X_train_transformed,train_handler], axis=1)
                self._X_val_transformed = pd.concat([self._X_val_transformed, val_handler], axis=1)
                self._X_test_transformed = pd.concat([self._X_test_transformed, test_handler], axis=1)

            except AttributeError:
                train_handler = pd.DataFrame(encoder.fit_transform(self._X_train[cardinality_cols]))
                val_handler = pd.DataFrame(encoder.fit_transform(self._X_val[cardinality_cols]))
                test_handler = pd.DataFrame(encoder.fit_transform(self._X_test[cardinality_cols]))

                train_handler.index = self._X_train.index
                val_handler.index = self._X_val.index
                test_handler.index = self._X_test.index

                x_train_handle = self._X_train_transformed.drop(cardinality_cols, axis=1)
                x_val_handle = self._X_val_transformed.drop(cardinality_cols, axis=1)
                x_test_handle = self._X_test_transformed.drop(cardinality_cols, axis=1)

                self._X_train_transformed = pd.concat([x_train_handle, train_handler], axis=1)
                self._X_val_transformed = pd.concat([x_val_handle, val_handler], axis=1)
                self._X_test_transformed = pd.concat([x_test_handle, test_handler], axis=1)

        elif encoding_type == 'label':
            encoder = LabelEncoder()

            try:
                object_cols = [col for col in self._X_train_transformed if self._X_train_transformed[col].dtype == 'object']
            except AttributeError:
                object_cols = [col for col in self._X_train if self._X_train[col].dtype == 'object']

            try:
                for col in object_cols:
                    self._X_train_transformed[col][self._X_train_transformed[col].isna()] = 'NaN'
                    self._X_val_transformed[col][self._X_val_transformed[col].isna()] = 'NaN'
                    self._X_test_transformed[col][self._X_test_transformed[col].isna()] = 'NaN'

                    handle = self._X_train_transformed[col].append([self._X_val_transformed[col], self._X_test_transformed[col]])
                    encoder.fit(handle)
                    self._X_train_transformed[col] = encoder.transform(self._X_train_transformed[col])
                    self._X_val_transformed[col] = encoder.transform(self._X_val_transformed[col])
                    self._X_test_transformed[col] = encoder.transform(self._X_test_transformed[col])
            except AttributeError:
                self._X_train_transformed = self._X_train
                self._X_val_transformed = self._X_val
                self._X_test_transformed = self._X_test
                for col in object_cols:
                    self._X_train_transformed[col][self._X_train_transformed[col].isna()] = 'NaN'
                    self._X_val_transformed[col][self._X_val_transformed[col].isna()] = 'NaN'
                    self._X_test_transformed[col][self._X_test_transformed[col].isna()] = 'NaN'

                    handle = self._X_train[col].append([self._X_val[col], self._X_test[col]])
                    encoder.fit(handle)
                    self._X_train_transformed[col] = encoder.transform(self._X_train_transformed[col])
                    self._X_val_transformed[col] = encoder.transform(self._X_val_transformed[col])
                    self._X_test_transformed[col] = encoder.transform(self._X_test_transformed[col])
        elif encoding_type == 'binary':
            try:
                object_cols = [col for col in self._X_train_transformed if self._X_train_transformed[col].dtype == 'object']
            except AttributeError:
                object_cols = [col for col in self._X_train if self._X_train[col].dtype == 'object']

            binary_cols = [col for col in object_cols if self._X_train[col].nunique() == 2]
            binarizer = LabelBinarizer()

            try:
                for col in binary_cols:
                    self._X_train_transformed[col] = binarizer.fit_transform(self._X_train_transformed[col])
                    self._X_val_transformed[col] = binarizer.transform(self._X_val_transformed[col])
                    self._X_test_transformed[col] = binarizer.transform(self._X_test_transformed[col])
            except AttributeError:
                self._X_train_transformed = self._X_train
                self._X_val_transformed = self._X_val
                self._X_test_transformed = self._X_test
                for col in binary_cols:
                    self._X_train_transformed[col] = binarizer.fit_transform(self._X_train_transformed[col])
                    self._X_val_transformed[col] = binarizer.transform(self._X_val_transformed[col])
                    self._X_test_transformed[col] = binarizer.transform(self._X_test_transformed[col])
        elif encoding_type == 'sum':
            try:
                object_cols = [col for col in self._X_train_transformed if self._X_train_transformed[col].dtype == 'object']
            except AttributeError:
                object_cols = [col for col in self._X_train if self._X_train[col].dtype == 'object']

            if column_names == None:
                column_names = object_cols
                UserWarning("column_names was left 'None'. Using Object data type columns ")
                #raise ValueError("Column names must be supplied when using ")

            encoder = ce.sum_coding.SumEncoder(cols=list(column_names))
            try:
                self._X_train_transformed = encoder.fit_transform(self._X_train_transformed)
                self._X_val_transformed = encoder.transform(self._X_val_transformed)
                self._X_test_transformed = encoder.transform(self._X_test_transformed)
            except AttributeError:
                self._X_train_transformed = self._X_train
                self._X_val_transformed = self._X_val
                self._X_test_transformed = self._X_test

                self._X_train_transformed = encoder.fit_transform(self._X_train_transformed)
                self._X_val_transformed = encoder.transform(self._X_val_transformed)
                self._X_test_transformed = encoder.transform(self._X_test_transformed)

        elif encoding_type == 'helmert':
            try:
                object_cols = [col for col in self._X_train_transformed if self._X_train_transformed[col].dtype == 'object']
            except AttributeError:
                object_cols = [col for col in self._X_train if self._X_train[col].dtype == 'object']

            if column_names == None:
                column_names = object_cols
                UserWarning("column_names was left 'None'. Using Object data type columns ")
                #raise ValueError("Column names must be supplied when using ")

            encoder = ce.helmert.HelmertEncoder(cols=list(column_names))
            try:
                self._X_train_transformed = encoder.fit_transform(self._X_train_transformed)
                self._X_val_transformed = encoder.transform(self._X_val_transformed)
                self._X_test_transformed = encoder.transform(self._X_test_transformed)
            except AttributeError:
                self._X_train_transformed = self._X_train
                self._X_val_transformed = self._X_val
                self._X_test_transformed = self._X_test

                self._X_train_transformed = encoder.fit_transform(self._X_train_transformed)
                self._X_val_transformed = encoder.transform(self._X_val_transformed)
                self._X_test_transformed = encoder.transform(self._X_test_transformed)
        elif encoding_type =='backward_difference':
            try:
                object_cols = [col for col in self._X_train_transformed if self._X_train_transformed[col].dtype == 'object']
            except AttributeError:
                object_cols = [col for col in self._X_train if self._X_train[col].dtype == 'object']

            if column_names == None:
                column_names = object_cols
                UserWarning("column_names was left 'None'. Using Object data type columns ")
                #raise ValueError("Column names must be supplied when using ")

            encoder = ce.backward_difference.BackwardDifferenceEncoder(cols=list(column_names))
            try:
                self._X_train_transformed = encoder.fit_transform(self._X_train_transformed)
                self._X_val_transformed = encoder.transform(self._X_val_transformed)
                self._X_test_transformed = encoder.transform(self._X_test_transformed)
            except AttributeError:
                self._X_train_transformed = self._X_train
                self._X_val_transformed = self._X_val
                self._X_test_transformed = self._X_test

                self._X_train_transformed = encoder.fit_transform(self._X_train_transformed)
                self._X_val_transformed = encoder.transform(self._X_val_transformed)
                self._X_test_transformed = encoder.transform(self._X_test_transformed)

    def pipeline(self, steps):
        """
        Executes a pipeline of steps to pre process a dataset through.
        :param steps: Dictionary of methods to call with their value consisting of a list/tuple of parameters to call the method from.
        :return: Dictionary of processed data.

        E.g.

        steps = {'impute':["mean",None,['Col_A','Col_B']], 'label_encoding':['helmert',None,['Col_A','Col_D']] }
         or
        steps = {'label_encoding': ['one_hot',5], 'label_encoding':['sum', None, ['Col_A', 'Col_B']], 'impute': ['constant',3,['Col_C']], 'impute': ['median']}
         or
        steps = {'remove_na_target' : (), 'label_encoding' : ('label')}
        """

        if type(steps) != dict:
            raise ValueError
        elif len(steps) == 0:
            raise ValueError("No steps were supplied. Pipeline constructor failed.")

        methods = list(steps.keys())
        self._X_train_transformed = self._X_train
        self._X_val_transformed = self._X_val
        self._X_test_transformed = self._X_test

        available_methods = [func for func in list(DataProcessor.__dict__.keys()) if "__" not in func]

        for method in methods:
            if method not in available_methods:
                raise ValueError("Not applicable method to use in pipeline. Method: {}".format(method))
            params = steps[method]
            _ = getattr(DataProcessor,method)(tuple(params))

        return self.get_datasets()

    def vif(self):
        """
        :param threshold: VIF threshold for dropping or removing Independent Variables
        :param drop: Inidicator (default False) to return and drop Independent Variables from the Matrix or DataFrame
        :return: If Drop set to (True) will return the VIF values, along with the DataFrame or Matrix with the dropped Independent Variables. Otherwise will return the VIF values.
        """
        try:
            total_na = self._X_train_transformed.isnull().sum().sum()
        except AttributeError:
            total_na = self._X_train.isnull().sum().sum()
        except Exception:
            logging.debug("Unexpected error: {}".format(sys.exc_info()[0]))

        if total_na > 0 :
            raise ValueError("To use Variance Inflation Factor function, you must not contain Null data within the training dataset.")
        try:
            X = add_constant(self._X_train_transformed)
        except AttributeError:
            X = add_constant(self._X_train)

        #t = threading.Thread(target=log_waiting('Variance Inflation Factor'))
        #t.start()
        self.vif = pd.Series([variance_inflation_factor(X.values,i) for i in range(X.shape[1])], X.columns).sort_values(ascending=False)
        #done = True
        return self.vif

    def __len__(self):
        return self._X_train.shape[0] + self._X_test.shape[0] + self._X_val.shape[0]

    def __str__(self):
        try:
            return """
            Datasets schemas:\n
            \tTraining data - Row Count: {}; Column Count: {}\n
            \tValidation data - Row Count: {}; Column Count: {}\n
            \tTesting data - Row Count: {}; Column Count: {}\n
            
            Total Training Data by col N/A: \n{}\n
            Total Validation Data by col N/A: \n{}\n
            Total Testing Data by col N/A: \n{}\n
                    
            Total Training Data N/A: {}\n
            Total Validation Data N/A: {}\n
            Total Testing Data N/A: {}\n
            """.format(self._X_train_transformed.shape[0],self._X_train_transformed.shape[1],self._X_val_transformed.shape[0],
                       self._X_val_transformed.shape[1],self._X_test_transformed.shape[0],self._X_test_transformed.shape[1],
                       self._X_train_transformed.isna().sum()[self._X_train_transformed.isna().sum() > 0], self._X_val_transformed.isna().sum()[self._X_val_transformed.isna().sum() > 0],
                       self._X_test_transformed.isna().sum()[self._X_test_transformed.isna().sum() > 0], self._X_train_transformed.isnull().sum().sum(),
                       self._X_val_transformed.isnull().sum().sum(), self._X_test_transformed.isnull().sum().sum())

        except AttributeError:
            return """
                    Datasets schemas:\n
                    \tTraining data - Row Count: {}; Column Count: {}\n
                    \tValidation data - Row Count: {}; Column Count: {}\n
                    \tTesting data - Row Count: {}; Column Count: {}\n

                    Total Training Data by col N/A: \n{}\n
                    Total Validation Data by col N/A: \n{}\n
                    Total Testing Data by col N/A: \n{}\n
                    
                    Total Training Data N/A: \n{}\n
                    Total Validation Data N/A: \n{}\n
                    Total Testing Data N/A: \n{}\n
                    """.format(self._X_train.shape[0], self._X_train.shape[1], self._X_val.shape[0],
                               self._X_val.shape[1], self._X_test.shape[0], self._X_test.shape[1],
                               self._X_train.isna().sum()[self._X_train.isna().sum() > 0], self._X_val.isna().sum()[self._X_val.isna().sum() > 0], self._X_test.isna().sum()[self._X_test.isna().sum() > 0],
                               self._X_train.isnull().sum().sum(), self._X_val.isnull().sum().sum(), self._X_test.isnull().sum().sum())


class EDA:

    def __init__(self, data, logs=True, labels=None):

        if isinstance(data, pd.DataFrame):
            if logs:
                logging.info("Initializing EDA object for Pandas DataFrame")
                logging.info("...............................................")
            self.data_type = 'dataframe'
            self.data = data
        elif isinstance(data, np.ndarray):
            if logs:
                logging.info("Initializing EDA object Numpy ndarray")
                logging.info("...............................................")
            self.data_type = 'ndarray'
            self.data = data
            if (labels is None) or (type(labels) != list) or (type(labels) != tuple):
                raise ValueError("Labels must be passed as a list or tuple for an ndarray dataset.")
            self.labels = labels
        else:
            raise ValueError('Unexpected dataset type. Expected pandas.DataFrame or numpy.ndarray')

    @classmethod
    def load_data_from_file(cls, fulldata_filepath="", index_col = None):

        _, file_ext = os.path.splitext(fulldata_filepath)
        if file_ext == '.csv':
            logging.info("...............................................")
            logging.info("Loading CSV file")
            try:
                X = pd.read_csv(fulldata_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        elif file_ext == '.json':
            logging.info("...............................................")
            logging.info("Loading json file")
            try:
                X = pd.read_json(fulldata_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        elif file_ext == '.parquet':
            logging.info("...............................................")
            logging.info("Loading parquet file")
            try:
                X = pd.read_parquet(fulldata_filepath, index_col=index_col)
            except IOError as e:
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except Exception:  # handle other exceptions such as attribute errors
                print("Unexpected error:", sys.exc_info()[0])
        else:
            raise ValueError("Expected the following files: [csv, json, parquet]")

        return cls(data=X)

    def corr_plot(self, **kwargs):
        """
        :param kwargs: kwargs that relate to the Seaborn interface for heatmaps. https://seaborn.pydata.org/generated/seaborn.heatmap.html
        :return: None
        """
        if self.data_type == 'dataframe':
            fig, ax = plt.subplots()
            ax = sns.heatmap(self.data.corr(), *kwargs)
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax = sns.heatmap(np.corrcoef(self.data.T), *kwargs)
            plt.show()

    def scatter_plot(self,x_data_name, y_data_name, x_label, y_label, title, **kwargs):
        _, ax = plt.subplots()
        ax.scatter(self.data[x_data_name], self.data[y_data_name], s = 30, color = '#539caf', alpha = 0.75, *kwargs)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.show()

    def describe_data(self):
        numeric_cols = [col for col in self.data if (self.data[col].dtype == 'int64') or (self.data[col].dtype == 'float64')]
        msg = ''
        msg += colored("DATASET DESCRIPTION (NUMERICAL VARIABLES)\n", 'blue', attrs=['bold'])
        for col in numeric_cols:
            temp = self.data[col].describe()
            line = colored("\t{} : Mean - {:.2f} ; Standard Deviation - {:.2f} ; Minimum Value - {:.2f} ; Maximum Value - {:.2f}\n".format(col, temp['mean'], temp['std'], temp['min'], temp['max']),'blue')
            msg += line

        return msg
    @staticmethod
    def corr_matrix(dataset):
        if hasattr(dataset,pd.DataFrame):
            return dataset.corr()
        elif hasattr(dataset, np.ndarray):
            return np.corrcoef(dataset)