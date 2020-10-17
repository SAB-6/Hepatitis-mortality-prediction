import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib


class model:
    def__init__(self)
    self.df = None

    def read_data(self, file_path):
        print("\nReading dataframe\n")
        self.df = pd.read_table(file_path, sep=",")

    def preproces(self, float_cols):
        """
        Preprocess dataframe
        Args:
            float_cols:  type- list- colums to be changed to float type
        """
        print("preprocessing data")
        # replace missing values (?) with zeros
        self.df = self.df.replace("?", 0)
        self.df.columns = self.df.columns.str.lower().str.replace('" ", "_")
        int_columns = [col for col in list(
            self.df.columns) if self.df[col].dtypes != int if col not in float_cols]
        self.df[int_columns] = self.df[int_columns].astype(int)
        print(self.df.head)
        print("\n\n")
        # change both bilirubin and albumin columns to float
        self.df[float_cols] = self.df[float_cols].astype(float)
        self.df[target] = self.df[target].map({1: 1, 2: 0})
        print(self.df.dtypes)

    @staticmethod
    def train_test_split(features, target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3,
                                                            stratify=target, random_state=42)
        return X_train, X_test, y_train, y_test


def get_features(self, target, df):
    """
    Get feature columns
    Args:
        df - preprocessed dataframe
        target - target column, type- string
    Return:
        feature columns
    """
    features = [col for col in df.columns if col not in target]
    return features

    @staticmethod
    def pipe_model(X_train, X_test, y_train, y_test):
        # instantiate scaler
        scaler = StandardScaler()
        # each of thesesticRegression(solver='liblinear', random_state= 42)
        clf1 = LogisticRegression(solver='liblinear', random_state=42)
        clf2 = RandomForestClassifier(random_state=42)
        clf3 = GradientBoostingClassifier(validation_fraction=0.2,
                                          n_iter_no_change=5, tol=0.01,
                                          random_state=42)
        pipe = Pipeline([('scaler', scaler), ('classifier', clf1)])
        # create the parameter dictionary for clf1
        params1 = {}
        params1["classifier__penalty"] = ["l1", "l2", "none"]
        params1["classifier__C"] = [0.1, 1, 10]
        params1["classifier"] = [clf1]
        # create the parameter dictionary for clf2
        params2 = {}
        params2["classifier__n_estimators"] = [100, 200, 300]
        params2["classifier__min_samples_leaf"] = [1, 2, 5, 10]
        params2["classifier"] = [clf2]
        # create the parameter dictionary for clf2
        params3 = params2.copy()
        param3["classifier"] = [clf3]
        params = [params1, params2, params3]
        # Use GridSearchCV to get the best model
        grid = GridSearchCV(pipe, params, cv=5)
        grid.fit(X_train, y_train)
        best_param = grid.best_params_
        print("\nbest_param")
        best_estimator = grid.best_estimator_
        return best_estimator

        @staticmethod
        def save_model(X_train, X_test, y_train, y_test, model_path):
            best_estimator.fit(X_train, y_train)
            # Save model to directory
            print("Saving model to {}".format(model_path))
            with open(model_path, 'wb') as file:
                joblib.dump(model, file)
            print("\nmodel saved\n")


if __name__ == "__main__":
    target = "class"
    file_path = "../hepatitis.py"
    float_cols = ["bilirubin", "albumin"]
    model_path = "./"
    df =
