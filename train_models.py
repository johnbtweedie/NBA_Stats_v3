import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import joblib
from sklearn.decomposition import PCA
import sqlite3
import pickle
import itertools
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
# Ignore all warnings
warnings.filterwarnings('ignore')

def model_classification_performance(y_test, y_pred, model_name, model_call):
    con_matrix = confusion_matrix(y_test, y_pred.round())
    TN, FP, FN, TP = con_matrix.ravel()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = f1_score(y_test, y_pred.round())
    roc = roc_auc_score(y_test, y_pred.round())

    results_dict = {
        "Model": model_name,
        "Call": str(model_call),
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "f1-score": f1,
        "roc auc score": roc}

    return pd.DataFrame(results_dict, index=[model_name])

class assembleClassificationModel:
    def __init__(self, 
                 conn=sqlite3.connect('nba_database_2025-01-18.db'),
                 model_type='classification',
                 target_response='WL',
                 scale=True,
                 pca=True):
        
        self.conn = conn
        
        self.features = self.load_data('feature_table')
        self.process_features()
        
        self.responses = self.load_data('response_table')
        self.process_response(target_response)
        
        self.test_train_split()
        if scale:
            self.scale_data()
        if pca:
            self.pca()

        if model_type == 'classification':
            self.train_classification_models()

    def load_data(self, db_table_name):
        '''
        get features and responses from database
        '''
        print(f'loading {db_table_name} from database...')
        df = pd.read_sql(f'SELECT * FROM {db_table_name}', self.conn)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        print('...complete\n')
        return df

    def process_features(self):
        '''
        prune and process feature set
        '''
        print('processing feature data...')
        feature_cols = pd.read_excel(r'/Users/johntweedie/Dev/Projects/PN24001_NBA_Stats/catalogs/features_cols.xlsx',
                             sheet_name='all')['feature_cols'].tolist()
        feature_cols = [col.strip().replace("'", "") for col in feature_cols]
        self.features = self.features.dropna()
        # self.features = self.features.drop(columns='DaysElapsed')
        self.features = self.features[feature_cols]
        print('...complete\n')

    def process_response(self, target_response):
        print('procesing response data...')
        self.responses = self.responses.loc[self.features.index]
        self.responses = self.responses[f'{target_response}_r']
        print('...complete\n')

    def test_train_split(self, test_size=0.2):
        print('segmenting test/train sets...')
        X_train, X_test, y_train, y_test = train_test_split(self.features,
                                                            self.responses, 
                                                            test_size=test_size, 
                                                            random_state=100)
        self.data_dict = {
            'X_train' : X_train,
            'X_test' : X_test,
            'y_train' : y_train,
            'y_test' : y_test
        }
        print('...complete\n')

    def scale_data(self):
        '''
        standardize features and save to data dictionary
        '''
        print('standardizing data...')
        self.scaler = StandardScaler()
        self.data_dict['X_train'] = pd.DataFrame(self.scaler.fit_transform(self.data_dict['X_train']), 
                                                index=self.data_dict['y_train'].index,
                                                columns=self.features.columns)
        self.data_dict['X_test'] = pd.DataFrame(self.scaler.transform(self.data_dict['X_test']), 
                                                index=self.data_dict['y_test'].index, 
                                                columns=self.features.columns)
        print('...complete\n')

    def pca(self):
        '''
        pca-transform features using kaiser criteria for n comoponents, and save to data dictionary
        '''
        print('performing PCA transformation...')
        cov_matrix = np.cov(self.data_dict['X_train'].T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        num_components_kaiser = sum(sorted_eigenvalues > 1 + 1)
        self.pca_model = PCA(n_components=num_components_kaiser)
        self.data_dict['X_train'] = pd.DataFrame(self.pca_model.fit_transform(self.data_dict['X_train']), 
                                                index=self.data_dict['y_train'].index)
        self.data_dict['X_test'] = pd.DataFrame(self.pca_model.transform(self.data_dict['X_test']), 
                                                index=self.data_dict['y_test'].index)
        print('...complete\n')

    def train_classification_models(self):
        self.models = {}
        self.tune_model_rf()
        self.tune_model_logreg()
        # self.tune_model_svm()
        # self.tune_model_nnet()
        self.save_best_models()

    def tune_model_nnet(self):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Neural Net Model
        # ----------------------------------------------------------------------------------------------------------------------#

        print('fitting nn model')
        nn_model = MLPClassifier(solver='sgd', random_state=100)

        param_grid = {
            'hidden_layer_sizes': [(6, 2, 2), (10, 5), (10, 10), (50, 30, 10)],  # Different layer sizes
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # Regularization strength
            'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
            'max_iter': [200, 300, 500],  # Maximum number of iterations
            'solver': ['sgd', 'adam'],  # Different solvers
        }
        # Set up the grid search
        grid_search = GridSearchCV(estimator=nn_model,
                                param_grid=param_grid,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=2)

        print('Fitting nn model with Grid Search')
        grid_search.fit(self.data_dict['X_train'], self.data_dict['y_train'])

        # Best parameters from grid search
        best_params = grid_search.best_params_
        print("Best Parameters from Grid Search:", best_params)

        # Use the best estimator to predict
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.data_dict['X_test'])

        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="nn model", model_call=best_model)

        self.models['class_nnet'] = (best_model, df_results)
        print('...complete\n')

    def tune_model_svm(self):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Support Vector Machine
        # ----------------------------------------------------------------------------------------------------------------------#

        print('Fitting SVM model')

        # Define a more refined param grid (restrict degrees for poly, only use gamma for rbf/poly)
        param_grid = [
            {'C': np.logspace(-4, 1, 6), 'kernel': ['linear']},
            {'C': np.logspace(-4, 1, 6), 'kernel': ['rbf'], 'gamma': np.logspace(-4, 1, 6)},
            {'C': np.logspace(-4, 1, 6), 'kernel': ['poly'], 'degree': [2, 3], 'gamma': np.logspace(-4, -1, 4)}
        ]

        grid_search = GridSearchCV(estimator=SVC(probability=True),
                                param_grid=param_grid,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=2)

        grid_search.fit(self.data_dict['X_train'], self.data_dict['y_train'])

        # Best parameters from grid search
        best_params = grid_search.best_params_
        print(f"Best Parameters from Grid Search: {best_params}")

        # Use the best estimator to predict
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.data_dict['X_test'])

        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="SVM model", model_call=best_model)

        self.models['class_svm'] = (best_model, df_results)
        print('...complete\n')

    def tune_model_logreg(self):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Logistic Regression Model
        # ----------------------------------------------------------------------------------------------------------------------#
        print('Fitting Logistic Regression model')

        # Define the parameter grid for Logistic Regression
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Different regularization penalties
            'C': np.logspace(-4, 4, 10),  # Inverse of regularization strength
            'solver': ['lbfgs', 'liblinear', 'saga'],  # Different solvers
            'max_iter': [100, 200, 500]  # Maximum number of iterations
        }

        # Set up GridSearchCV for Logistic Regression
        grid_search = GridSearchCV(estimator=LogisticRegression(random_state=100),
                                param_grid=param_grid,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=2)

        grid_search.fit(self.data_dict['X_train'], self.data_dict['y_train'])

        # Best parameters from grid search
        best_params = grid_search.best_params_
        print("Best Parameters from Grid Search:", best_params)

        # Use the best estimator to predict
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.data_dict['X_test'])

        # Evaluate and save results
        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="Logistic Regression", model_call=best_model)
        
        self.models['class_logistic'] = (best_model, df_results)
        print('...complete\n')

    def tune_model_rf(self, dense_grid=True):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Random Forest Model
        # ----------------------------------------------------------------------------------------------------------------------#
        print('Fitting Random Forest model')

        # Define the parameter grid for Random Forest
        if dense_grid:
            param_grid = {
                'n_estimators': [100, 200, 500],  # Number of trees in the forest
                'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
                'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
                'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
                'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
            }
        else:
            param_grid = {
                'n_estimators': [100, 200],  # Number of trees in the forest
                'max_depth': [10, 20],  # Maximum depth of each tree
                'min_samples_split': [2, 5],  # Minimum samples required to split an internal node
                'min_samples_leaf': [1, 2],  # Minimum samples required to be at a leaf node
                'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
            }
        # Set up GridSearchCV for Random Forest
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=100),
                                param_grid=param_grid,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=2)

        grid_search.fit(self.data_dict['X_train'], self.data_dict['y_train'])

        # Best parameters from grid search
        best_params = grid_search.best_params_
        print("Best Parameters from Grid Search:", best_params)

        # Use the best estimator to predict
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.data_dict['X_test'])

        # Evaluate and save results
        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="Random Forest", model_call=best_model)

        self.models['class_randomForest'] = (best_model, df_results)
        print('...complete\n')

    def save_best_models(self):
        print('saving best models...')
        datestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_filename = f'best_models_{datestamp}.pkl'
        joblib.dump(self.models, model_filename)
        print('...complete\n')

WL_models = assembleClassificationModel()
print("complete")

