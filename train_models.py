import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score, log_loss
from sklearn.linear_model import LogisticRegression
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

def holdout_classification_performance(y_test, y_prob, model_name, model_call):
    '''
    evaluate WL classification by grouping predictions for each game in the holdout set
    the higher probability gets assigned the win
    This is improving the holdout accuracy roughly 9% compared to the test accuracy for a single game
    '''
    df = pd.DataFrame(y_test)
    if len(y_prob.shape) == 1:
        df['prob_class_1'], df['prob_class_0'] = y_prob, (1-probs) # from the ensemble
    else:
        df['prob_class_0'], df['prob_class_1'] = y_prob[:, 0], y_prob[:, 1]
    # Group by the game id to compare predicted probabilities of winning to opponent
    # higher probability of win is assigned the win ('1')
    df['higher_prob'] = df.groupby(level=1)['prob_class_1'].transform(lambda x: (x == x.max()).astype(int))

    con_matrix = confusion_matrix(df['WL_r'], df['higher_prob'])

    TN, FP, FN, TP = con_matrix.ravel()

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = f1_score(df['WL_r'], df['higher_prob'])
    roc = roc_auc_score(df['WL_r'], df['higher_prob'])

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

class trainClassificationModel:
    def __init__(self, 
                 conn=sqlite3.connect('nba_database_2025-01-18.db'),
                 model_type='classification',
                 target_response='WL',
                 scale=True,
                 pca=True,
                 dense_grid=True):
        
        self.conn = conn
        
        self.features = self.load_data('feature_table')
        self.process_features()
        
        self.responses = self.load_data('response_table')
        self.process_response(target_response)
        
        self.test_train_split()
        self.scaler_used = False
        self.pca_used = False
        if scale:
            self.scale_data()
            self.scaler_used = True
        if pca:
            self.pca()
            self.pca_used = True

        if model_type == 'classification':
            self.train_classification_models(dense_grid)
        print('complete')

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

    def process_features(self, use_all=False):
        '''
        prune and process feature set
        '''
        print('processing feature data...')
        feature_cols = pd.read_excel(r'/Users/johntweedie/Dev/Projects/PN24001_NBA_Stats/catalogs/features_cols.xlsx',
                             sheet_name='feat_2024-05-19')['feature_cols'].tolist()
        feature_cols = [col.strip().replace("'", "") for col in feature_cols]
        self.features = self.features.dropna()
        # self.features = self.features.drop(columns='DaysElapsed')
        if not use_all:            
            self.features = self.features[feature_cols]
        print('...complete\n')

    def process_response(self, target_response):
        print('procesing response data...')
        self.responses = self.responses.loc[self.features.index]
        self.responses = self.responses[f'{target_response}_r']
        print('...complete\n')

    def test_train_split(self, test_size=0.2, holdout=True, n_games=1000):
        print('segmenting test/train sets...')
        # Determine holdout and training/test split
        self.holdout = False
        if holdout:
            self.holdout = True
            self.data_dict = {
                'X_hold': self.features.iloc[-n_games:],
                'y_hold': self.responses.iloc[-n_games:],
            }
            X_data, y_data = self.features.iloc[:-n_games], self.responses.iloc[:-n_games]
        else:
            X_data, y_data = self.features, self.responses

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=test_size, random_state=100
        )

        # Store results in data dictionary
        self.data_dict.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
        print('...complete\n')

    def scale_data(self):
        '''
        standardize features and save to data dictionary
        '''
        print('standardizing data...')
        self.data_dict['features'] = self.features.columns
        self.scaler = StandardScaler()
        self.data_dict['X_train'] = pd.DataFrame(self.scaler.fit_transform(self.data_dict['X_train']), 
                                                index=self.data_dict['y_train'].index,
                                                columns=self.features.columns)
        self.data_dict['X_test'] = pd.DataFrame(self.scaler.transform(self.data_dict['X_test']), 
                                                index=self.data_dict['y_test'].index, 
                                                columns=self.features.columns)
        if self.holdout:
            self.data_dict['X_hold'] = pd.DataFrame(self.scaler.transform(self.data_dict['X_hold']), 
                                                    index=self.data_dict['y_hold'].index, 
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
        num_components_kaiser = sum(sorted_eigenvalues > 1 + 1) * 2 # 2 times the kaiser criteria for num components
        self.pca_model = PCA(n_components=num_components_kaiser)
        self.data_dict['X_train'] = pd.DataFrame(self.pca_model.fit_transform(self.data_dict['X_train']), 
                                                index=self.data_dict['y_train'].index)
        self.data_dict['X_test'] = pd.DataFrame(self.pca_model.transform(self.data_dict['X_test']), 
                                                index=self.data_dict['y_test'].index)
        if self.holdout:
            self.data_dict['X_hold'] = pd.DataFrame(self.pca_model.transform(self.data_dict['X_hold']), 
                                                index=self.data_dict['y_hold'].index)
        print('...complete\n')

    def train_classification_models(self, dense_grid):
        self.models = {}
        self.tune_model_nnet(dense_grid)
        self.tune_model_svm(dense_grid)
        self.tune_model_logit()
        self.tune_model_rf(dense_grid)
        self.tune_model_gradient_boost(dense_grid)
        self.fit_logistic_ensemble()
        self.save_best_models()

    def tune_model_nnet(self, dense_grid=False):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Neural Net Model
        # ----------------------------------------------------------------------------------------------------------------------#

        print('fitting nn model')
        nn_model = MLPClassifier(solver='sgd', random_state=100)

        if dense_grid:
            param_grid = {
                'hidden_layer_sizes': [(6, 2, 2), (10, 5), (10, 10), (50, 30, 10)],  # Different layer sizes
                'alpha': [0.0001, 0.001, 0.01, 0.1, 1],  # Regularization strength
                'learning_rate_init': [0.001, 0.01, 0.1],  # Initial learning rate
                'max_iter': [200, 300, 500],  # Maximum number of iterations
                'solver': ['sgd', 'adam'],  # Different solvers
            }
        else:
            param_grid = {
                'hidden_layer_sizes': [(6, 2, 2), (10, 5)],  # Different layer sizes
                'alpha': [0.001, 0.1],  # Regularization strength
                'learning_rate_init': [0.001, 0.1],  # Initial learning rate
                'max_iter': [200, 300],  # Maximum number of iterations
                'solver': ['sgd'],  # Different solvers
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
        y_prob = best_model.predict_proba(self.data_dict['X_test'])
        y_pred = best_model.predict(self.data_dict['X_test'])

        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="nnet model", model_call=best_model)
        if self.holdout:
            y_hold_pred = best_model.predict(self.data_dict['X_hold'])
            y_hold_prob = best_model.predict_proba(self.data_dict['X_hold'])
            df_holdout_results = holdout_classification_performance(self.data_dict['y_hold'], y_hold_prob, model_name="nnet model", model_call=best_model)

        self.models['class_nnet'] = {    'best model' : best_model, 
                                         'test results' : df_results,
                                         'holdout results' : df_holdout_results,
                                         'test prob' : y_prob,
                                         'test pred' : y_pred,
                                         'y_test'    : self.data_dict['y_test'],
                                         'hold prob' : y_hold_prob,
                                         'hold pred' : y_hold_pred,
                                         'y_hold'    : self.data_dict['y_hold']
        }
        print('...complete\n')

    def tune_model_svm(self, dense_grid=False):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Support Vector Machine
        # ----------------------------------------------------------------------------------------------------------------------#

        print('Fitting SVM model')

        if dense_grid:
            # Define a more refined param grid (restrict degrees for poly, only use gamma for rbf/poly)
            param_grid = [
                {'C': np.logspace(-5, -1, 5), 'kernel': ['linear']},
                {'C': np.logspace(-5, -1, 5), 'kernel': ['rbf'], 'gamma': np.logspace(-5, -1, 5)},
                {'C': np.logspace(-5, -1, 5), 'kernel': ['poly'], 'degree': [2, 3], 'gamma': np.logspace(-5, -1, 5)}
            ]
        else: 
            param_grid = [
                {'C': [0.1], 'kernel': ['linear']},
                {'C': [0.1], 'kernel': ['rbf'], 'gamma': [0.1]},
                {'C': [0.1], 'kernel': ['poly'], 'degree': [2, 3], 'gamma': [0.1]}
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
        y_prob = best_model.predict_proba(self.data_dict['X_test'])
        y_pred = best_model.predict(self.data_dict['X_test'])

        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="SVM model", model_call=best_model)
        if self.holdout:
            y_hold_pred = best_model.predict(self.data_dict['X_hold'])
            y_hold_prob = best_model.predict_proba(self.data_dict['X_hold'])
            df_holdout_results = holdout_classification_performance(self.data_dict['y_hold'], y_hold_prob, model_name="SVM model", model_call=best_model)

        self.models['class_svm'] = {     'best model' : best_model, 
                                         'test results' : df_results,
                                         'holdout results' : df_holdout_results,
                                         'test prob' : y_prob,
                                         'test pred' : y_pred,
                                         'y_test'    : self.data_dict['y_test'],
                                         'hold prob' : y_hold_prob,
                                         'hold pred' : y_hold_pred,
                                         'y_hold'    : self.data_dict['y_hold']
        }
        print('...complete\n')

    def tune_model_logit(self):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Logistic Regression Model
        # ----------------------------------------------------------------------------------------------------------------------#
        print('Fitting Logistic Regression model')

        # Define the parameter grid for Logistic Regression
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Different regularization penalties
            'C': np.logspace(-5, 4, 20),  # Inverse of regularization strength
            'solver': ['lbfgs', 'liblinear', 'saga'],  # Different solvers
            'max_iter': [1, 2, 5, 10, 25, 50, 100, 200, 500]  # Maximum number of iterations
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
        y_prob = best_model.predict_proba(self.data_dict['X_test'])
        y_pred = best_model.predict(self.data_dict['X_test'])

        # Evaluate and save results
        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="logit model", model_call=best_model)
        if self.holdout:
            y_hold_pred = best_model.predict(self.data_dict['X_hold'])
            y_hold_prob = best_model.predict_proba(self.data_dict['X_hold'])
            df_holdout_results = holdout_classification_performance(self.data_dict['y_hold'], y_hold_prob, model_name="logit model", model_call=best_model)

        self.models['class_logit'] = {   'best model' : best_model, 
                                         'test results' : df_results,
                                         'holdout results' : df_holdout_results,
                                         'test prob' : y_prob,
                                         'test pred' : y_pred,
                                         'y_test'    : self.data_dict['y_test'],
                                         'hold prob' : y_hold_prob,
                                         'hold pred' : y_hold_pred,
                                         'y_hold'    : self.data_dict['y_hold']
        }
        print('...complete\n')

    def tune_model_rf(self, dense_grid=False):
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
                'bootstrap': [True]  # Whether bootstrap samples are used when building trees
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
        y_prob = best_model.predict_proba(self.data_dict['X_test'])
        y_pred = best_model.predict(self.data_dict['X_test'])

        # Evaluate and save results
        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="random forest", model_call=best_model)
        if self.holdout:
            y_hold_pred = best_model.predict(self.data_dict['X_hold'])
            y_hold_prob = best_model.predict_proba(self.data_dict['X_hold'])
            df_holdout_results = holdout_classification_performance(self.data_dict['y_hold'], y_hold_prob, model_name="random forest", model_call=best_model)

        self.models['class_randomForest'] = {   'best model' : best_model, 
                                                'test results' : df_results,
                                                'holdout results' : df_holdout_results,
                                                'test prob' : y_prob,
                                                'test pred' : y_pred,
                                                'y_test'    : self.data_dict['y_test'],
                                                'hold prob' : y_hold_prob,
                                                'hold pred' : y_hold_pred,
                                                'y_hold'    : self.data_dict['y_hold']
        }
        print('...complete\n')

    def tune_model_gradient_boost(self, dense_grid=False):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Gradient Boosting Model
        # ----------------------------------------------------------------------------------------------------------------------#
        print('Fitting Gradient Boosting model')

        if dense_grid:
            # Define the parameter grid for Gradient Boosting
            param_grid = {
                'n_estimators': [50, 100, 200, 500],  # Number of boosting stages
                'learning_rate': [0.001, 0.01, 0.1, 0.2],  # Step size shrinkage
                'max_depth': [3, 5, 10, 20],  # Maximum depth of individual estimators
                'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
                'min_samples_leaf': [1, 2, 4],  # Minimum samples required at each leaf node
                'subsample': [0.8, 1.0],  # Fraction of samples used for fitting each estimator
            }
        else:
            param_grid = {
                'n_estimators': [50, 100],  # Number of boosting stages
                'learning_rate': [0.001, 0.1],  # Step size shrinkage
                'max_depth': [3, 10],  # Maximum depth of individual estimators
                'min_samples_split': [2, 5],  # Minimum samples required to split a node
                'min_samples_leaf': [1, 2],  # Minimum samples required at each leaf node
                'subsample': [0.8],  # Fraction of samples used for fitting each estimator
            }

        # Set up GridSearchCV for Gradient Boosting
        grid_search = GridSearchCV(estimator=GradientBoostingClassifier(random_state=100),
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
        y_prob = best_model.predict_proba(self.data_dict['X_test'])
        y_pred = best_model.predict(self.data_dict['X_test'])

        # Evaluate and save results
        df_results = model_classification_performance(self.data_dict['y_test'], y_pred, model_name="gradient boost model", model_call=best_model)
        if self.holdout:
            y_hold_pred = best_model.predict(self.data_dict['X_hold'])
            y_hold_prob = best_model.predict_proba(self.data_dict['X_hold'])
            df_holdout_results = holdout_classification_performance(self.data_dict['y_hold'], y_hold_prob, model_name="gradient boost model", model_call=best_model)

        self.models['class_gradientBoost'] = {  'best model': best_model,
                                                'test results' : df_results,
                                                'holdout results' : df_holdout_results,
                                                'test prob' : y_prob,
                                                'test pred' : y_pred,
                                                'y_test'    : self.data_dict['y_test'],
                                                'hold prob' : y_hold_prob,
                                                'hold pred' : y_hold_pred,
                                                'y_hold'    : self.data_dict['y_hold']
        }
        print('...complete\n')

    def fit_logistic_ensemble(self, eval_metric='auc'):
        """
        Fits a logistic regression ensemble using predicted probabilities from multiple models.

        Parameters:
        - self: An object with
            - .models: dict of models with each sub-dict containing 'hold prob'
            - .data_dict['y_hold']: true binary labels (0/1)
        - eval_metric: 'auc' or 'logloss' to evaluate the model

        Returns:
        - ensemble_probs: np.array of predicted probabilities
        - weights: pd.Series of model coefficients (importance)
        - score: float, evaluation score
        """
        # Assemble model prediction probabilities
        X = pd.DataFrame({
            name: model['hold prob'][:,0]
            for name, model in self.models.items()
        })

        y = self.data_dict['y_hold']

        # Fit logistic regression
        clf = LogisticRegression(fit_intercept=True, solver='liblinear')
        clf.fit(X, y)
        
        # Get predicted probabilities
        ensemble_probs = clf.predict_proba(X)

        # Evaluate performance
        if eval_metric == 'auc':
            score = roc_auc_score(y, ensemble_probs[:, 1])
        elif eval_metric == 'logloss':
            score = log_loss(y, ensemble_probs[:, 1])
        else:
            raise ValueError("eval_metric must be 'auc' or 'logloss'")

        # Get model coefficients
        weights = pd.Series(clf.coef_[0], index=X.columns)

        ensemble_results = holdout_classification_performance(self.data_dict['y_hold'], 
                                                              ensemble_probs,
                                                              model_name='ensemble', 
                                                              model_call=weights)

        self.ensemble = {'ensemble probs' : ensemble_probs,
                         'score' : score,
                         'model' : clf,
                         'holdout performance' : ensemble_results}

        return ensemble_probs, score, ensemble_results

    def save_best_models(self):
        print('saving best models...')
        datestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_filename = f'best_models_{datestamp}.pkl'
        save_data_dict = {'X_hold' : self.data_dict['X_hold'],
                          'y_hold' : self.data_dict['y_hold'],
                          'X_test' : self.data_dict['X_test'],
                          'y_test' : self.data_dict['y_test'],
                          'features' : self.data_dict['features']}
        save_dict = {'models' : self.models,
                     'data' : save_data_dict,
                     'pca' : False,
                     'scaler' : False,
                     'ensemble' : self.ensemble}
        if self.scaler_used:
            save_dict['scaler'] = self.scaler
        if self.pca_used:
            save_dict['pca'] = self.pca_model
        joblib.dump(save_dict, model_filename)
        print('...complete\n')

WL_models = trainClassificationModel()
print("complete")
