import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import LogisticRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, StackingRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import mean_squared_error, confusion_matrix, f1_score, roc_auc_score, log_loss, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.decomposition import PCA
import sqlite3
import pickle
import itertools
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import time
import logging
from run_log import RunLog, LOGGER_NAME
# quiet by default; RunLog re-enables warnings while a run is in progress so they land in the log
warnings.filterwarnings('ignore')
logger = logging.getLogger(LOGGER_NAME)

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

class trainModel:
    def __init__(self, 
                 conn=sqlite3.connect('nba_database_test.db'),
                 model_type='classification',
                 target_response='WL',
                 scale=True,
                 pca=True,
                 dense_grid=True):
        
        self.conn = conn
        self.run_log = RunLog(conn, response=target_response, model_type=model_type,
                              settings={'scale': scale, 'pca': pca, 'dense_grid': dense_grid})

        with self.run_log:
            self.dataset = self.load_modeling_dataset()
            self.select_features()

            self.response = target_response
            self.select_response(target_response)

            self.partition_by_split_label()
            self.run_log.log_dataset(self.features, self.responses,
                                     self.dataset_split.loc[self.features.index], self.rows_dropped_for_nan)
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
            if model_type == 'regression':
                self.train_regression_models(dense_grid)
        print('complete')

    #--- Preprocess ---#
    def load_modeling_dataset(self, db_table_name='modeling_dataset'):
        '''
        load the prebuilt modeling dataset (features, responses and split labels) from the database
        '''
        print(f'loading {db_table_name} from database...')
        df = pd.read_sql(f'SELECT * FROM {db_table_name}', self.conn)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        print('...complete\n')
        return df

    def select_features(self, feature_set='basic_v2', use_all=False):
        '''
        keep the columns of the chosen feature set, drop incomplete rows, and hold on to
        each row's dataset_split label
        '''
        print('selecting features...')
        feature_cols = pd.read_excel(r'NBA_Stats_v3/catalogs/parameters/feature_sets.xlsx',
                             sheet_name=feature_set)['feature_cols'].tolist()
        feature_cols = [col.strip().replace("'", "") for col in feature_cols]
        rows_before = len(self.dataset)
        self.dataset = self.dataset.dropna()
        self.rows_dropped_for_nan = rows_before - len(self.dataset)
        missing = [c for c in feature_cols if c not in self.dataset.columns]
        if missing:
            logger.error(f"feature set '{feature_set}' lists {len(missing)} columns not in the dataset: {missing}")
        self.dataset_split = self.dataset['dataset_split']
        if not use_all:
            self.features = self.dataset[feature_cols]
        else:
            self.features = self.dataset.drop(columns=['dataset_split'])
        print('...complete\n')

    def select_response(self, target_response):
        print('selecting response...')
        self.responses = self.dataset.loc[self.features.index, f'{target_response}_r']
        print('...complete\n')

    def partition_by_split_label(self, holdout=True):
        '''
        distribute rows into the data dictionary according to the dataset_split labels
        already assigned by build_datasets.py (chronological holdout for validation, random
        train/test split of the remainder) - nothing is re-split here
        '''
        print('partitioning data by split label...')
        self.holdout = holdout
        split = self.dataset_split.loc[self.features.index]

        self.data_dict = {}
        if holdout:
            self.data_dict['X_hold'] = self.features[split == 'validation']
            self.data_dict['y_hold'] = self.responses[split == 'validation']

        self.data_dict.update({
            'X_train': self.features[split == 'train'],
            'X_test': self.features[split == 'test'],
            'y_train': self.responses[split == 'train'],
            'y_test': self.responses[split == 'test'],
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
        num_components_kaiser = min(num_components_kaiser, self.data_dict['X_train'].shape[1]) # ensure we don't exceed the number of features
        print("using PCA with", num_components_kaiser, "components")
        self.pca_model = PCA(n_components=num_components_kaiser)
        self.data_dict['X_train'] = pd.DataFrame(self.pca_model.fit_transform(self.data_dict['X_train']), 
                                                index=self.data_dict['y_train'].index)
        logger.info(f'PCA kept {num_components_kaiser} components from {len(self.features.columns)} features, '
                    f'explaining {self.pca_model.explained_variance_ratio_.sum():.1%} of the variance')
        self.data_dict['X_test'] = pd.DataFrame(self.pca_model.transform(self.data_dict['X_test']), 
                                                index=self.data_dict['y_test'].index)
        if self.holdout:
            self.data_dict['X_hold'] = pd.DataFrame(self.pca_model.transform(self.data_dict['X_hold']), 
                                                index=self.data_dict['y_hold'].index)
        print('...complete\n')

    #--- Classifiers ---#
    def train_classification_models(self, dense_grid):
        self.models = {}
        self.tune_model_nnet(dense_grid)
        # self.tune_model_svm(dense_grid)
        self.tune_model_logit()
        # self.tune_model_rf(dense_grid)
        # self.tune_model_gradient_boost(dense_grid)
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

        self.models[f'{self.response}_clf_nnet'] = {    'best model' : best_model, 
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

        self.models[f'{self.response}_clf_svm'] = {     'best model' : best_model, 
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

        self.models[f'{self.response}_clf_logit'] = {   'best model' : best_model, 
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

        self.models[f'{self.response}_clf_randomForest'] = {   'best model' : best_model, 
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

    def tune_model_gradient_boost_old(self, dense_grid=False):
        # ----------------------------------------------------------------------------------------------------------------------#
        # Gradient Boosting Model
        # ----------------------------------------------------------------------------------------------------------------------#
        print('Fitting Gradient Boosting model')

        if dense_grid:
            # Define the parameter grid for Gradient Boosting
            param_grid = {
                'n_estimators': [100, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.6, 0.8, 1.0],
            }
        else:
            param_grid = {
                'n_estimators': [100],  # Number of boosting stages
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

    def tune_model_gradient_boost(self, dense_grid=False):
        print('Fitting Gradient Boosting model')

        if dense_grid:
            param_grid = {
                'n_estimators': [100, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.6, 0.8, 1.0],
            }
        else:
            param_grid = {
                'n_estimators': [100],
                'learning_rate': [0.001, 0.1],
                'max_depth': [3, 10],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'subsample': [0.8],
            }

        # Use stratified CV for classification
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            estimator=GradientBoostingClassifier(random_state=100),
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )

        start_time = time.time()
        grid_search.fit(self.data_dict['X_train'], self.data_dict['y_train'])
        end_time = time.time()

        print(f"Grid search completed in {end_time - start_time:.2f} seconds")

        best_params = grid_search.best_params_
        print("Best Parameters from Grid Search:", best_params)

        # Visualization: Create a DataFrame from cv_results_
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("gradient_boost_grid_results.csv", index=False)

        # # Simple heatmap or scatter if 2 params vary
        # if len(param_grid) == 2:
        #     pivot = results_df.pivot_table(values="mean_test_score",
        #                                 index=list(param_grid.keys())[0],
        #                                 columns=list(param_grid.keys())[1])
        #     sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        #     plt.title("Grid Search Accuracy")
        #     plt.savefig("gradient_boost_grid_search_heatmap.png")
        #     plt.close()
        # else:
        #     # Plot accuracy vs learning rate or n_estimators
        #     for param in ['learning_rate', 'n_estimators', 'max_depth']:
        #         if param in param_grid:
        #             plt.figure()
        #             sns.lineplot(data=results_df, x=param, y='mean_test_score')
        #             plt.title(f'Accuracy vs {param}')
        #             plt.savefig(f"accuracy_vs_{param}.png")
        #             plt.close()

        best_model = grid_search.best_estimator_
        y_prob = best_model.predict_proba(self.data_dict['X_test'])
        y_pred = best_model.predict(self.data_dict['X_test'])

        df_results = model_classification_performance(
            self.data_dict['y_test'], y_pred,
            model_name="gradient boost model", model_call=best_model
        )

        y_hold_pred, y_hold_prob, df_holdout_results = None, None, None
        if self.holdout:
            y_hold_pred = best_model.predict(self.data_dict['X_hold'])
            y_hold_prob = best_model.predict_proba(self.data_dict['X_hold'])
            df_holdout_results = holdout_classification_performance(
                self.data_dict['y_hold'], y_hold_prob,
                model_name="gradient boost model", model_call=best_model
            )

        self.models[f'{self.response}_clf_gradientBoost'] = {
            'best model': best_model,
            'test results': df_results,
            'holdout results': df_holdout_results,
            'test prob': y_prob,
            'test pred': y_pred,
            'y_test': self.data_dict['y_test'],
            'hold prob': y_hold_prob,
            'hold pred': y_hold_pred,
            'y_hold': self.data_dict['y_hold']
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

    #--- Regressors ---#
    def train_regression_models(self, dense_grid):
        self.models = {}
        self.tune_model_nnet_regressor(dense_grid)
        self.tune_model_svr(dense_grid)
        self.tune_model_elastic_net()
        # self.tune_model_rf(dense_grid)
        # self.tune_model_gradient_boost(dense_grid)
        self.fit_stack_ensemble(dense_grid)
        self.save_best_models()

    def eval_regression_model(self, grid_search, model_type: str=''):
        '''
        standardize model performance evaluation and artifact saving for regressors
        '''

        # Best parameters/model
        best_params = grid_search.best_params_
        print("Best Parameters from Grid Search:", best_params)
        best_model = grid_search.best_estimator_

        # Predict on test
        X_test = self.data_dict['X_test']
        y_test = np.asarray(self.data_dict['y_test'])
        if y_test.ndim == 2 and y_test.shape[1] == 1:
            y_test = y_test.ravel()

        y_pred = best_model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        df_results = pd.DataFrame([{
            'model': 'nnet regressor',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mse': mse,
            'params': best_params
        }])

        # Holdout (optional)
        df_holdout_results = None
        y_hold_pred = None
        if getattr(self, 'holdout', True) and 'X_hold' in self.data_dict and 'y_hold' in self.data_dict:
            X_hold = self.data_dict['X_hold']
            y_hold = np.asarray(self.data_dict['y_hold'])
            if y_hold.ndim == 2 and y_hold.shape[1] == 1:
                y_hold = y_hold.ravel()

            y_hold_pred = best_model.predict(X_hold)
            hold_mse = mean_squared_error(y_hold, y_hold_pred)
            hold_rmse = float(np.sqrt(hold_mse))
            hold_mae = mean_absolute_error(y_hold, y_hold_pred)
            hold_r2 = r2_score(y_hold, y_hold_pred)

            df_holdout_results = pd.DataFrame([{
                'model': f'{model_type} regressor',
                'rmse': hold_rmse,
                'mae': hold_mae,
                'r2': hold_r2,
                'mse': hold_mse,
                'params': best_params
            }])

        # Store artifacts 
        self.models[f'{self.response}_reg_{model_type}'] = {
            'best model': best_model,
            'best params': best_params,
            'cv results': grid_search.cv_results_,
            'test results': df_results,
            'test pred': y_pred,
            'y_test': self.data_dict['y_test'],
            'holdout results': df_holdout_results,
            'hold pred': y_hold_pred,
            'y_hold': self.data_dict.get('y_hold', None),
        }

        print(f"Test — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        if df_holdout_results is not None:
            print(f"Holdout — RMSE: {df_holdout_results['rmse'].iloc[0]:.4f} | "
                f"MAE: {df_holdout_results['mae'].iloc[0]:.4f} | "
                f"R²: {df_holdout_results['r2'].iloc[0]:.4f}")
        print('...complete\n')

    def tune_model_nnet_regressor(self, dense_grid: bool = False):
        # ------------------------------------------------------------------------------------------------------------------ #
        # Neural Net Regressor (MLPRegressor)
        # ------------------------------------------------------------------------------------------------------------------ #
        print('fitting nn regressor')

        # Base model (data already scaled per you). Keep early_stopping for speed/regularization.
        self.nn_model = MLPRegressor(
            random_state=100,
            early_stopping=True,          # requires solver='adam'
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=300
        )

        if dense_grid:
            param_grid = {
                'hidden_layer_sizes': [(32,), (64,), (64, 32), (128, 64, 32)],
                'alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                'learning_rate_init': [1e-3, 3e-3, 1e-2],
                'max_iter': [300, 500],
                'solver': ['adam'],                  # keep adam for early_stopping
                'activation': ['relu', 'tanh'],
            }
        else:
            param_grid = {
                'hidden_layer_sizes': [(32,)], #, (64, 32)],
                'alpha': [1e-4], #, 1e-3],
                'learning_rate_init': [1e-3, 1e-2],
                'max_iter': [300],
                'solver': ['adam'],
                'activation': ['relu'],
            }

        # Multi-metric scoring; refit on lowest MSE (note sklearn uses negative losses for “higher is better”)
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2',
        }

        grid_search = GridSearchCV(
            estimator=self.nn_model,
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='neg_mse',             # best by MSE; switch to 'r2' if you prefer
            n_jobs=-1,
            verbose=2
        )

        print('Fitting nn regressor with Grid Search')

        X_train = self.data_dict['X_train']
        y_train = self.data_dict['y_train']

        # Ensure y is proper shape
        y_train = np.asarray(y_train)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()  # single-target

        grid_search.fit(X_train, y_train)

        self.eval_regression_model(grid_search,
                                   model_type='nnet')
        # # Best parameters/model
        # best_params = grid_search.best_params_
        # print("Best Parameters from Grid Search:", best_params)
        # best_model = grid_search.best_estimator_

        # # Predict on test
        # X_test = self.data_dict['X_test']
        # y_test = np.asarray(self.data_dict['y_test'])
        # if y_test.ndim == 2 and y_test.shape[1] == 1:
        #     y_test = y_test.ravel()

        # y_pred = best_model.predict(X_test)

        # # Metrics
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = float(np.sqrt(mse))
        # mae = mean_absolute_error(y_test, y_pred)
        # r2 = r2_score(y_test, y_pred)

        # df_results = pd.DataFrame([{
        #     'model': 'nnet regressor',
        #     'rmse': rmse,
        #     'mae': mae,
        #     'r2': r2,
        #     'mse': mse,
        #     'params': best_params
        # }])

        # # Holdout (optional)
        # df_holdout_results = None
        # y_hold_pred = None
        # if getattr(self, 'holdout', False) and 'X_hold' in self.data_dict and 'y_hold' in self.data_dict:
        #     X_hold = self.data_dict['X_hold']
        #     y_hold = np.asarray(self.data_dict['y_hold'])
        #     if y_hold.ndim == 2 and y_hold.shape[1] == 1:
        #         y_hold = y_hold.ravel()

        #     y_hold_pred = best_model.predict(X_hold)
        #     hold_mse = mean_squared_error(y_hold, y_hold_pred)
        #     hold_rmse = float(np.sqrt(hold_mse))
        #     hold_mae = mean_absolute_error(y_hold, y_hold_pred)
        #     hold_r2 = r2_score(y_hold, y_hold_pred)

        #     df_holdout_results = pd.DataFrame([{
        #         'model': 'nnet regressor',
        #         'rmse': hold_rmse,
        #         'mae': hold_mae,
        #         'r2': hold_r2,
        #         'mse': hold_mse,
        #         'params': best_params
        #     }])

        # # Store artifacts 
        # self.models[f'{self.response}_reg_nnet'] = {
        #     'best model': best_model,
        #     'best params': best_params,
        #     'cv results': grid_search.cv_results_,
        #     'test results': df_results,
        #     'test pred': y_pred,
        #     'y_test': self.data_dict['y_test'],
        #     'holdout results': df_holdout_results,
        #     'hold pred': y_hold_pred,
        #     'y_hold': self.data_dict.get('y_hold', None),
        # }

        # print(f"Test — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        # if df_holdout_results is not None:
        #     print(f"Holdout — RMSE: {df_holdout_results['rmse'].iloc[0]:.4f} | "
        #         f"MAE: {df_holdout_results['mae'].iloc[0]:.4f} | "
        #         f"R²: {df_holdout_results['r2'].iloc[0]:.4f}")
        # print('...complete\n')

    def tune_model_svr(self, dense_grid: bool = False):
        # ------------------------------------------------------------------------------------------------------------------ #
        # SV Regression
        # ------------------------------------------------------------------------------------------------------------------ #
        print('fitting support vector regressor')

        # Base model (data already scaled per you). Keep early_stopping for speed/regularization.

        self.svr_model = SVR(kernel='rbf')  # data already scaled

        if dense_grid:
            param_grid = {
                "kernel": ["rbf"],                     # keep to rbf; add "linear","poly" if desired
                "C": [0.5, 1, 3, 10, 30, 100],
                "epsilon": [0.01, 0.05, 0.1, 0.2],
                "gamma": ["scale", "auto", 0.01, 0.03, 0.1],
            }
        else:
            param_grid = {
                "kernel": ["rbf"],
                "C": [1, 3],#[1, 3, 10],
                "epsilon": [0.05], #, 0.1],
                "gamma": ["scale", 0.03],
            }
        # Multi-metric scoring; refit on lowest MSE (note sklearn uses negative losses for “higher is better”)
        scoring = {
                'neg_mse': 'neg_mean_squared_error',
                'neg_mae': 'neg_mean_absolute_error',
                'r2': 'r2',
            }

        grid_search = GridSearchCV(
            estimator=self.svr_model,
            param_grid=param_grid,
            cv=5,
            refit='neg_mse',             # best by MSE; switch to 'r2' if you prefer
            n_jobs=-1,
            verbose=2
        )

        print('Fitting nn regressor with Grid Search')

        X_train = self.data_dict['X_train']
        y_train = self.data_dict['y_train']

        # Ensure y is proper shape
        y_train = np.asarray(y_train)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()  # single-target

        grid_search.fit(X_train, y_train)

        self.eval_regression_model(grid_search,
                            model_type='svr')

        # # Best parameters/model
        # best_params = grid_search.best_params_
        # print("Best Parameters from Grid Search:", best_params)
        # best_model = grid_search.best_estimator_

        # # Predict on test
        # X_test = self.data_dict['X_test']
        # y_test = np.asarray(self.data_dict['y_test'])
        # if y_test.ndim == 2 and y_test.shape[1] == 1:
        #     y_test = y_test.ravel()

        # y_pred = best_model.predict(X_test)

        # # Metrics
        # mse = mean_squared_error(y_test, y_pred)
        # rmse = float(np.sqrt(mse))
        # mae = mean_absolute_error(y_test, y_pred)
        # r2 = r2_score(y_test, y_pred)

        # df_results = pd.DataFrame([{
        #     'model': 'elastic net',
        #     'rmse': rmse,
        #     'mae': mae,
        #     'r2': r2,
        #     'mse': mse,
        #     'params': best_params
        # }])

        # # Holdout (optional)
        # df_holdout_results = None
        # y_hold_pred = None
        # if getattr(self, 'holdout', False) and 'X_hold' in self.data_dict and 'y_hold' in self.data_dict:
        #     X_hold = self.data_dict['X_hold']
        #     y_hold = np.asarray(self.data_dict['y_hold'])
        #     if y_hold.ndim == 2 and y_hold.shape[1] == 1:
        #         y_hold = y_hold.ravel()

        #     y_hold_pred = best_model.predict(X_hold)
        #     hold_mse = mean_squared_error(y_hold, y_hold_pred)
        #     hold_rmse = float(np.sqrt(hold_mse))
        #     hold_mae = mean_absolute_error(y_hold, y_hold_pred)
        #     hold_r2 = r2_score(y_hold, y_hold_pred)

        #     df_holdout_results = pd.DataFrame([{
        #         'model': 'elastic net',
        #         'rmse': hold_rmse,
        #         'mae': hold_mae,
        #         'r2': hold_r2,
        #         'mse': hold_mse,
        #         'params': best_params
        #     }])

        # # Store artifacts 
        # self.models[f'{self.response}_reg_svr'] = {
        #     'best model': best_model,
        #     'best params': best_params,
        #     'cv results': grid_search.cv_results_,
        #     'test results': df_results,
        #     'test pred': y_pred,
        #     'y_test': self.data_dict['y_test'],
        #     'holdout results': df_holdout_results,
        #     'hold pred': y_hold_pred,
        #     'y_hold': self.data_dict.get('y_hold', None),
        # }

        # print(f"Test — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        # if df_holdout_results is not None:
        #     print(f"Holdout — RMSE: {df_holdout_results['rmse'].iloc[0]:.4f} | "
        #         f"MAE: {df_holdout_results['mae'].iloc[0]:.4f} | "
        #         f"R²: {df_holdout_results['r2'].iloc[0]:.4f}")
        # print('...complete\n')

    def tune_model_elastic_net(self, dense_grid: bool = False):
        # ------------------------------------------------------------------------------------------------------------------ #
        # Elastic Net Regression
        # ------------------------------------------------------------------------------------------------------------------ #
        print('fitting elastic net regressor')

        # Base model (data already scaled per you). Keep early_stopping for speed/regularization.
        self.enet_model = ElasticNet(
            alpha=0.001,
            l1_ratio=0.5,
            random_state=42
            )

        if dense_grid:
            param_grid = {
                "alpha": [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0],     # Regularization strength
                "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],        # Mix between L1 (Lasso) and L2 (Ridge)
                "max_iter": [500, 1000, 2000],                  # Iteration caps
                "selection": ["cyclic", "random"],              # Coordinate descent update rule
                "tol": [1e-4, 1e-3],                            # Convergence tolerance
            }
        else:
            param_grid = {
                "alpha": [1e-3],# 1e-2, 0.1],                     # Regularization strength
                "l1_ratio": [0.25],#, 0.5, 0.75],                  # L1/L2 balance
                "max_iter": [500, 1000],                        # Iteration caps
                "selection": ["cyclic"],                        # Simpler option
                "tol": [1e-4],                                  # Convergence tolerance
            }

        # Multi-metric scoring; refit on lowest MSE (note sklearn uses negative losses for “higher is better”)
        scoring = {
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2',
        }

        grid_search = GridSearchCV(
            estimator=self.enet_model,
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            refit='neg_mse',             # best by MSE; switch to 'r2' if you prefer
            n_jobs=-1,
            verbose=2
        )

        print('Fitting nn regressor with Grid Search')

        X_train = self.data_dict['X_train']
        y_train = self.data_dict['y_train']

        # Ensure y is proper shape
        y_train = np.asarray(y_train)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train = y_train.ravel()  # single-target

        grid_search.fit(X_train, y_train)

        self.eval_regression_model(grid_search,
                                   model_type='enet')

    def fit_stack_ensemble(self, eval_metric='auc'):
        """
        Fits a stacking ridge regression ensemble using fitted regression models.

        Parameters:
        - self: An object with
            - .models: dict of models
            - .data_dict['y_hold']
        - eval_metric: 'auc' or 'logloss' to evaluate the model

        Returns:
        - ensemble_probs: np.array of predicted probabilities
        - weights: pd.Series of model coefficients (importance)
        - score: float, evaluation score
        """
        # Assemble model prediction probabilities
        X = pd.DataFrame({
            name: model['test pred']
            for name, model in self.models.items()
        })

        y = self.data_dict['y_test']

        stack = StackingRegressor(
            estimators=[
                ("svr", self.svr_model),
                ("nnet", self.nn_model),
                ("enet", self.enet_model)
            ],
            final_estimator=Ridge(alpha=1.0, positive=True, random_state=42),
            cv=5,
            n_jobs=-1,
            passthrough=False  # set to True if you want to include original features
            )
        # Train
        stack.fit(X, y)

        # # Test
        # y_hat = stack.predict(self.data_dict['X_test'])
        # rmse = mean_squared_error(self.data_dict['y_test'], y_hat, squared=False)
        # mae  = mean_absolute_error(self.data_dict['y_test'], y_hat)
        # r2   = r2_score(self.data_dict['y_test'], y_hat)
        # print(f"Stack — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        
        X = pd.DataFrame({
            name: model['hold pred']
            for name, model in self.models.items()
        })

        y = self.data_dict['y_hold']

        y_hat = stack.predict(X)
        rmse = mean_squared_error(y, y_hat, squared=False)
        mae  = mean_absolute_error(y, y_hat)
        r2   = r2_score(y, y_hat)
        print(f"Stack — RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")
        print('complete')

        # ensemble_results = holdout_classification_performance(self.data_dict['y_hold'], 
        #                                                       ensemble_probs,
        #                                                       model_name='ensemble', 
        #                                                       model_call=weights)

        # self.ensemble = {'ensemble probs' : ensemble_probs,
        #                  'score' : score,
        #                  'model' : stack,
        #                  'holdout performance' : ensemble_results}

        # return ensemble_probs, score, ensemble_results

    def save_best_models(self):
        print('saving best models...')
        self.run_log.log_models(self.models)
        datestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        model_filename = f'best_models_{self.response}_{datestamp}.pkl'
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
        self.run_log.set_artifact(model_filename)
        print('...complete\n')


WL_models = trainModel(dense_grid=False,
                       target_response='WL',
                       model_type='classification')
# PTS_models = trainModel(dense_grid=True,
#                                   target_response='PTS_per48',
#                                   model_type='regression')
print("complete")
