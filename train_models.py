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
# Ignore all warnings
warnings.filterwarnings('ignore')

def load_feature_data():
    # import/process
    conn = sqlite3.connect('nba_database_2024-08-18.db')
    # df_feat = pd.read_csv('features.csv')
    df_feat = pd.read_sql('SELECT * FROM feature_table', conn)
    df_feat['GAME_DATE'] = pd.to_datetime(df_feat['GAME_DATE'])
    df_feat = df_feat.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
    # df_feat = df_feat.sort_index(level=['TEAM_ABBREVIATION', 'GAME_DATE'])
    return df_feat

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

df_feat = load_feature_data()

# fix
# cols_to_not_shift = ['DaysElapsed', 'DaysRest', 'DaysRest_opp',
#         'Matchup', 'WL', 'Home', 'roadtrip', 'roadtrip_opp',
#         'MATCHUP', 'Poss_r', 'OffRat_r', 'DefRat_r', 'PTS_per48_r', 'PTS_per48_against_r', 'PTS_per48_diff_r']
# prev_matchup_cols = ['OREB_per48', 'OREB%_z', 'DREB_per48', 'DREB%_z', 'REB_per48',
#                      'FGA_per48', 'FG3A_per48', 'FTA_per48',
#                      'WL', 'OffRat', 'DefRat', 'OffRat_z', 'DefRat_z',
#                      'PTS_per48_z', 'PTS_per48_against_z',
#                      'FGM_per48_z', 'FGA_per48_z', 'FG3M_per48_z', 'FG3A_per48_z',
#                      'FTM_per48_z', 'FTA_per48_z', 'FGM_per48_against_z', 'FGA_per48_against_z',
#                      'FG3M_per48_against_z', 'FG3A_per48_against_z', 'FTM_per48_against_z', 'FTA_per48_against']
# # prev_matchup_cols = [col + '_prev' for col in prev_matchup_cols]
# # cols_to_not_shift = cols_to_not_shift + prev_matchup_cols
# # cols_to_shift = [col for col in df_feat.columns if col not in cols_to_not_shift]
# # df_feat[cols_to_shift] = df_feat.groupby('TEAM_ABBREVIATION',
# #            group_keys=False)[cols_to_shift].shift(1)
df_feat = df_feat.dropna()
# get list of potential response variables
response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']

# select features to use in model
# feature_cols = [
# 'Win',
# 'Win_opp',
# 'AST%_z',
# 'DaysRest',
# 'DaysRest_opp',
# 'DefRat_prev',
# 'DefRat_z',
# 'DefRat_z_prev',
# 'DREB_per48_prev',
# 'DREB_per48_z',
# 'DREB%_z',
# 'DREB%_z_prev',
# 'eFG%_against_z',
# 'eFG%_z',
# 'FG3A_per48_against_z',
# 'FG3A_per48_against_z_prev',
# 'FG3A_per48_prev',
# 'FG3A_per48_z',
# 'FG3A_per48_z_prev',
# 'FG3M_per48_against_z',
# 'FG3M_per48_against_z_prev',
# 'FG3M_per48_z',
# 'FG3M_per48_z_prev',
# 'FGA_per48_against_z',
# 'FGA_per48_against_z_prev',
# 'FGA_per48_prev',
# 'FGA_per48_z',
# 'FGA_per48_z_prev',
# 'FGM_per48_against_z',
# 'FGM_per48_against_z_prev',
# 'FGM_per48_z',
# 'FGM_per48_z_prev',
# 'FTA_per48_against_prev',
# 'FTA_per48_against_z',
# 'FTA_per48_z',
# 'FTM_per48_against_z',
# 'FTM_per48_against_z_prev',
# 'FTM_per48_z',
# 'FTM_per48_z_prev',
# 'Home',
# 'OffRat_prev',
# 'OffRat_z',
# 'OffRat_z_prev',
# 'OREB_per48_prev',
# 'OREB_per48_z',
# 'OREB%_z',
# 'OREB%_z_prev',
# 'Poss',
# 'Poss_opp',
# 'PTS_per48_against_z',
# 'PTS_per48_against_z_prev',
# 'PTS_per48_z',
# 'PTS_per48_z_prev',
# 'roadtrip',
# 'roadtrip_opp',
# 'STL%_z',
# 'TOV_forced%_z',
# 'TOV%_z',
# 'TS%_against_z',
# 'TS%_z'
# ]

# feature_cols = pd.read_excel(r'/Users/johntweedie/Dev/Projects/PN24001_NBA_Stats/catalogs/features_cols.xlsx',
#                              sheet_name='pairwise')['feature_cols'].tolist()
# feature_cols = [col.strip().replace("'", "") for col in feature_cols]
#
# def create_pairwise_data(df):
#     pair_data = []
#     indices = []
#
#     # Group by 'GAME_ID' to get pairs of observations within each game
#     for game_id, group in df_feat.groupby(level='GAME_ID'):
#         # Reset index to work with individual rows
#         group = group.reset_index(level='GAME_ID', drop=True)
#         for i in range(len(group)):
#             for j in range(i + 1, len(group)):
#                 obj1 = group.iloc[i]
#                 obj2 = group.iloc[j]
#
#                 # Calculate differences only for the specified feature columns
#                 feature_diff = obj1[feature_cols] - obj2[feature_cols]
#
#                 # Label is 1 if obj1 'outcome' is better than obj2 'outcome', else 0
#                 label = 1 if obj1['WL'] > obj2['WL'] else 0
#
#                 indices.append((group.index[i], group.index[j]))
#
#                 # Combine the feature differences and the label into one array
#                 pair_data.append(np.hstack([feature_diff.values, label]))
#
#     return np.array(pair_data), indices
#
# # Apply the function to the dataset
# pairwise_data, pair_indices = create_pairwise_data(df_feat)
# pair_indices = [str(pair[0][0]).split(' ')[0] + ' ' + pair[0][1] + ' vs. ' + pair[1][1] for pair in pair_indices]
# pairwise_data = pd.DataFrame(pairwise_data, columns=feature_cols+['WL'], index=pair_indices)


# X = pairwise_data[feature_cols]
# y = pairwise_data['WL']

feature_cols = pd.read_excel(r'/Users/johntweedie/Dev/Projects/PN24001_NBA_Stats/catalogs/features_cols.xlsx',
                             sheet_name='nnet')['feature_cols'].tolist()
feature_cols = [col.strip().replace("'", "") for col in feature_cols]

print('segmenting test/train set')
X = df_feat[feature_cols]
y = df_feat['WL']

# get hold out set
n_games_for_holdout = 8
X_holdout = X.iloc[-8:]
X = X.iloc[:-8]

y_holdout = y.iloc[-8:]
y = y.iloc[:-8]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), index=y_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), index=y_test.index)
X_holdout = pd.DataFrame(scaler.transform(X_holdout), index=y_holdout.index)

# Compute the covariance matrix
cov_matrix = np.cov(X_train.T)
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
# Sort eigenvalues in descending order
sorted_eigenvalues = np.sort(eigenvalues)[::-1]
# Plot the scree plot
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(sorted_eigenvalues) + 1), sorted_eigenvalues, marker='o', linestyle='-')
# plt.xlabel('Number of Principal Components')
# plt.ylabel('Eigenvalue')
# plt.title('Scree Plot')
# plt.grid(True)
# plt.show()

# Determine the number of components to keep
# Using Kaiser Criterion (eigenvalues > 1)
num_components_kaiser = sum(sorted_eigenvalues > 1 + 1)
print(f"Number of components to keep (Kaiser Criterion): {num_components_kaiser}")

# PCA
print('performing PCA')
# n_components = 16
pca = PCA(n_components=num_components_kaiser)
X_train_pca = pd.DataFrame(pca.fit_transform(X_train), index=y_train.index)
X_test_pca = pd.DataFrame(pca.transform(X_test), index=y_test.index)
X_holdout_pca = pd.DataFrame(pca.transform(X_holdout), index=y_holdout.index)


explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")

print('saving holdout data to pickle')
data_dict = {
    'X_holdout_pca': X_holdout_pca,
    'X_holdout': X_holdout,
    'y_holdout': y_holdout,
    'X_train' : X_train,
    'X_train_pca' : X_train_pca,
    'pca' : pca
}

# Save the dictionary to a pickle file
with open('holdout_data.pkl', 'wb') as f:
    pickle.dump(data_dict, f)


def tune_model_nnet():
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
    grid_search.fit(X_train, y_train)

    # Best parameters from grid search
    best_params = grid_search.best_params_
    print("Best Parameters from Grid Search:", best_params)

    # Use the best estimator to predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    df_results = model_classification_performance(y_test, y_pred, model_name="nn model", model_call=best_model)

    print('complete, saving best model')
    # Save the best model as a .pkl file
    model_filename = 'best_nn_mode_pca.pkl'
    joblib.dump(best_model, model_filename)

    return df_results, grid_search

def tune_model_svm():
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
    # grid_search = RandomizedSearchCV(estimator=SVC(),
    #                                  param_distributions=param_grid,
    #                                  n_iter=5,
    #                                  cv=5,
    #                                  scoring='accuracy',
    #                                  n_jobs=-1,
    #                                  verbose=2)
    grid_search = GridSearchCV(estimator=SVC(probability=True),
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=2)

    grid_search.fit(X_train, y_train)

    # Best parameters from grid search
    best_params = grid_search.best_params_
    print(f"Best Parameters from Grid Search: {best_params}")

    # Use the best estimator to predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    df_results = model_classification_performance(y_test, y_pred, model_name="SVM model", model_call=best_model)

    # Save the best model
    model_filename = 'best_svm_model_pca.pkl'
    joblib.dump(best_model, model_filename)
    print(f"Model saved as {model_filename}")

    return df_results, grid_search

def tune_model_logreg():
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

    print('Fitting Logistic Regression model with Grid Search')
    grid_search.fit(X_train, y_train)

    # Best parameters from grid search
    best_params = grid_search.best_params_
    print("Best Parameters from Grid Search:", best_params)

    # Use the best estimator to predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate and save results
    df_results = model_classification_performance(y_test, y_pred, model_name="Logistic Regression", model_call=best_model)

    print('Complete, saving best model')
    # Save the best model as a .pkl file
    model_filename = 'best_logreg_model_pca.pkl'
    joblib.dump(best_model, model_filename)

    return df_results, grid_search

def tune_model_rf():
    # ----------------------------------------------------------------------------------------------------------------------#
    # Random Forest Model
    # ----------------------------------------------------------------------------------------------------------------------#
    print('Fitting Random Forest model')

    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 500],  # Number of trees in the forest
        'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],  # Minimum samples required to be at a leaf node
        'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    }

    # Set up GridSearchCV for Random Forest
    grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=100),
                               param_grid=param_grid,
                               cv=5,
                               scoring='accuracy',
                               n_jobs=-1,
                               verbose=2)

    print('Fitting Random Forest model with Grid Search')
    grid_search.fit(X_train, y_train)

    # Best parameters from grid search
    best_params = grid_search.best_params_
    print("Best Parameters from Grid Search:", best_params)

    # Use the best estimator to predict
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate and save results
    df_results = model_classification_performance(y_test, y_pred, model_name="Random Forest", model_call=best_model)

    print('Complete, saving best model')
    # Save the best model as a .pkl file
    model_filename = 'best_rf_model_pca.pkl'
    joblib.dump(best_model, model_filename)

    return df_results, grid_search


df_results_nnet, grid_search_nnet = tune_model_nnet()
df_results_svm, grid_search_svm = tune_model_svm()
df_results_logreg, grid_search_logreg = tune_model_logreg()
df_results_rf, grid_search_rf = tune_model_rf()

report = {
    'Model Performance Summary': pd.concat([df_results_nnet, df_results_svm, df_results_logreg, df_results_rf]),
    'Feature Set': feature_cols,
    'PCA Components': pca,
    'Fit Models with PCA?': 'No'
}

# Export the report to a text file
with open('model_log.txt', 'w') as f:
    # Write the model performance summary
    f.write("Model Performance Summary:\n")
    f.write(report['Model Performance Summary'].to_string(index=False))  # Converting DataFrame to string

    f.write("\n\n")  # Adding some space between sections

    # Write the feature set
    f.write("Feature Set:\n")
    f.write(', '.join(report['Feature Set']))  # Joining the feature list as a comma-separated string

    f.write("\n\n")

    # Write the PCA components
    f.write("PCA Components:\n")
    f.write(str(report['PCA Components']))  # Writing PCA object (or use specific components if needed)

    f.write("\n\n")

    # Write whether PCA was used in the models
    f.write("Fit Models with PCA?\n")
    f.write(report['Fit Models with PCA?'])

print("complete")

