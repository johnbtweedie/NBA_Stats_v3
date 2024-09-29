import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



best_model_folder = r'catalogs/best_models_2024-09-29/'
logreg_model = joblib.load(best_model_folder + 'best_logreg_model_pca.pkl')
rf_model = joblib.load(best_model_folder + 'best_rf_model_pca.pkl')
nnet_model = joblib.load(best_model_folder + 'best_nn_mode_pca.pkl')
svm_model = joblib.load(best_model_folder + 'best_svm_model_pca.pkl')

# Initialize variables
report_file = r'catalogs/best_models_2024-09-18/model_log.txt'

# Create variables to hold the loaded data
feature_cols = []
pca = None
df_results = None

# Read and re-assign values from the text file
with open(report_file, 'r') as f:
    lines = f.readlines()

    # Helper function to find the next blank line
    def find_next_blank_line(start_idx):
        for idx in range(start_idx, len(lines)):
            if lines[idx].strip() == "":
                return idx
        return len(lines)  # In case there's no blank line

    # Extract feature set
    feature_set_start = lines.index("Feature Set:\n") + 1
    feature_cols = lines[feature_set_start:find_next_blank_line(feature_set_start) ][0].strip().split(", ")

    # Extract if models used PCA
    pca_used = lines[lines.index("Fit Models with PCA?\n") + 1].strip()
    print(f"PCA used: {pca_used}")

    if pca_used == 'Yes':
        # Extract PCA components
        pca_start = (lines.index("PCA Components:\n") + 1)
        n_components = int(''.join(lines[pca_start:(find_next_blank_line(pca_start))]).strip().split('=')[1].split(')')[0])

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

def reprocess_holdout_data(df_feat, pca_used):
    print('segmenting test/train set')
    X = df_feat[feature_cols]
    y = df_feat['WL']

    # get hold out set
    n_games_for_holdout = 8
    X_holdout = X.iloc[-8:]
    X = X.iloc[:-8]

    y_holdout = y.iloc[-8:]
    y = y.iloc[:-8]

    # make sure this is the same as the train models split parameters
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), index=y_train.index)
    X_holdout = pd.DataFrame(scaler.transform(X_holdout), index=y_holdout.index)

    if pca_used == 'Yes':
        pca = PCA(n_components=n_components)
    else:
        cov_matrix = np.cov(X_train.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        num_components_kaiser = sum(sorted_eigenvalues > 1 + 1)
        pca = PCA(n_components=num_components_kaiser)

    X_train_pca = pd.DataFrame(pca.fit_transform(X_train), index=y_train.index)
    X_holdout_pca = pd.DataFrame(pca.transform(X_holdout), index=y_holdout.index)

    return {
        'X_holdout_pca': X_holdout_pca,
        'X_holdout': X_holdout,
        'y_holdout': y_holdout,
        'X_train' : X_train,
        'X_train_pca' : X_train_pca}

def load_or_process_data(file_path='holdout_data.pkl', pca_used=pca_used,):
    if os.path.exists(file_path):
        try:
            # Load data from the pickle file if it exists
            with open(file_path, 'rb') as f:
                # get list of potential response variables
                df_feat = load_feature_data()
                df_feat = df_feat.dropna()
                response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']
                data = pickle.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            print(f"Reloading features...")
            df_feat = load_feature_data()
            df_feat = df_feat.dropna()
            # get list of potential response variables
            response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']
            # If there's an error in loading, process the data instead
            data = reprocess_holdout_data(df_feat, pca_used)
            # Save the processed data
            # with open(file_path, 'wb') as f:
            #     pickle.dump(data, f)
    else:
        # If the file doesn't exist, process and save the data
        print(f"Reloading features...")
        df_feat = load_feature_data()
        df_feat = df_feat.dropna()
        # get list of potential response variables
        response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']
        data = reprocess_holdout_data(df_feat, pca_used)
        # with open(file_path, 'wb') as f:
        #     pickle.dump(data, f)

    return response_cols, data


# Example usage
response_cols, data = load_or_process_data()
X_holdout_pca = data['X_holdout_pca']
X_holdout = data['X_holdout']
y_holdout = data['y_holdout']

# Apply models to the holdout data based on whether PCA was used
if pca_used == 'Yes':
    X_to_use = X_holdout_pca  # Use PCA-transformed data
else:
    X_to_use = X_holdout      # Use non-PCA data

# Make predictions using each model and get probabilities
logreg_predictions = logreg_model.predict(X_to_use)
logreg_probs = logreg_model.predict_proba(X_to_use)[:, 1]  # Probability of class 1

rf_predictions = rf_model.predict(X_to_use)
rf_probs = rf_model.predict_proba(X_to_use)[:, 1]          # Probability of class 1

nnet_predictions = nnet_model.predict(X_to_use)
nnet_probs = nnet_model.predict_proba(X_to_use)[:, 1]      # Probability of class 1

svm_predictions = svm_model.predict(X_to_use)
# svm_probs = svm_model.predict_proba(X_to_use)[:, 1]        # Probability of class 1

# Create a DataFrame to compare results
df_results = pd.DataFrame({
    'y_holdout': y_holdout,                 # True values
    'Logistic Regression Prediction': logreg_predictions,
    'Logistic Regression Probability': logreg_probs,
    'Random Forest Prediction': rf_predictions,
    'Random Forest Probability': rf_probs,
    'Neural Network Prediction': nnet_predictions,
    'Neural Network Probability': nnet_probs,
    'SVM Prediction': svm_predictions,
    # 'SVM Probability': svm_probs
})

probability_columns = [col for col in df_results.columns if 'Probability' in col]
df_results['Sum of Probabilities'] = df_results[probability_columns].sum(axis=1)

import joblib
from sklearn.metrics import accuracy_score

# Example Data: Assuming df_results is already created with probability sums

# Group the DataFrame by the second level of the index
grouped = df_results.groupby(df_results.index.get_level_values(1))

# Function to assign 1 or 0 based on probability comparison
def assign_prob_vote(group):
    # Get the index of the rows in the group
    idx = group.index
    # Compare the sum of probabilities
    if group.loc[idx[0], 'Sum of Probabilities'] > group.loc[idx[1], 'Sum of Probabilities']:
        group['prob_vote'] = 0  # Initialize with 0
        group.loc[idx[0], 'prob_vote'] = 1  # Assign 1 to the higher value
    else:
        group['prob_vote'] = 0  # Initialize with 0
        group.loc[idx[1], 'prob_vote'] = 1  # Assign 1 to the higher value

    # calculated weighted win probability
    total_probability = group.loc[idx[0], 'Sum of Probabilities'] + group.loc[idx[1], 'Sum of Probabilities']
    group.loc[idx[0], 'Win Probability'] = group.loc[idx[0], 'Sum of Probabilities'] / total_probability
    group.loc[idx[1], 'Win Probability'] = group.loc[idx[1], 'Sum of Probabilities'] / total_probability

    return group

# Apply the function to each group and assign the result back to df_results
df_results = grouped.apply(assign_prob_vote)
df_results = df_results.droplevel(level=0)
# Print the updated DataFrame with the new 'prob_vote' column
# print(df_results)



# create dataframe to pass to the dashboard
df_out = df_results[['Win Probability', 'prob_vote', ]]
df_out['Home'] = X_to_use.iloc[:,feature_cols.index('Home')].replace({-1: 'Away', 1: 'Home'})
df_out['Team'] = X_to_use.index.get_level_values(2)

df_out.to_pickle('games_data.pkl')


print('complete')