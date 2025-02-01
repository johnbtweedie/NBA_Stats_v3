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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score


class AssembleModels:

    def __init__(self, best_models_path):
        self.models = self.load_models(best_models_path) # load the model dictionary
        self.data_dict = self.load_data_dict()
        self.compiled_probs_hold = self.compile_probabilities()
        self.compiled_probs_test = self.compile_probabilities(holdout=False)
        self.train_gradient_boosting()
        print('initialized')

    def load_models(self, best_models_path):
        '''
        load best models dictionary from pickle file
        '''
        print('loading best models file...')
        model_dict = joblib.load(best_models_path)
        print('...complete')
        return model_dict
    
    def load_data_dict(self, data_dict_path='data_2025-01-25_19-11-52.pkl'):
        '''
        load data dictionary from pickle file
        '''
        print('loading data dict file...')
        data_dict = joblib.load(data_dict_path)
        print('...complete')
        return data_dict
    
    def compile_probabilities(self, holdout=True):
        '''
        compile all predicted probabilities on the holdout set to a dataframe
        to be used to train assembly model
        matches opponent probabilities ('_opp')
        '''

        def add_opponent_probs(group):
            group = group.copy()
            
            # Get the opponent's probabilities by shifting within the group
            for col in group.filter(like='_prob').columns:
                group[f'{col}_opp'] = group[col].iloc[::-1].values
            
            return group
        
        if holdout:
            df = pd.DataFrame(index=self.data_dict['X_hold'].index).reset_index()
            
            for model in self.models['models']:
                print(model)
                df[f'{model.split("_")[1]}_prob'] = self.models['models'][model]['hold prob'][:,0]
            
            df_opp = df.groupby('GAME_ID').apply(add_opponent_probs).reset_index(drop=True)
            df_opp = df_opp.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
            df_opp['y'] = self.data_dict['y_hold']

        else:
            df = pd.DataFrame(index=self.data_dict['X_test'].index).reset_index()
            
            for model in self.models['models']:
                print(model)
                df[f'{model.split("_")[1]}_prob'] = self.models['models'][model]['test prob'][:,0]
            
            df_opp = df.groupby('GAME_ID').apply(add_opponent_probs).reset_index(drop=True)
            df_opp = df_opp.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
            df_opp['y'] = self.data_dict['y_test']

        return df_opp
        
    def fit_logistic_regression(self):
        # Extract features (all columns ending in '_prob' or '_prob_opp') and target
        X = self.compiled_probs.filter(like='_prob').values  # Use probability columns
        y = self.compiled_probs['y_hold'].values  # Target variable

        # Optional: Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and fit logistic regression model
        model = LogisticRegression(random_state=100, max_iter=1000)
        model.fit(X_train, y_train)

        # Predict probabilities and classes
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy:.2f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))

        # Store results
        self.models['logistic_regression'] = {
            'model': model,
            'test_accuracy': accuracy,
            'test_predictions': y_pred,
            'test_probabilities': y_prob
        }

        print("Logistic Regression Model Fitted Successfully")

    def train_gradient_boosting(self):
        print("Training Gradient Boosting Classifier to aggregate model predictions...")

        # Define the feature columns (model predictions and their opponent versions)
        self.feature_cols = []
        #     'randomForest_prob', 'logit_prob', 'svm_prob', 
        #     'nnet_prob', 'gradientBoost_prob',
        #     'randomForest_prob_opp', 'logit_prob_opp', 'svm_prob_opp', 
        #     'nnet_prob_opp', 'gradientBoost_prob_opp'
        # ]
        for model in self.models['models']:
            self.feature_cols.append(model.split('class_')[1] + '_prob')
            self.feature_cols.append(model.split('class_')[1] + '_prob_opp')
        # Extract features and target variable from compiled_probs
        X = self.compiled_probs_test[self.feature_cols]
        y = self.compiled_probs_test['y']

        # Split into train/test sets using 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train the Gradient Boosting Classifier
        gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gb_clf.fit(X_train, y_train)

        # Predict probabilities and binary outcomes
        self.gb_clf = gb_clf
        y_pred_prob = gb_clf.predict_proba(X_test)[:, 1]  # Get win probabilities
        y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary labels

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        print(f'Gradient Boosting Model Accuracy: {accuracy:.4f}')
        print(f'Gradient Boosting Model ROC AUC: {roc_auc:.4f}')

        # Store results in self.models
        self.models['assemble_gradientBoost'] = {
            'best model': gb_clf,
            'test accuracy': accuracy,
            'test roc_auc': roc_auc,
            'test prob': y_pred_prob,
            'test pred': y_pred
        }
        test_df = pd.DataFrame(y_pred_prob, index=y_test.index, columns=['final_prob'])
        test_df['y'] = y_test

        self.classification_performance(y_prob=test_df)
        self.classification_performance(y_prob=self.compiled_probs_hold, holdout=True)
        print("...Gradient Boosting training complete.\n")
    
    def classification_performance(self, y_prob, holdout=False):
        '''
        evaluate WL classification by grouping predictions for each game
        the higher probability gets assigned the win
        '''
        df = pd.DataFrame(y_prob).reset_index()

        if holdout:
            # df['final_pred'] = self.gb_clf.predict(df[self.feature_cols])
            df['final_prob'] = self.gb_clf.predict_proba(df[self.feature_cols])[:, 1]

        df['higher_prob'] = df.groupby('GAME_ID')['final_prob'].transform(lambda x: (x == x.max()).astype(int))

        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

        con_matrix = confusion_matrix(df['y'], df['higher_prob'])

        TN, FP, FN, TP = con_matrix.ravel()

        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        f1 = f1_score(df['y'], df['higher_prob'])
        roc = roc_auc_score(df['y'], df['higher_prob'])

        results_dict = {
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "f1-score": f1,
            "roc auc score": roc}

        return results_dict

class AssembleFeatures:
    def __init__(self, conn=sqlite3.connect('nba_database_2025-01-18.db')):
        self.conn = conn
        self.data = self.load_data('current_team_data')
        self.prev_data = self.load_data('prev_team_data')

    def load_data(self, db_table_name):
        '''
        get features and responses from database
        '''
        print(f'loading {db_table_name} from database...')
        df = pd.read_sql(f'SELECT * FROM {db_table_name}', self.conn)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        if db_table_name == 'prev_team_data':
            df = df.set_index(['TEAM_ABBREVIATION', 'oppAbv'])
        else:
            df = df.set_index('TEAM_ABBREVIATION')
        print('...complete\n')
        return df

    def get_team_features(self, home_abv, away_abv):
        '''
        assemble the features required for a prediction for a given team
        '''
        home_features = self.data.loc[[home_abv]].drop(columns=['GAME_DATE', 'GAME_ID', 'oppAbv', 'DaysElapsed'])
        away_features = self.data.loc[[away_abv]].drop(columns=['GAME_DATE', 'GAME_ID', 'oppAbv', 'DaysElapsed'])

        home_prev = self.prev_data.loc[[(home_abv, away_abv)]].drop(columns=['GAME_DATE', 'GAME_ID'])
        home_prev.index = home_prev.index.droplevel(level=1)
        home_prev = home_prev[[col for col in home_features.columns if '_prev' in col]]
        
        away_prev = self.prev_data.loc[[(away_abv, home_abv)]].drop(columns=['GAME_DATE', 'GAME_ID'])
        away_prev.index = away_prev.index.droplevel(level=1)
        away_prev = away_prev[[col for col in home_features.columns if '_prev' in col]]

        home_features.update(home_prev)
        away_features.update(away_prev)

        home_features_renamed = home_features.add_suffix('_opp')
        away_features_renamed = away_features.add_suffix('_opp')

        home_features_renamed.index = ([away_abv])
        away_features_renamed.index = ([home_abv])
        
        home_features['Home'] = 1
        away_features['Home'] = 0
        
        home_features = pd.concat([home_features, away_features_renamed], axis=1)
        away_features = pd.concat([away_features, home_features_renamed], axis=1)

        return home_features, away_features
    
    def get_team_features_old(self, team_abv, opp_abv, is_home):
        '''
        assemble the features required for a prediction for a given team
        '''
        df = self.data.loc[[team_abv]].drop(columns=['GAME_DATE', 'GAME_ID', 'oppAbv', 'DaysElapsed'])
        if is_home:
            df['Home'] = 1
        else:
            df['Home'] = 0

        df_prev = self.prev_data.loc[[(team_abv, opp_abv)]].drop(columns=['GAME_DATE', 'GAME_ID'])
        df_prev.index = df_prev.index.droplevel(level=1)

        return pd.concat([df, df_prev], axis=1)
    
    def assemble_game_features(self, feature_list, games, models):
        game_features = {}
                
        for game_id, teams in games.items():
            home_team = teams['home']
            away_team = teams['away']
            
            home_features, away_features = self.get_team_features(home_team, away_team)

            home_features = home_features[feature_list]
            away_features = away_features[feature_list]

            if models.models['scaler']:
                home_features = models.models['scaler'].transform(home_features)
                away_features = models.models['scaler'].transform(away_features)

            if models.models['pca']:
                home_features = models.models['pca'].transform(home_features)
                away_features = models.models['pca'].transform(away_features)

            # Populate features for both teams
            game_features[game_id] = {
                'home': home_features,
                'away': away_features
            }

        return game_features

def make_predictions(games, features, models):
    '''
    pass games dict, features dict, and assembled models
    returns dict of predicted probabilities for each game
    '''
    model_list = [model for model in  models.models['models']]
    predictions = {}
    for model in model_list:
        model_obj = models.models['models'][model]['best model']
        game_results = {}

        for game in games:
            win_prob_home = model_obj.predict_proba(features[game]['home'])[:, 1][0]
            win_prob_away = model_obj.predict_proba(features[game]['away'])[:, 1][0]
            
            total = win_prob_away + win_prob_home
            win_prob_home = win_prob_home / total
            win_prob_away = win_prob_away / total
            
            game_results[game] = {'home' : win_prob_home,
                            'away' : win_prob_away
                            }
            
        predictions[model] = game_results
    return predictions
        



if __name__ == '__main__':
            # AssembleModels(best_models_path='best_models_2025-01-25_14-44-24.pkl')
            models = AssembleModels(best_models_path='best_models_2025-01-28_23-54-24.pkl')
            
            feature_list = models.models['data']['features']

            # list of games, home vs away
            games = {
                1 : {'home' : 'OKC',
                     'away' : 'TOR'},
                2 : {'home' : 'OKC',
                     'away' : 'WAS'}
            }
            
            features = AssembleFeatures().assemble_game_features(feature_list, games, models)
            predictions = make_predictions(games, features, models)
            

            
            # standardize and pca using the saved scaler and pca model in the pickle
            # run model predictions, return win probability in data dictionary-like object
            #`1: models.models['models']['class_logit']['best model']
            # 2: models.models['assemble_gradientBoost']['best model'].predict_proba(''' classifier probabilites''')
            # pass this to web app to display
            print('ok')

# best_model_folder = r'catalogs/best_models_2024-09-29/'
# logreg_model = joblib.load(best_model_folder + 'best_logreg_model_pca.pkl')
# rf_model = joblib.load(best_model_folder + 'best_rf_model_pca.pkl')
# nnet_model = joblib.load(best_model_folder + 'best_nn_mode_pca.pkl')
# svm_model = joblib.load(best_model_folder + 'best_svm_model_pca.pkl')

# # Initialize variables
# report_file = r'catalogs/best_models_2024-09-18/model_log.txt'

# # Create variables to hold the loaded data
# feature_cols = []
# pca = None
# df_results = None

# # Read and re-assign values from the text file
# with open(report_file, 'r') as f:
#     lines = f.readlines()

#     # Helper function to find the next blank line
#     def find_next_blank_line(start_idx):
#         for idx in range(start_idx, len(lines)):
#             if lines[idx].strip() == "":
#                 return idx
#         return len(lines)  # In case there's no blank line

#     # Extract feature set
#     feature_set_start = lines.index("Feature Set:\n") + 1
#     feature_cols = lines[feature_set_start:find_next_blank_line(feature_set_start) ][0].strip().split(", ")

#     # Extract if models used PCA
#     pca_used = lines[lines.index("Fit Models with PCA?\n") + 1].strip()
#     print(f"PCA used: {pca_used}")

#     if pca_used == 'Yes':
#         # Extract PCA components
#         pca_start = (lines.index("PCA Components:\n") + 1)
#         n_components = int(''.join(lines[pca_start:(find_next_blank_line(pca_start))]).strip().split('=')[1].split(')')[0])

# def load_feature_data():
#     # import/process
#     conn = sqlite3.connect('nba_database_2024-08-18.db')
#     # df_feat = pd.read_csv('features.csv')
#     df_feat = pd.read_sql('SELECT * FROM feature_table', conn)
#     df_feat['GAME_DATE'] = pd.to_datetime(df_feat['GAME_DATE'])
#     df_feat = df_feat.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
#     # df_feat = df_feat.sort_index(level=['TEAM_ABBREVIATION', 'GAME_DATE'])
#     return df_feat

# def model_classification_performance(y_test, y_pred, model_name, model_call):
#     con_matrix = confusion_matrix(y_test, y_pred.round())
#     TN, FP, FN, TP = con_matrix.ravel()

#     sensitivity = TP / (TP + FN)
#     specificity = TN / (TN + FP)
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     f1 = f1_score(y_test, y_pred.round())
#     roc = roc_auc_score(y_test, y_pred.round())

#     results_dict = {
#         "Model": model_name,
#         "Call": str(model_call),
#         "TP": TP,
#         "TN": TN,
#         "FP": FP,
#         "FN": FN,
#         "Accuracy": accuracy,
#         "Sensitivity": sensitivity,
#         "Specificity": specificity,
#         "f1-score": f1,
#         "roc auc score": roc}

#     return pd.DataFrame(results_dict, index=[model_name])

# def reprocess_holdout_data(df_feat, pca_used):
#     print('segmenting test/train set')
#     X = df_feat[feature_cols]
#     y = df_feat['WL']

#     # get hold out set
#     n_games_for_holdout = 8
#     X_holdout = X.iloc[-8:]
#     X = X.iloc[:-8]

#     y_holdout = y.iloc[-8:]
#     y = y.iloc[:-8]

#     # make sure this is the same as the train models split parameters
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
#     scaler = StandardScaler()
#     X_train = pd.DataFrame(scaler.fit_transform(X_train), index=y_train.index)
#     X_holdout = pd.DataFrame(scaler.transform(X_holdout), index=y_holdout.index)

#     if pca_used == 'Yes':
#         pca = PCA(n_components=n_components) # need to save transformer from model fitting
#     else:
#         cov_matrix = np.cov(X_train.T)
#         eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
#         sorted_eigenvalues = np.sort(eigenvalues)[::-1]
#         num_components_kaiser = sum(sorted_eigenvalues > 1 + 1)
#         pca = PCA(n_components=num_components_kaiser)

#     X_train_pca = pd.DataFrame(pca.fit_transform(X_train), index=y_train.index)
#     X_holdout_pca = pd.DataFrame(pca.transform(X_holdout), index=y_holdout.index)

#     return {
#         'X_holdout_pca': X_holdout_pca,
#         'X_holdout': X_holdout,
#         'y_holdout': y_holdout,
#         'X_train' : X_train,
#         'X_train_pca' : X_train_pca}

# def load_or_process_data(file_path='holdout_data.pkl', pca_used=pca_used,):
#     if os.path.exists(file_path):
#         try:
#             # Load data from the pickle file if it exists
#             with open(file_path, 'rb') as f:
#                 # get list of potential response variables
#                 df_feat = load_feature_data()
#                 df_feat = df_feat.dropna()
#                 response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']
#                 data = pickle.load(f)
#         except Exception as e:
#             print(f"Error loading file: {e}")
#             print(f"Reloading features...")
#             df_feat = load_feature_data()
#             df_feat = df_feat.dropna()
#             # get list of potential response variables
#             response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']
#             # If there's an error in loading, process the data instead
#             data = reprocess_holdout_data(df_feat, pca_used)
#             # Save the processed data
#             # with open(file_path, 'wb') as f:
#             #     pickle.dump(data, f)
#     else:
#         # If the file doesn't exist, process and save the data
#         print(f"Reloading features...")
#         df_feat = load_feature_data()
#         df_feat = df_feat.dropna()
#         # get list of potential response variables
#         response_cols = [col for col in df_feat.columns if '_r' in col] + ['WL']
#         data = reprocess_holdout_data(df_feat, pca_used)
#         # with open(file_path, 'wb') as f:
#         #     pickle.dump(data, f)

#     return response_cols, data

# def load_required_data_for_any_matchup_prediction():
#     '''
#     funciton loads and processes the necessary data in order to make any matchup prediction
#     '''
#     conn = sqlite3.connect('nba_database_2024-08-18.db')

#     df_feat_existing = pd.read_sql('SELECT * FROM feature_table', conn)
#     df_feat_existing = df_feat_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

#     unique_teams = df_feat_existing.index.get_level_values('TEAM_ABBREVIATION').unique()
#     # try:
#     #     unique_teams = unique_teams.drop('NOH')

#     # get the past 5 games rolling average stats vs the opponent
#     print('determining previous matchup stats...')
#     required_retrieval_index = []
#     for i, team_1 in enumerate(unique_teams):
#         print('Processing', i + 1, 'of', len(unique_teams), 'teams')

#         for team_2 in unique_teams[i + 1:]:
#             # #
#             # team_1 = 'ATL'
#             # team_2 = 'NOP'
#             print(team_1, 'vs.', team_2)
#             df_team_1 = df_feat_existing.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)
#             df_team_2 = df_feat_existing.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)

#             shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

#             # store 5th most recent games index value
#             required_retrieval_index.append(shared_game_ids[-1])

#     max_required_retrieval_index = pd.to_numeric(
#         required_retrieval_index).min()  # index of the least recent game we have to grab data from
#     df = pd.read_sql(
#         f"SELECT * FROM feature_table WHERE CAST(GAME_ID AS INTEGER) >= {max_required_retrieval_index}",
#         conn)
#     df['opponent'] = df.groupby('GAME_ID')['TEAM_ABBREVIATION'].transform(lambda x: x[::-1].values)
#     prev_columns = [col for col in df.columns if '_prev' in col]
#     previous_matchup_dict = {}
#     # get a dictionary of dictionaries containing each teams current stats, and their 'prev' stats against all possible opponents
#     for team in unique_teams:
#         df_team = df.loc[df['TEAM_ABBREVIATION'] == team]
#         team_stats_for_each_opponent_dict = {}
#         for opponent in df_team['opponent'].unique():
#             df_opponent = df_team.loc[df_team['opponent'] == opponent, prev_columns].iloc[-1]  # iloc[-1] in case of duplicates
#             team_stats_for_each_opponent_dict[opponent] = df_team[prev_columns].iloc[-1]
#         team_stats_for_each_opponent_dict[team] = df_opponent
#         previous_matchup_dict[team] = team_stats_for_each_opponent_dict

#     return df, previous_matchup_dict

# df, previous_matchup_dict = load_required_data_for_any_matchup_prediction()


# # Example usage
# response_cols, data = load_or_process_data()
# X_holdout_pca = data['X_holdout_pca']
# X_holdout = data['X_holdout']
# y_holdout = data['y_holdout']

# # Apply models to the holdout data based on whether PCA was used
# if pca_used == 'Yes':
#     X_to_use = X_holdout_pca  # Use PCA-transformed data
# else:
#     X_to_use = X_holdout      # Use non-PCA data

# # Make predictions using each model and get probabilities
# logreg_predictions = logreg_model.predict(X_to_use)
# logreg_probs = logreg_model.predict_proba(X_to_use)[:, 1]  # Probability of class 1

# rf_predictions = rf_model.predict(X_to_use)
# rf_probs = rf_model.predict_proba(X_to_use)[:, 1]          # Probability of class 1

# nnet_predictions = nnet_model.predict(X_to_use)
# nnet_probs = nnet_model.predict_proba(X_to_use)[:, 1]      # Probability of class 1

# svm_predictions = svm_model.predict(X_to_use)
# # svm_probs = svm_model.predict_proba(X_to_use)[:, 1]        # Probability of class 1

# # Create a DataFrame to compare results
# df_results = pd.DataFrame({
#     'y_holdout': y_holdout,                 # True values
#     'Logistic Regression Prediction': logreg_predictions,
#     'Logistic Regression Probability': logreg_probs,
#     'Random Forest Prediction': rf_predictions,
#     'Random Forest Probability': rf_probs,
#     'Neural Network Prediction': nnet_predictions,
#     'Neural Network Probability': nnet_probs,
#     'SVM Prediction': svm_predictions,
#     # 'SVM Probability': svm_probs
# })

# probability_columns = [col for col in df_results.columns if 'Probability' in col]
# df_results['Sum of Probabilities'] = df_results[probability_columns].sum(axis=1)

# import joblib
# from sklearn.metrics import accuracy_score

# # Example Data: Assuming df_results is already created with probability sums

# # Group the DataFrame by the second level of the index
# grouped = df_results.groupby(df_results.index.get_level_values(1))

# # Function to assign 1 or 0 based on probability comparison
# def assign_prob_vote(group):
#     # Get the index of the rows in the group
#     idx = group.index
#     # Compare the sum of probabilities
#     if group.loc[idx[0], 'Sum of Probabilities'] > group.loc[idx[1], 'Sum of Probabilities']:
#         group['prob_vote'] = 0  # Initialize with 0
#         group.loc[idx[0], 'prob_vote'] = 1  # Assign 1 to the higher value
#     else:
#         group['prob_vote'] = 0  # Initialize with 0
#         group.loc[idx[1], 'prob_vote'] = 1  # Assign 1 to the higher value

#     # calculated weighted win probability
#     total_probability = group.loc[idx[0], 'Sum of Probabilities'] + group.loc[idx[1], 'Sum of Probabilities']
#     group.loc[idx[0], 'Win Probability'] = group.loc[idx[0], 'Sum of Probabilities'] / total_probability
#     group.loc[idx[1], 'Win Probability'] = group.loc[idx[1], 'Sum of Probabilities'] / total_probability

#     return group

# # Apply the function to each group and assign the result back to df_results
# df_results = grouped.apply(assign_prob_vote)
# df_results = df_results.droplevel(level=0)
# # Print the updated DataFrame with the new 'prob_vote' column
# # print(df_results)



# # create dataframe to pass to the dashboard
# df_out = df_results[['Win Probability', 'prob_vote', ]]
# df_out['Home'] = X_to_use.iloc[:,feature_cols.index('Home')].replace({-1: 'Away', 1: 'Home'})
# df_out['Team'] = X_to_use.index.get_level_values(2)

# df_out.to_pickle('games_data.pkl')


# print('complete')