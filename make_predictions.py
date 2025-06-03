import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import pickle
import itertools
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
import os
print("Running script from:", __file__)
print("Current working directory:", os.getcwd())


class AssembleModels:

    def __init__(self, best_models_path):
        self.models = self.load_models(best_models_path) # load the model dictionary
        self.data_dict = self.models['data']#self.load_data_dict()
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
    
    # def load_ensemble(self, best_models_path)
    #     '''
    #     load ensemble model from pickle file
    #     '''
    #     print('loading ensemble mode file...')
    #     model_dict = joblib.load(best_models_path)
    #     print('...complete')
    #     return model_dict
    
    # def load_data_dict(self, data_dict_path='data_2025-01-25_19-11-52.pkl'):
    #     '''
    #     load data dictionary from pickle file
    #     '''
    #     print('loading data dict file...')
    #     data_dict = joblib.load(data_dict_path)
    #     print('...complete')
    #     return data_dict
    
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
    def __init__(self, conn=sqlite3.connect('nba_database_2025-01-18.db'), for_custom_matchups=False):
        self.conn = conn
        self.data = self.load_data('current_team_data')
        self.prev_data = self.load_data('prev_team_data')
        self.for_custom_matchups = for_custom_matchups

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
        
        if self.for_custom_matchups:
            home_features['DaysRest'] = 1
            away_features['DaysRest'] = 1

            home_features['roadtrip'] = 0
            away_features['roadtrip'] = 1

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

    # predictions using individual models
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

    # ensemble model predictions
    model_obj = models.models['ensemble']['model']
    ensemble_features_names = model_obj.feature_names_in_
    game_results = {}
    for game in games:
        home_features = pd.DataFrame()
        away_features = pd.DataFrame()
        for i, ensemble_feature in enumerate(ensemble_features_names):
            home_features[ensemble_feature] = [predictions[model_obj.feature_names_in_[i]][game]['home']]
            away_features[ensemble_feature] = [predictions[model_obj.feature_names_in_[i]][game]['away']]
        win_prob_home = model_obj.predict_proba(home_features)[:, 0][0]
        win_prob_away = model_obj.predict_proba(away_features)[:, 0][0]
        
        total = win_prob_away + win_prob_home
        win_prob_home = win_prob_home / total
        win_prob_away = win_prob_away / total
        
        game_results[game] = {'home' : win_prob_home,
                        'away' : win_prob_away
                        }
        
    predictions['ensemble'] = game_results


    return predictions


if __name__ == '__main__':

            print('generating prediction data for all possible team matchups...')

            # create dictionary of all possible matchups
            team_abbreviations = [
                "ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN",
                "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM", "MIA",
                "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX",
                "POR", "SAC", "SAS", "TOR", "UTA", "WAS"
            ]

            matchups = list(itertools.permutations(team_abbreviations, 2))

            # Build the dictionary
            games = {
                i + 1: {'home': home, 'away': away}
                for i, (home, away) in enumerate(matchups)
            }

            # AssembleModels(best_models_path='best_models_2025-01-25_14-44-24.pkl')
            models = AssembleModels(best_models_path='/Users/johntweedie/Dev/Projects/PN24001_NBA_Stats/catalogs/best_models_2025-05-25_03-21-15.pkl')
            
            feature_list = models.models['data']['features']
            features = AssembleFeatures(for_custom_matchups=True).assemble_game_features(feature_list, games, models)
            predictions = make_predictions(games, features, models)
            results = pd.DataFrame(games).T
            for model in predictions.items():
                results = pd.concat([pd.DataFrame(predictions[model[0]]).T.add_suffix(f'_{model[0]}'), results], axis=1)

            # save and pass this to web app to display
            model_filename = os.path.join('data', 'WL_predictions.pkl')
            os.makedirs('data', exist_ok=True)
            joblib.dump(results, model_filename)

            print('complete')