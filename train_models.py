import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import joblib
import sqlite3

def load_feature_data():
    # import/process
    conn = sqlite3.connect('nba_database_2024-08-18.db')
    # df_feat = pd.read_csv('features.csv')
    df_feat = pd.read_sql('SELECT * FROM feature_table', conn)
    df_feat['GAME_DATE'] = pd.to_datetime(df_feat['GAME_DATE'])
    df_feat = df_feat.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
    # df_feat = df_feat.sort_index(level=['TEAM_ABBREVIATION', 'GAME_DATE'])
    return df_feat

df_feat = load_feature_data()

# fix
cols = ['DaysElapsed', 'DaysRest', 'DaysRest_opp',
        'Matchup', 'WL', 'Home', 'roadtrip', 'roadtrip_opp',
        'MATCHUP', 'Poss_r', 'OffRat_r', 'DefRat_r', 'PTS_per48_r', 'PTS_per48_against_r', 'PTS_per48_diff_r']
prev_matchup_cols = ['OREB_per48', 'OREB%_z', 'DREB_per48', 'DREB%_z', 'REB_per48',
                     'FGA_per48', 'FG3A_per48', 'FTA_per48',
                     'WL', 'OffRat', 'DefRat', 'OffRat_z', 'DefRat_z',
                     'PTS_per48_z', 'PTS_per48_against_z',
                     'FGM_per48_z', 'FGA_per48_z', 'FG3M_per48_z', 'FG3A_per48_z',
                     'FTM_per48_z', 'FTA_per48_z', 'FGM_per48_against_z', 'FGA_per48_against_z',
                     'FG3M_per48_against_z', 'FG3A_per48_against_z', 'FTM_per48_against_z', 'FTA_per48_against']
prev_matchup_cols = [col + '_prev' for col in prev_matchup_cols]
cols = cols + prev_matchup_cols
#cols + prev_matchup_cols
cols_shift = [col for col in df_feat.columns if col not in cols]
df_feat[cols_shift] = df_feat.groupby('TEAM_ABBREVIATION',
           group_keys=False)[cols_shift].shift(1)
df_feat = df_feat.dropna()

# select features to use in model
feature_cols = [
'Win',
'Win_opp',
'AST%_z',
'DaysRest',
'DaysRest_opp',
'DefRat_prev',
'DefRat_z',
'DefRat_z_prev',
'DREB_per48_prev',
'DREB_per48_z',
'DREB%_z',
'DREB%_z_prev',
'eFG%_against_z',
'eFG%_z',
'FG3A_per48_against_z',
'FG3A_per48_against_z_prev',
'FG3A_per48_prev',
'FG3A_per48_z',
'FG3A_per48_z_prev',
'FG3M_per48_against_z',
'FG3M_per48_against_z_prev',
'FG3M_per48_z',
'FG3M_per48_z_prev',
'FGA_per48_against_z',
'FGA_per48_against_z_prev',
'FGA_per48_prev',
'FGA_per48_z',
'FGA_per48_z_prev',
'FGM_per48_against_z',
'FGM_per48_against_z_prev',
'FGM_per48_z',
'FGM_per48_z_prev',
'FTA_per48_against_prev',
'FTA_per48_against_z',
'FTA_per48_z',
'FTM_per48_against_z',
'FTM_per48_against_z_prev',
'FTM_per48_z',
'FTM_per48_z_prev',
'Home',
'OffRat_prev',
'OffRat_z',
'OffRat_z_prev',
'OREB_per48_prev',
'OREB_per48_z',
'OREB%_z',
'OREB%_z_prev',
'Poss',
'Poss_opp',
'PTS_per48_against_z',
'PTS_per48_against_z_prev',
'PTS_per48_z',
'PTS_per48_z_prev',
'roadtrip',
'roadtrip_opp',
'STL%_z',
'TOV_forced%_z',
'TOV%_z',
'TS%_against_z',
'TS%_z'
]

print('segmenting test/train set')
X = df_feat[feature_cols]
y = df_feat['WL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print('fitting model')
nn_model = MLPClassifier(solver='sgd', alpha=1,
                    hidden_layer_sizes=(6, 2, 2), random_state=100)

nn_model.fit(X_train, y_train)

y_pred = nn_model.predict(X_test)
con_matrix = confusion_matrix(y_test, y_pred.round())
TN, FP, FN, TP = con_matrix.ravel()

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)
accuracy = (TP + TN) / (TP + TN + FP + FN)
f1 = f1_score(y_test, y_pred.round())
roc = roc_auc_score(y_test, y_pred.round())

print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("f1-score:", f1)
print("roc auc score:", roc)

joblib.dump(nn_model, 'nn_model.pkl')