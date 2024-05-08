import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
import joblib


# import/process
df_feat = pd.read_csv('features.csv')
df_feat['GAME_DATE'] = pd.to_datetime(df_feat['GAME_DATE'])
df_feat = df_feat.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
df_feat = df_feat.sort_index(level=['TEAM_ABBREVIATION', 'GAME_DATE'])

cols = ['DaysElapsed', 'DaysRest', 'DaysRest_opp',
        'Matchup', 'WL', 'Home', 'roadtrip', 'roadtrip_opp']
cols_shift = [col for col in df_feat.columns if col not in cols]
df_feat[cols_shift] = df_feat.groupby('TEAM_ABBREVIATION',
           group_keys=False)[cols_shift].shift(1)
df_feat = df_feat.dropna()

# select features to use in model
feature_cols = [
    'AST_per48',
    'AST_per48_against',
    'AST_per48_against_opp',
    'AST_per48_opp',
    'BLKA_per48',
    'BLKA_per48_against',
    'BLKA_per48_against_opp',
    'BLKA_per48_opp',
    'BLK_per48',
    'BLK_per48_against',
    'BLK_per48_against_opp',
    'BLK_per48_opp',
    'DREB_per48',
    'DREB_per48_against',
    'DREB_per48_against_opp',
    'DREB_per48_opp',
    'DREB_per48_z',
    'DREB_per48_z_opp',
    'FG3A_per48',
    'FG3A_per48_against',
    'FG3A_per48_against_opp',
    'FG3A_per48_against_z',
    'FG3A_per48_against_z_opp',
    'FG3A_per48_opp',
    'FG3A_per48_z',
    'FG3A_per48_z_opp',
    'FG3M_per48',
    'FG3M_per48_against',
    'FG3M_per48_against_opp',
    'FG3M_per48_against_z',
    'FG3M_per48_against_z_opp',
    'FG3M_per48_opp',
    'FG3M_per48_z',
    'FG3M_per48_z_opp',
    'FG3_PCT',
    'FG3_PCT_against',
    'FG3_PCT_against_opp',
    'FG3_PCT_opp',
    'FGA_per48',
    'FGA_per48_against',
    'FGA_per48_against_opp',
    'FGA_per48_against_z',
    'FGA_per48_against_z_opp',
    'FGA_per48_opp',
    'FGA_per48_z',
    'FGA_per48_z_opp',
    'FGM',
    'FGM_per48',
    'FGM_per48_against',
    'FGM_per48_against_opp',
    'FGM_per48_against_z',
    'FGM_per48_against_z_opp',
    'FGM_per48_opp',
    'FGM_per48_z',
    'FGM_per48_z_opp',
    'FG_PCT',
    'FG_PCT_against',
    'FG_PCT_against_opp',
    'FG_PCT_opp',
    'FTA_per48',
    'FTA_per48_against',
    'FTA_per48_against_opp',
    'FTA_per48_against_z',
    'FTA_per48_against_z_opp',
    'FTA_per48_opp',
    'FTA_per48_z',
    'FTA_per48_z_opp',
    'FTM_per48',
    'FTM_per48_against',
    'FTM_per48_against_opp',
    'FTM_per48_against_z',
    'FTM_per48_against_z_opp',
    'FTM_per48_opp',
    'FTM_per48_z',
    'FTM_per48_z_opp',
    'FT_PCT',
    'FT_PCT_opp',
    'OREB_per48',
    'OREB_per48_against',
    'OREB_per48_against_opp',
    'OREB_per48_opp',
    'OREB_per48_z',
    'OREB_per48_z_opp',
    'PFD_per48',
    'PFD_per48_against',
    'PFD_per48_against_opp',
    'PFD_per48_opp',
    'PF_per48',
    'PF_per48_against',
    'PF_per48_against_opp',
    'PF_per48_opp',
    'PLUS_MINUS_per48',
    'PLUS_MINUS_per48_opp',
    'PTS_per48',
    'PTS_per48_against',
    'PTS_per48_against_opp',
    'PTS_per48_against_z',
    'PTS_per48_against_z_opp',
    'PTS_per48_opp',
    'PTS_per48_z',
    'PTS_per48_z_opp',
    'REB_per48',
    'REB_per48_against',
    'REB_per48_against_opp',
    'REB_per48_opp',
    'STL_per48',
    'STL_per48_against',
    'STL_per48_against_opp',
    'STL_per48_opp',
    'TOV_per48',
    'TOV_per48_against',
    'TOV_per48_against_opp',
    'TOV_per48_opp',
    'Win',
    'Win_opp',
    'DaysRest',
    'DaysRest_opp',
    'DefRat_z',
    'DefRat_z_opp',
    'DREB%_opp',
    'DREB%',
    'OffRat_z',
    'OffRat_z_opp',
    'OREB%_opp',
    'OREB%',
    'TOV%_z_opp',
    'TOV%_z',
    'eFG%',
    'eFG%_opp',
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