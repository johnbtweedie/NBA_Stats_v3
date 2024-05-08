import pandas as pd
import numpy as np

# LOAD
df = pd.read_csv('historical_stats.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
df = df.sort_index(level=['TEAM_ABBREVIATION', 'GAME_DATE'])

cols = [ 'Win',
         'FGM',
         'FG_PCT',
         'FG3_PCT',
         'FT_PCT',
         'FGM_per48',
         'FGA_per48',
         'FG3M_per48',
         'FG3A_per48',
         'FTM_per48',
         'FTA_per48',
         'OREB_per48',
         'DREB_per48',
         'REB_per48',
         'AST_per48',
         'TOV_per48',
         'STL_per48',
         'BLK_per48',
         'BLKA_per48',
         'PF_per48',
         'PFD_per48',
         'PTS_per48',
         'PLUS_MINUS_per48',
         'Win_against',
         'FG_PCT_against',
         'FG3_PCT_against',
         'FGM_per48_against',
         'FGA_per48_against',
         'FG3M_per48_against',
         'FG3A_per48_against',
         'FTM_per48_against',
         'FTA_per48_against',
         'OREB_per48_against',
         'DREB_per48_against',
         'REB_per48_against',
         'AST_per48_against',
         'TOV_per48_against',
         'STL_per48_against',
         'BLK_per48_against',
         'BLKA_per48_against',
         'PF_per48_against',
         'PFD_per48_against',
         'PTS_per48_against',
       ]

# rolling average for training data (shifted by 1 game)
df = df.sort_index(level=['GAME_DATE'], ascending=True)
df_feat = df.groupby('TEAM_ABBREVIATION',
           group_keys=False)[cols].apply(lambda x: x.rolling(window=41,
                                                             min_periods=1).mean().shift(1))
# feature engineering

df_feat['OER'] = df_feat['PTS_per48'] / (df_feat['FGA_per48'] +
                                         ((df_feat['FTA_per48']*0.9)/2) -
                                         df_feat['TOV_per48'])
df_feat['DER'] = df_feat['PTS_per48_against'] / (df_feat['FGA_per48_against'] +
                                                 ((df_feat['FTA_per48_against']*0.9)/2) -
                                                 df_feat['TOV_per48_against'])

df_feat['eFG%'] = ((df_feat['FGM_per48'] + (0.5 * df_feat['FG3M_per48'])) / df_feat['FGA_per48']) * 100
df_feat['eFG%_against'] = ((df_feat['FGM_per48_against'] +
                            (0.5 * df_feat['FG3M_per48_against'])) /
                           df_feat['FGA_per48_against']) * 100

df_feat['TS%'] = (df_feat['PTS_per48'] / (2 * (df_feat['FGA_per48'] + (0.44 * df_feat['FTA_per48'])))) * 100
df_feat['TS%_against'] = (df_feat['PTS_per48_against'] /
                          (2 * (df_feat['FGA_per48_against'] +
                                (0.44 * df_feat['FTA_per48_against'])))) * 100

min_date = df_feat.index.get_level_values('GAME_DATE').min()
df_feat['DaysElapsed'] = (df_feat.index.get_level_values('GAME_DATE') - min_date).days
df_feat['DaysRest'] = df_feat.groupby('TEAM_ABBREVIATION',
           group_keys=False)['DaysElapsed'].diff()
df_feat['DaysRest'] = df_feat['DaysRest'].apply(lambda x: 5 if x>5 else x)

# misc
df_feat[['MATCHUP','WL']] = df[['MATCHUP','WL']]
df_feat['Home'] = df_feat['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
df_feat['WL'] = df_feat['WL'].map({'W': 1, 'L': 0})
df['WL'] = df['WL'].map({'W': 1, 'L': 0})

def consecutive_zeros(lst):
    count = 0
    result = []
    for num in lst:
        if num == 0:
            count += 1
        else:
            count = 0
        result.append(count)
    return result

df_roadtrip = df.groupby('TEAM_ABBREVIATION', group_keys=False)['Home'].apply(lambda x: consecutive_zeros(x))

df_results = pd.DataFrame()
for i, group in enumerate(df_roadtrip):
    df_group = pd.DataFrame(df_roadtrip[i],
                            index=df.loc[:,:,df_roadtrip.index[i]].index,
                            columns=['roadtrip'])
    df_group['TEAM_ABBREVIATION'] = df_roadtrip.index[i]
    df_group = df_group.set_index('TEAM_ABBREVIATION', append=True)
    df_results = pd.concat([df_group, df_results])

df_feat['roadtrip'] = df_results['roadtrip']

# this one appears more in line with other sources
df_feat['Poss'] = (df_feat['FGA_per48'] - df_feat['OREB_per48'] + df_feat['TOV_per48'] +
                   0.4 * df_feat['FTA_per48'])

df_feat['OffRat'] = df_feat['PTS_per48'] / df_feat['Poss'] * 100
df_feat['DefRat'] = df_feat['PTS_per48_against'] / df_feat['Poss'] * 100

# Rebound percentages (rebounds per shot)
# choosing not to account for rebounds on the last missed foul shot, for now
df_feat['OREB%'] = df_feat['OREB_per48'] / (df_feat['FGA_per48'] - df_feat['FGM_per48'])
df_feat['DREB%'] = df_feat['DREB_per48'] / (df_feat['FGA_per48_against'] - df_feat['FGM_per48_against'])

# TOV percentage
df_feat['TOV%'] = df_feat['TOV_per48'] / df_feat['Poss']
df_feat['TOV_forced%'] = df_feat['TOV_per48_against'] / df_feat['Poss']
df_feat['STL%'] = df_feat['TOV_per48'] / df_feat['Poss']

# Assist rate
df_feat['AST%'] = df_feat['AST_per48'] / df_feat['FGM_per48']


# WIP: BB%, ball-back %, pct chance team will get the ball back from the opponent on a possession
df_feat['BB%'] = (df_feat['TOV_per48_against'] +
                  (df_feat['FGA_per48_against'] -
                  df_feat['FGM_per48_against'])) / df_feat['Poss']

cols = ['eFG%', 'eFG%_against', 'TS%', 'TS%_against',
        'OffRat', 'DefRat',
        'OREB%', 'OREB_per48', 'DREB%', 'DREB_per48',
        'TOV%', 'TOV_forced%', 'STL%', 'AST%',
        'PTS_per48', 'PTS_per48_against',
        'FGM_per48','FGA_per48',
        'FG3M_per48','FG3A_per48',
        'FTM_per48','FTA_per48',
        'FGM_per48_against','FGA_per48_against',
        'FG3M_per48_against','FG3A_per48_against',
        'FTM_per48_against','FTA_per48_against']

for col in cols:
    print(col+'_z')
    df_feat[(col+'_z')] = ((df_feat[col] -
                            (df_feat[col].transform(lambda x: x.rolling(window=960).mean()))) /
                           df_feat[col].transform(lambda x: x.rolling(window=960).std()))

num_games = len(df_feat.index.get_level_values('GAME_ID').unique())

cols = ['Win', 'DaysRest', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FGM_per48', 'FGA_per48', 'FG3M_per48', 'FG3A_per48',
        'FTM_per48',
        'FTA_per48', 'OREB_per48', 'DREB_per48', 'REB_per48', 'AST_per48', 'TOV_per48', 'STL_per48', 'BLK_per48',
        'BLKA_per48', 'PF_per48', 'PFD_per48', 'PTS_per48', 'PLUS_MINUS_per48', 'FG_PCT_against',
        'FG3_PCT_against', 'FGM_per48_against', 'FGA_per48_against', 'FG3M_per48_against',
        'FG3A_per48_against', 'FTM_per48_against', 'FTA_per48_against', 'OREB_per48_against', 'DREB_per48_against',
        'REB_per48_against', 'AST_per48_against', 'TOV_per48_against', 'STL_per48_against', 'BLK_per48_against',
        'BLKA_per48_against', 'PF_per48_against', 'PFD_per48_against', 'PTS_per48_against',
        'eFG%', 'TS%', 'eFG%_against', 'TS%_against', 'DaysRest', 'roadtrip', 'Poss', 'OffRat', 'DefRat', 'OREB%',
        'DREB%',
        'TOV%', 'TOV_forced%', 'STL%', 'AST%',
        'eFG%_z', 'eFG%_against_z', 'TS%_against_z', 'OffRat_z', 'DefRat_z',
        'OREB%_z', 'OREB_per48_z', 'DREB%_z', 'DREB_per48_z', 'TOV%_z', 'TOV_forced%_z', 'STL%_z', 'AST%_z',
        'PTS_per48_z', 'PTS_per48_against_z', 'FGM_per48_z', 'FGA_per48_z', 'FG3M_per48_z', 'FG3A_per48_z',
        'FTM_per48_z', 'FTA_per48_z', 'FGM_per48_against_z', 'FGA_per48_against_z',
        'FG3M_per48_against_z', 'FG3A_per48_against_z', 'FTM_per48_against_z', 'FTA_per48_against_z']

prev_matchup_cols = ['OREB_per48', 'OREB%_z', 'DREB_per48', 'DREB%_z', 'REB_per48',
                     'FGA_per48', 'FG3A_per48', 'FTA_per48',
                     'WL', 'OffRat', 'DefRat', 'OffRat_z', 'DefRat_z',
                     'PTS_per48_z', 'PTS_per48_against_z',
                     'FGM_per48_z', 'FGA_per48_z', 'FG3M_per48_z', 'FG3A_per48_z',
                     'FTM_per48_z', 'FTA_per48_z', 'FGM_per48_against_z', 'FGA_per48_against_z',
                     'FG3M_per48_against_z', 'FG3A_per48_against_z', 'FTM_per48_against_z', 'FTA_per48_against']

num_games = len(df_feat.index.get_level_values('GAME_ID').unique())

for i, game_id in enumerate(df_feat.index.get_level_values('GAME_ID').unique()):
    # Print 'ok' every 5% of the loop
    if i % (num_games // 20) == 0:
        print(np.round(i / num_games * 100))

    team_abvs = df_feat.loc[:, game_id, :].index.get_level_values('TEAM_ABBREVIATION')
    if len(team_abvs) < 2:
        print('no opponent match')
    else:
        for col in cols:
            #         for col in df_feat.columns:
            col_name = col + '_opp'

            df_feat.loc[(slice(None), game_id, team_abvs[0]),
                        col_name] = df_feat.loc[(slice(None),
                                                 game_id, team_abvs[1])][col].iloc[0]

            df_feat.loc[(slice(None), game_id, team_abvs[1]),
                        col_name] = df_feat.loc[(slice(None),
                                                 game_id, team_abvs[0])][col].iloc[0]

            # calc rolling avg stats for previous 6 games against opponent ("_prev")
            if col in prev_matchup_cols:
                team1_index = df.loc[:, :, team_abvs[0]].index.get_level_values(1).unique()
                team2_index = df.loc[:, :, team_abvs[1]].index.get_level_values(1).unique()
                df_matchup = df_feat.loc[:, team1_index.intersection(team2_index), :]
                #                 df_matchup = df_feat.loc[:, df_feat.index.get_level_values('TEAM_ABBREVIATION').isin(team1_index.intersection(team2_index)), :]

                df_team1 = df_matchup.loc[:, :, team_abvs[0]][col].rolling(6).mean().shift(1)
                df_team2 = df_matchup.loc[:, :, team_abvs[1]][col].rolling(6).mean().shift(1)
                col_name = col + '_prev'

                df_feat.loc[(slice(None), game_id, team_abvs[0]),
                            col_name] = df_team1.loc[(slice(None), game_id)].iloc[0]
                df_feat.loc[(slice(None), game_id, team_abvs[1]),
                            col_name] = df_team2.loc[(slice(None), game_id)].iloc[0]

df_feat = df_feat.dropna()
df_feat.to_csv('features.csv')