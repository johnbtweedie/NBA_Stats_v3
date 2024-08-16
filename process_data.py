import pandas as pd
import numpy as np
import sqlite3
import time
from nba_api.stats.static import teams

# LOAD
print('reading databse')
conn = sqlite3.connect('nba_database_2024-07-24.db')
# df = pd.read_csv('historical_stats.csv')
df = pd.read_sql('SELECT * FROM formatted_data_table', conn)
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
df = df.sort_index(level=['TEAM_ABBREVIATION', 'GAME_DATE'])



# Create a connection to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('nba_database_2024-07-24.db')
teams_dict = teams.get_teams()

# grab the most recent game's date
query = """
SELECT MAX(GAME_DATE) as most_recent_date
FROM feature_table
"""
most_recent_game_date = pd.read_sql(query, conn)

df_feat_existing = pd.read_sql('SELECT * FROM feature_table', conn)
'''
To DO:
[x] find most game date from features table (and?)
[] find the index of the 5th (4th?) most recent entry of each team matchup
[] find the furthest-back index of these
[] use this index to grab the required data from the formatted_data_table
[] run the feature computations on this data
[] merge with full  
'''
unique_teams = df.index.get_level_values('TEAM_ABBREVIATION').unique()
unique_teams = unique_teams.drop('NOH')

# get the past 5 games rolling average stats vs the opponent
print('determining previous matchup stats...')
required_retrieval_index = []
for i, team_1 in enumerate(unique_teams):
    print('Processing', i + 1, 'of', len(unique_teams), 'teams')

    for team_2 in unique_teams[i + 1:]:
        # #
        # team_1 = 'ATL'
        # team_2 = 'NOP'
        print(team_1, 'vs.', team_2)
        df_team_1 = df.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)
        df_team_2 = df.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)

        shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

        # store 5th most recent games index value
        required_retrieval_index.append(shared_game_ids[-4])

max_required_retrieval_index = pd.to_numeric(required_retrieval_index).min() # index of the least recent game we have to grab data from

def compute_features():
    cols = ['Win',
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
    print('feature engineering')
    # rolling average for training data (shifted by 1 game)
    df = df.sort_index(level=['GAME_DATE'], ascending=True)
    df_feat = df.groupby('TEAM_ABBREVIATION',
               group_keys=False)[cols].apply(lambda x: x.rolling(window=41,
                                                                 min_periods=1).mean())#.shift(1)) # shifting in train models now
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
    df_feat['DaysRest'] = df_feat.groupby('TEAM_ABBREVIATION', group_keys=False)['DaysElapsed'].diff()
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

    # leaguewide rolling standardization
    window = 960
    for col in cols:
        print(col+'_z')
        df_feat[(col+'_z')] = ((df_feat[col] -
                                (df_feat[col].transform(lambda x: x.rolling(window=window).mean()))) /
                               df_feat[col].transform(lambda x: x.rolling(window=window).std()))

    num_games = len(df_feat.index.get_level_values('GAME_ID').unique())

    cols = ['Win',  'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FGM_per48', 'FGA_per48', 'FG3M_per48', 'FG3A_per48',
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


    print('opponent features')

    num_games = len(df_feat.index.get_level_values('GAME_ID').unique())

    for col in prev_matchup_cols:
        prev_col = col + '_prev'
        if prev_col not in df_feat.columns:
            df_feat[prev_col] = np.nan

    opp_cols = []
    for col in cols:
        opp_col = col + '_opp'
        opp_cols.append(opp_col)
        df_feat[opp_col] = np.nan

    unique_teams = df_feat.index.get_level_values('TEAM_ABBREVIATION').unique()

    # get the past 5 games rolling average stats vs the opponent
    print('determining previous matchup stats...')
    for i, team_1 in enumerate(unique_teams):
        print('Processing', i + 1, 'of', len(unique_teams), 'teams')

        for team_2 in unique_teams[i + 1:]:
            start_time = time.time()  # Start time of the loop

            print(team_1, 'vs.', team_2)
            df_team_1 = df_feat.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)[prev_matchup_cols]
            df_team_2 = df_feat.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)[prev_matchup_cols]

            shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

            df_feat.update(df_team_1.loc[:,shared_game_ids,:][prev_matchup_cols].rolling(5).mean().shift(1).add_suffix('_prev'))
            df_feat.update(df_team_2.loc[:,shared_game_ids,:][prev_matchup_cols].rolling(5).mean().shift(1).add_suffix('_prev'))

            # df_team_1 = df_team_1.loc[:,shared_game_ids,:]
            # df_team_2 = df_team_2.loc[:,shared_game_ids,:]
            #
            # df_team_1 = df_team_1[prev_matchup_cols].rolling(5).mean().shift(1)
            # df_team_2 = df_team_2[prev_matchup_cols].rolling(5).mean().shift(1)
            #
            # df_team_1 = df_team_1.add_suffix('_prev')
            # df_team_2 = df_team_2.add_suffix('_prev')
            #
            # df_feat.update(df_team_1)
            # df_feat.update(df_team_2)

            # End time of the loop and print the time taken
            elapsed_time = time.time() - start_time

            print(f"Time taken for {team_1} vs {team_2}: {elapsed_time:.4f} seconds")

    # match opponent stats
    print('matching opponent stats...')
    for i, team_1 in enumerate(unique_teams):
        print('processing ', i + 1, ' of ', len(unique_teams), ' teams')
        for team_2 in unique_teams[i + 1:]:
            print(team_1, ' vs. ', team_2)

            # get dataframe of each teams features/stats
            df_team_1 = df_feat.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)[cols]
            df_team_2 = df_feat.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)[cols]

            shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

            df_team_1 = df_team_1.loc[:,shared_game_ids,:]
            df_team_2 = df_team_2.loc[:,shared_game_ids,:]

            df_team_1_opp = df_team_2.add_suffix('_opp')
            df_team_2_opp = df_team_1.add_suffix('_opp')

            team_1_index = df_team_1_opp.index.intersection(df_feat.index)
            team_2_index = df_team_2_opp.index.intersection(df_feat.index)

            df_feat.loc[team_1_index, opp_cols] = df_team_1_opp.loc[team_1_index, opp_cols]
            df_feat.loc[team_2_index, opp_cols] = df_team_2_opp.loc[team_2_index, opp_cols]

    print('saving features')
    df_feat = df_feat.dropna()
    # df_feat.to_csv('features.csv')
    df_feat.to_sql('feature_table', conn, if_exists='replace', index=True)
    conn.close()

print('complete')