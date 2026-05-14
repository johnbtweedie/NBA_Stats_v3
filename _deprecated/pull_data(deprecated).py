import pandas as pd
import numpy as np
import json
from nba_api.stats.endpoints import playercareerstats, teamgamelogs, playergamelogs
from nba_api.stats.static import players, teams
import time

def get_team_gamelog_data(team_id, n_games, season, max_retries=3):
    attempt = 0
    while attempt < max_retries:
        try:
            team_gamelog_data = teamgamelogs.TeamGameLogs(team_id_nullable=team_id,
                                                          season_nullable=season,
                                                          last_n_games_nullable=n_games,
                                                          ).get_json()

            data_dict = json.loads(team_gamelog_data)
            df = pd.DataFrame(data_dict['resultSets'][0]['rowSet'],
                              columns=data_dict['resultSets'][0]['headers'])

            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

            return df
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            attempt += 1
            time.sleep(5)  # Wait for 5 seconds before retrying

    print(f"Max retries ({max_retries}) reached. Couldn't fetch data.")
    return None


teams_dict = teams.get_teams()
n_games = 82
seasons = ['2012-13',
           '2013-14',
           '2014-15',
           '2015-16',
           '2016-17',
           '2017-18',
           '2018-19',
           '2019-20',
           '2020-21',
           '2021-22',
           '2022-23',
           '2023-24']

df = pd.DataFrame()
for season in seasons:
    #print(season)
    for i in range(0,len(teams_dict)):
        df = pd.concat([df, get_team_gamelog_data(i, n_games, season)], axis=0)

# SAVE...use sql in future...
df.to_csv('historical_stats_raw.csv')

# factor encoding
df['Home'] = df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
df['Win'] = df['WL'].map({'W': 1, 'L': 0})

# calculate per-48 minute stats (corrects for overtime games)
cols = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
        'FTA','OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL',
        'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']

new_cols = []
for col in cols:
    new_col_name = col + '_per48'
    df[new_col_name] = df[col]/df['MIN']*48
    new_cols.append(new_col_name)

df[new_cols] = np.round(df[new_cols], 2)

cols = ['Win', 'FGM_per48', 'FGA_per48',
        'FG3M_per48', 'FG3A_per48',
        'FTM_per48', 'FTA_per48',
        'OREB_per48', 'DREB_per48',
        'REB_per48', 'AST_per48', 'TOV_per48',
        'STL_per48', 'BLK_per48', 'BLKA_per48',
        'PF_per48', 'PFD_per48', 'PTS_per48',
        'FG_PCT', 'FG3_PCT']

# Stats against

for game_id in df.index.get_level_values('GAME_ID').unique():

    team_abvs = df.loc[:, game_id, :].index.get_level_values('TEAM_ABBREVIATION')
    if len(team_abvs) < 2:
        print('no opponent match')
    else:
        for col in cols:
            col_name = col + '_against'

            df.loc[(slice(None), game_id, team_abvs[0]),
                   col_name] = df.loc[(slice(None),
                                       game_id, team_abvs[1])][col].iloc[0]

            df.loc[(slice(None), game_id, team_abvs[1]),
                   col_name] = df.loc[(slice(None),
                                       game_id, team_abvs[0])][col].iloc[0]


# SAVE for Feature Prep
df.to_csv('historical_stats.csv')
