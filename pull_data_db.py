import pandas as pd
import numpy as np
import json
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams
import time
import sqlite3
from datetime import datetime
import random

def get_team_gamelog_data(team_id, n_games, season, max_retries=10):
    attempt = 0
    while attempt < max_retries:
        try:
            team_gamelog_data = teamgamelogs.TeamGameLogs(team_id_nullable=team_id,
                                                          season_nullable=season,
                                                          last_n_games_nullable=n_games,
                                                          timeout=1
                                                          ).get_json()

            data_dict = json.loads(team_gamelog_data)
            df = pd.DataFrame(data_dict['resultSets'][0]['rowSet'],
                              columns=data_dict['resultSets'][0]['headers'])

            # df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            df['GAME_DATE'] = df['GAME_DATE'].str.split('T').str[0]
            df['GAME_ID'] = df['GAME_ID'].astype('int')
            df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

            return df
        except Exception as e:
            print(f"Error: {e}. Retrying...")
            attempt += 1
            time.sleep(random.randint(1, 5))  # Wait for 5 seconds before retrying

    print(f"Max retries ({max_retries}) reached. Couldn't fetch data.")
    return None

def update_database():

    # Create a connection to the SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('nba_database_2025-01-18.db')
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
               '2023-24',
               '2024-25',
               '2025-26',
               '2026-27',
               '2027-28',
               '2028-29',
               '2029-30']

    # grab the most recent game's date
    query = """
    SELECT MAX(GAME_DATE) as most_recent_date
    FROM formatted_data_table
    """
    try:
        most_recent_game_date = pd.read_sql(query, conn)

        # get string of the most recent game's year and month
        # use these to determine which seasons needs to be updated
        year_last_updated = most_recent_game_date.iloc[0].str.split('-').str[0].iloc[0]
        month_last_updated = most_recent_game_date.iloc[0].str.split('-').str[1].iloc[0]

        if str(month_last_updated) == 'None':
            month_last_updated = '1'

        if str(year_last_updated) == 'None':
            year_last_updated = seasons[1].split('-')[0]

        if int(month_last_updated) >= 10:
            season_last_updated = str(int(year_last_updated)) + '-' + str((int(year_last_updated) % 100) + 1)
        else:
            season_last_updated = str(int(year_last_updated) - 1) + '-' + str(int(year_last_updated) % 100)

        position = seasons.index(season_last_updated)

    except Exception as e:
        # season_last_updated = '2012-13'
        season_last_updated = seasons[0]
        position = seasons.index(season_last_updated)

    if datetime.now().month >= 10:
        current_season = str(int(datetime.now().year)) + '-' + str((int(datetime.now().year) % 100) + 1)
    else:
        current_season = str(int(datetime.now().year) - 1) + '-' + str(int(datetime.now().year) % 100)

    # update from the season last updated until the current season
    df = pd.DataFrame()
    for season in seasons[(seasons.index(season_last_updated)):(seasons.index(current_season)+1)]:
        print('grabbing api data for ', season)
        for i in range(0, len(teams_dict)):
            df = pd.concat([df, get_team_gamelog_data(i, n_games, season)], axis=0)

    # factor encoding
    print('processing api data...')
    df['Home'] = df['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
    df['Win'] = df['WL'].map({'W': 1, 'L': 0})

    # calculate per-48 minute stats (corrects for overtime games)
    cols = ['FGM', 'FGA', 'FG3M', 'FG3A', 'FTM',
            'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL',
            'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']

    new_cols = []
    for col in cols:
        new_col_name = col + '_per48'
        df[new_col_name] = df[col] / df['MIN'] * 48
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
    print('matching opponents for stats against...')
    total_games = len(df.index.get_level_values('GAME_ID').unique())
    interval = total_games / 10

    for i, game_id in enumerate(df.index.get_level_values('GAME_ID').unique()):
        if i % int(interval) == 0:
            print(f"Progress: {int(100 * i / total_games)}%")

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

    print('saving processed data...')

    try:
        df_existing = pd.read_sql('SELECT * FROM formatted_data_table', conn)
        # df_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
    except pd.io.sql.DatabaseError:
        print('no existing data detected')
        columns = df.reset_index().columns.to_list()
        df_existing = pd.DataFrame(columns=columns)

    df_existing['GAME_ID'] = df_existing['GAME_ID'].astype('int')
    df_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'], inplace=True)

    if 'level_0' in df_existing.columns:
        df_existing.drop('level_0', axis=1, inplace=True)
    if 'index' in df_existing.columns:
        df_existing.drop('index', axis=1, inplace=True)

    combined_data = pd.concat([df_existing, df])
    # combined_data.reset_index(inplace=True)
    # combined_data.drop('index', axis=1, inplace=True)
    #
    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
    combined_data = combined_data.sort_index(level=['GAME_ID'])
    # combined_data = combined_data.sort_values(by='GAME_ID')



    print('saving to database...')
    combined_data.to_sql('formatted_data_table', conn, if_exists='replace', index=True)
    combined_data.to_csv('historical_stats.csv')

    # df_new = pd.read_sql('SELECT * FROM formatted_data_table', conn)
    conn.close()

if __name__ == '__main__':
    update_database()

print('complete')
