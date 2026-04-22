import pandas as pd
import numpy as np
import json
import sqlite3
import time
import random
from datetime import datetime
from nba_api.stats.endpoints import teamgamelogs
from nba_api.stats.static import teams


class getData:

    def __init__(self, db_path='nba_database.db', n_games=82):
        self.db_path = db_path
        self.n_games = n_games
        self.teams = teams.get_teams()

    def fetch_team_gamelog(self, team_id, season, max_retries=10):

        headers = {
            "Host": "stats.nba.com",
            "Connection": "keep-alive",
            "Accept": "application/json, text/plain, /",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Referer": "https://www.nba.com/",
            "Origin": "https://www.nba.com",
            "x-nba-stats-origin": "stats",
            "x-nba-stats-token": "true",
            "Accept-Language": "en-US,en;q=0.9"
            }
        for attempt in range(max_retries):
            try:
                response = teamgamelogs.TeamGameLogs(
                    team_id_nullable=team_id,
                    season_nullable=season,
                    last_n_games_nullable=self.n_games,
                    timeout=2,
                    headers=headers
                ).get_json()

                data = json.loads(response)
                df = pd.DataFrame(
                    data['resultSets'][0]['rowSet'],
                    columns=data['resultSets'][0]['headers']
                )

                df['GAME_DATE'] = df['GAME_DATE'].str.split('T').str[0]
                df['GAME_ID'] = df['GAME_ID'].astype(int)

                return df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

            except Exception as e:
                time.sleep(random.uniform(1, 3))

        return None

    def get_current_season(self):
        now = datetime.now()
        if now.month >= 10:
            return f"{now.year}-{(now.year % 100) + 1}"
        else:
            return f"{now.year - 1}-{now.year % 100}"

    def get_last_updated_season(self, conn, seasons):
        '''
        open the formatted data table and find the most recent date
        return this date to help specify which season's stats to grab from the api
        '''
        try:
            df = pd.read_sql("SELECT MAX(GAME_DATE) as d FROM formatted_data_table", conn)
            date = df.iloc[0, 0]

            if pd.isna(date):
                return seasons[0]

            year, month, *_ = date.split('-')

            # nba season starts in october (month 10)
            if int(month) >= 10:
                return f"{year}-{(int(year) % 100) + 1}"
            else:
                return f"{int(year) - 1}-{int(year) % 100}"

        except:
            return seasons[0]

    def add_features(self, df):

        df['Home'] = df['MATCHUP'].str.contains('@').map({True: 0, False: 1})
        df['Win'] = df['WL'].map({'W': 1, 'L': 0})

        stats = [
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'OREB', 'DREB', 'REB', 'AST', 'TOV',
            'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS'
        ]

        # normalize relevant stats to per 48-minutes to correct for overtime games
        for col in stats:
            df[f"{col}_per48"] = (df[col] / df['MIN'] * 48).round(2)

        return df

    def add_opponent_stats(self, df):

        df_reset = df.reset_index()

        # self-merge on GAME_ID
        merged = df_reset.merge(
            df_reset,
            on='GAME_ID',
            suffixes=('', '_opp')
        )

        # remove self-join (team vs itself)
        merged = merged[
            merged['TEAM_ABBREVIATION'] != merged['TEAM_ABBREVIATION_opp']
        ]

        # keep only one opponent row per team
        merged = merged.drop_duplicates(
            subset=['GAME_ID', 'TEAM_ABBREVIATION']
        )

        # relevant opponents stat columns to copy
        cols = ['Win', 'FGM_per48', 'FGA_per48',
            'FG3M_per48', 'FG3A_per48',
            'FTM_per48', 'FTA_per48',
            'OREB_per48', 'DREB_per48',
            'REB_per48', 'AST_per48', 'TOV_per48',
            'STL_per48', 'BLK_per48', 'BLKA_per48',
            'PF_per48', 'PFD_per48', 'PTS_per48',
            'FG_PCT', 'FG3_PCT']

        for col in cols:
            merged[f"{col}_against"] = merged[f"{col}_opp"]

        keep_cols = [
            'GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'
        ] + list(df.columns) + [f"{c}_against" for c in cols]

        final = merged[keep_cols]

        return final.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

    def run(self):

        conn = sqlite3.connect(self.db_path)

        seasons = [f"{y}-{(y % 100) + 1}" for y in range(2022, 2030)]

        # construct indicies: grab the most recent season present from the database and get the current season based on today's date
        last_season = self.get_last_updated_season(conn, seasons)
        current_season = self.get_current_season()
        start_idx = seasons.index(last_season)
        end_idx = seasons.index(current_season)

        df_all = []

        # loop through all specified seasons to grab data season by season
        for season in seasons[start_idx:end_idx + 1]:
            print("\n")
            print("="*50)
            print(f"Fetching {season}")
            print("="*50)
            # loop through teams list and fetch their season's gamelogs
            i = 0
            for team in self.teams:
                print(f"Fetching {team['full_name']}...")
                df = self.fetch_team_gamelog(team['id'], season)
                if df is not None:
                    i += 1
                    df_all.append(df)
                    print("...complete")
                else:
                    print(f"Data aquisition failed for {team['full_name']}.")
            print("="*50)
            print(f"{season} complete. Data for {i} teams obtained successfully.")
            print("="*50)

        df = pd.concat(df_all)

        print("\nProcessing features...")
        df = self.add_features(df)
        print("...complete")

        print("\nMatching opponents...")
        df = self.add_opponent_stats(df)
        print("...complete")

        print("\nSaving to DB...")
        df.to_sql('formatted_data_table', conn, if_exists='replace')
        print("...complete")

        conn.close()
        print("\n")
        print("="*50)
        print("="*50)
        print("Data collection script complete.")
        print("="*50)
        print("="*50)

if __name__ == '__main__':
    data_getter = getData()
    data_getter.run()
