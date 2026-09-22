import pandas as pd
import json
import sqlite3
import time
import random
from datetime import datetime
from nba_api.stats.endpoints import teamgamelogs, boxscoreadvancedv3
from nba_api.stats.library.http import NBAStatsHTTP
from nba_api.stats.static import teams
from requests.exceptions import ReadTimeout, ConnectionError
from curl_cffi import requests as cr
from curl_cffi.requests.exceptions import RequestException as CurlRequestException


class getData:
    '''
    pull raw data from stats.nba.com and store it, untouched, in the database:
        team_gamelogs   - one row per team per game
        player_gamelogs - one row per player per game (advanced box score)
    all feature generation happens downstream in process_data_v2.py
    '''

    def __init__(self, db_path='nba_database_test.db', n_games=82):
        self.db_path = db_path
        self.n_games = n_games
        self.teams = teams.get_teams()
        self.init_session()

    def init_session(self):
        '''
        stats.nba.com drops connections that don't look like a real browser, so route
        all nba_api requests through a session that impersonates chrome's TLS handshake
        '''
        session = cr.Session(impersonate="chrome120")

        # warm up session cookies by hitting the main page first
        try:
            session.get("https://www.nba.com/stats/", timeout=20)
        except CurlRequestException as e:
            print(f"Session warm-up failed ({e}); continuing anyway.")

        NBAStatsHTTP.set_session(session)

    def fetch_team_gamelog(self, team_id, season, max_retries=15):

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
                    # nba_api 1.11.4 otherwise sends MeasureType=None, which the API rejects with a 400
                    measure_type_player_game_logs_nullable='Base',
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

                return df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION']).sort_index()

            except Exception as e:
                time.sleep(random.uniform(1, 3))

        return None

    def fetch_player_gamelog(self, game_id, max_retries=5, timeout=30):
        '''
        get every player's advanced box score for one game, retrying with exponential backoff on timeouts
        '''
        for attempt in range(max_retries):
            try:
                boxscore = boxscoreadvancedv3.BoxScoreAdvancedV3(
                    game_id=str(game_id).zfill(10),
                    timeout=timeout
                )
                # the first DataFrame of those returned is the player-level stats
                return boxscore.get_data_frames()[0]
            except (ReadTimeout, ConnectionError, CurlRequestException):
                wait = min(2 ** attempt, 30) + random.uniform(0, 1)
                print(f"   timed out (attempt {attempt + 1}/{max_retries}), retrying in {wait:.0f}s")
                time.sleep(wait)

        return None

    def get_current_season(self):
        now = datetime.now()
        if now.month >= 10:
            return f"{now.year}-{(now.year % 100) + 1}"
        else:
            return f"{now.year - 1}-{now.year % 100}"

    def get_last_updated_season(self, conn, seasons):
        '''
        open the team gamelogs table and find the most recent date
        return this date to help specify which season's stats to grab from the api
        '''
        try:
            df = pd.read_sql("SELECT MAX(GAME_DATE) as d FROM team_gamelogs", conn)
            date = df.iloc[0, 0]

            if pd.isna(date):
                return seasons[0]

            year, month, *_ = date.split('-')

            # nba season starts in october (month 10)
            if int(month) >= 10:
                return f"{year}-{(int(year) % 100) + 1}"
            else:
                return f"{int(year) - 1}-{int(year) % 100}"

        except pd.errors.DatabaseError:
            # table doesn't exist yet
            return seasons[0]

    def get_saved_game_ids(self, conn):
        '''
        game ids already in the player gamelogs table, so an interrupted run can pick up where it left off
        '''
        try:
            df = pd.read_sql("SELECT DISTINCT gameId FROM player_gamelogs", conn)
            return set(df['gameId'].astype(int))
        except pd.errors.DatabaseError:
            return set()

    def save_team_gamelogs(self, conn, df):
        '''
        add team gamelogs to the database; the in-progress season gets re-pulled on every run,
        so first clear out any rows already stored for the same game/team
        '''
        df = df.reset_index()
        try:
            conn.executemany(
                "DELETE FROM team_gamelogs WHERE GAME_ID = ? AND TEAM_ABBREVIATION = ?",
                [(int(game_id), abv) for game_id, abv in zip(df['GAME_ID'], df['TEAM_ABBREVIATION'])]
            )
        except sqlite3.OperationalError:
            # table doesn't exist yet
            pass
        df.to_sql('team_gamelogs', conn, if_exists='append', index=False)

    def pull_team_gamelogs(self, conn):
        seasons = [f"{y}-{(y % 100) + 1}" for y in range(2015, 2030)]

        # construct indicies: grab the most recent season present from the database and get the current season based on today's date
        last_season = self.get_last_updated_season(conn, seasons)
        current_season = self.get_current_season()
        start_idx = seasons.index(last_season)
        end_idx = seasons.index(current_season)

        # loop through all specified seasons to grab data season by season
        for season in seasons[start_idx:end_idx + 1]:
            print("\n")
            print("="*50)
            print(f"Fetching {season}")
            print("="*50)
            # loop through teams list and fetch their season's gamelogs
            df_season = []
            for team in self.teams:
                print(f"Fetching {team['full_name']}...")
                df = self.fetch_team_gamelog(team['id'], season)
                if df is not None:
                    df_season.append(df)
                    print("...complete")
                else:
                    print(f"Data aquisition failed for {team['full_name']}.")
                # pause between teams to avoid being throttled by stats.nba.com
                time.sleep(random.uniform(0.5, 1))

            print("="*50)
            print(f"{season} complete. Data for {len(df_season)} teams obtained successfully.")
            print("="*50)

            # save each season as it completes so an interrupted run keeps its progress
            if df_season:
                self.save_team_gamelogs(conn, pd.concat(df_season))

    def pull_player_gamelogs(self, conn):
        '''
        pull the player box score for every game in the team gamelogs table that we don't have yet
        '''
        team_game_ids = set(pd.read_sql("SELECT DISTINCT GAME_ID FROM team_gamelogs", conn)['GAME_ID'].astype(int))
        game_ids = sorted(team_game_ids - self.get_saved_game_ids(conn))

        print("\n")
        print("="*50)
        print(f"Fetching player data for {len(game_ids)} games...")
        print("="*50)

        failed = []
        for n, game_id in enumerate(game_ids, start=1):
            df_player = self.fetch_player_gamelog(game_id)
            if df_player is None:
                failed.append(game_id)
            else:
                df_player.to_sql('player_gamelogs', conn, if_exists='append', index=False)
            if n % 50 == 0:
                print(f"   {n}/{len(game_ids)} games")
            time.sleep(random.uniform(0.3, 0.8))

        print("="*50)
        print(f"Player data complete. {len(game_ids) - len(failed)}/{len(game_ids)} games saved.")
        if failed:
            print(f"Failed game IDs (will be retried on the next run): {failed}")
        print("="*50)

    def run(self):

        conn = sqlite3.connect(self.db_path)

        self.pull_team_gamelogs(conn)
        self.pull_player_gamelogs(conn)

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
