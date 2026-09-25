import pandas as pd
import numpy as np
import sqlite3
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import pickle
import statsmodels.api as sm
import warnings

from db_utils import upsert_to_sql

warnings.filterwarnings('ignore')

TEAM_ABVS = ['ATL', 'BOS', 'BRK', 'CHI', 'CHA', 'CLE', 'DAL', 'DEN', 'DET', 'GSW',
             'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK',
             'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']

ARIMA_ORDERS_PATH = 'catalogs/parameters/arima_orders.pkl'
ARIMA_ENDOG_COLS = ['OffRat', 'DefRat']

class ComputeTeamFeatures:
    '''
    feature engineering only: turns raw team gamelogs into contemporaneous, per-team
    rolling/derived features (a row for (game, team) reflects that team's form as of
    and including that game). no shifting happens here - see build_datasets.py for
    how these features get turned into leakage-free training rows vs. unshifted
    deployment snapshots.
    '''
    def __init__(self, conn, purpose='train', refresh=False, tune_rolling_avg=False):

        # establish columns for various processing steps
        # ra - columns to perform rolling average on
        # opp - columns to use when matching/grabbing opponent stats
        # prev - columns to return previous n games avg against an opponent
        self.ra_cols = ['Win',
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
        self.opp_cols = ['Win',  'FG_PCT', 'FG3_PCT', 'FT_PCT', 'FGM_per48', 'FGA_per48', 'FG3M_per48', 'FG3A_per48',
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
        self.prev_cols = ['OREB_per48', 'OREB%_z', 'DREB_per48', 'DREB%_z', 'REB_per48',
                            'FGA_per48', 'FG3A_per48', 'FTA_per48',
                            'WL', 'OffRat', 'DefRat', 'OffRat_z', 'DefRat_z',
                            'PTS_per48_z', #'PTS_per48_against_z',
                            'FGM_per48_z', 'FGA_per48_z', 'FG3M_per48_z', 'FG3A_per48_z',
                            'FTM_per48_z', 'FTA_per48_z']#, 'FGM_per48_against_z', 'FGA_per48_against_z',
                            # 'FG3M_per48_against_z', 'FG3A_per48_against_z', 'FTM_per48_against_z', 'FTA_per48_against']
        self.conn = conn
        self.refresh = refresh

    def run(self):
        if self.refresh:
            self.raw_data = self.load_data()
        else:
            self.raw_data = self.load_required_formatted_data()
            print('data loaded')

        self.store_response_vars()
        self.compute_features()
        self.leaguewide_standardization()
        self.features = self.get_opponent_abv(self.features)
        self.previous_games_vs_opponent()

        self.features = self.compute_rolling_avg_optimal(df=self.features)
        self.features = self.compute_rolling_avg(df=self.features, window=8)
        self.features = self.compute_rolling_avg(df=self.features, window=16)
        self.features = self.compute_rolling_avg(df=self.features, window=32)
        self.features = self.compute_dfm(df=self.features)
        self.features = self.compute_arima(df=self.features)
        self.save_features()

        print('features computed')

    def load_required_formatted_data(self):
        try:
            df_feat_existing = pd.read_sql('SELECT * FROM team_features', self.conn)
            df_feat_existing = df_feat_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
            # df_feat_existing = df_feat_existing.sort_index(level=['GAME_ID'])

            unique_teams = df_feat_existing.index.get_level_values('TEAM_ABBREVIATION').unique()
            if 'NOH' in unique_teams:
                unique_teams = unique_teams.drop('NOH')

            # get the past 5 games rolling average stats vs the opponent
            print('determining required data for each matchup...')
            required_retrieval_index = []
            for i, team_1 in enumerate(unique_teams):
                for team_2 in unique_teams[i + 1:]:
                    df_team_1 = df_feat_existing.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)
                    df_team_2 = df_feat_existing.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)

                    shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

                    # store 6th most recent games index value
                    required_retrieval_index.append(shared_game_ids[-13]) # we take 13 to ensure enough games are there for each team to populate _prev 6 games rolling avg and all DaysRest correctly
            print('...complete')

            print('loading required data from database...')
            max_required_retrieval_index = pd.to_numeric(
                required_retrieval_index).min()  # index of the least recent game we have to grab data from
            df = pd.read_sql(
                f"SELECT * FROM team_gamelogs WHERE CAST(GAME_ID AS INTEGER) >= {max_required_retrieval_index}",
                self.conn)
            print('...complete')
        except (pd.errors.DatabaseError, sqlite3.OperationalError):
            print('no existing features detected, computing for full dataset')
            df = pd.read_sql(f"SELECT * FROM team_gamelogs", self.conn)

        df = self.format_team_gamelogs(df)

        return df

    def load_data(self):
        '''
        load data from sql db with multi-index and datetime GAME_DATE
        '''

        df = pd.read_sql(f"SELECT * FROM team_gamelogs", self.conn)
        df = self.format_team_gamelogs(df)

        return df

    def format_team_gamelogs(self, df):
        '''
        turn raw team gamelogs (as stored by pull_data_db_v2.py) into the formatted data:
        per-48 features and opponent stats added, multi-index and datetime GAME_DATE
        '''
        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION']).sort_index()

        df = self.add_features(df)
        df = self.add_opponent_stats(df)

        df = df.reset_index()
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION']).sort_index()
        df['WL'] = df['WL'].map({'W': 1, 'L': 0})

        return df

    def add_features(self, df):
        '''
        some light feature encoding/generation
        '''

        df['Home'] = df['MATCHUP'].str.contains('@').map({True: 0, False: 1})
        df['Win'] = df['WL'].map({'W': 1, 'L': 0})

        stats = [
            'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA',
            'OREB', 'DREB', 'REB', 'AST', 'TOV',
            'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS'
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

    def get_opponent_abv(self, df):
        '''
        get abbreviation of the opponent
        '''
        df['TEAM_ABBREVIATION_TEMP'] = df.index.get_level_values('TEAM_ABBREVIATION')

        df['oppAbv'] = (
            df.groupby('GAME_ID')['TEAM_ABBREVIATION_TEMP']
            .transform(lambda x: x[::-1].values)
        )

        df.drop(columns=['TEAM_ABBREVIATION_TEMP'], inplace=True)

        return df

    def store_response_vars(self):
        '''
        compute and store additional response features ('_r')
        these features won't be rolling averaged
        these will be used as response variables to predict
        '''
        print('storing response variables (_r)...')
        df = pd.DataFrame()
        df['Poss_r'] = (self.raw_data['FGA_per48'] - self.raw_data['OREB_per48'] + self.raw_data['TOV_per48'] +
                        0.4 * self.raw_data['FTA_per48'])

        df['OffRat_r'] = self.raw_data['PTS_per48'] / df['Poss_r'] * 100
        df['DefRat_r'] = self.raw_data['PTS_per48_against'] / df['Poss_r'] * 100

        df['PTS_per48_r'] = self.raw_data['PTS_per48']
        df['PTS_per48_against_r'] = self.raw_data['PTS_per48_against']
        df['PTS_per48_diff_r'] = df['PTS_per48_r'] - df['PTS_per48_against_r']

        df['WL_r'] = self.raw_data['WL']

        print('...complete')

        self.response = df
        upsert_to_sql(self.response, self.conn, 'response_table', pk_cols=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

    def compute_features(self):
        '''
        compute various features for model training
        offensive rating, defensing rating, assist rate, consecutive road games (roadtrip) etc
        '''
        print('computing features...')

        df_feat = self.raw_data.copy()
        df_feat = df_feat[self.ra_cols]

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
        df_feat['DaysRest'] = df_feat.sort_index().groupby('TEAM_ABBREVIATION', group_keys=False)['DaysElapsed'].diff()
        df_feat['DaysRest'] = df_feat['DaysRest'].apply(lambda x: 10 if x>10 else x)

        # misc
        df_feat[['MATCHUP','WL']] = self.raw_data[['MATCHUP','WL']]
        df_feat['Home'] = df_feat['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)
        df_feat.drop(columns=['MATCHUP'], inplace=True)


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

        df_roadtrip = self.raw_data.groupby('TEAM_ABBREVIATION', group_keys=False)['Home'].apply(lambda x: consecutive_zeros(x))

        groups = []
        for i in range(len(df_roadtrip)):
            df_group = pd.DataFrame(df_roadtrip.iloc[i],
                                    index=self.raw_data.loc[:, :, df_roadtrip.index[i]].index,
                                    columns=['roadtrip'])
            df_group['TEAM_ABBREVIATION'] = df_roadtrip.index[i]
            groups.append(df_group.set_index('TEAM_ABBREVIATION', append=True))
        df_feat['roadtrip'] = pd.concat(groups)['roadtrip']

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

        self.features = df_feat
        self.feature_list = df_feat.columns.to_list()
        print('...complete')

    def leaguewide_standardization(self, window=1230):
        '''
        standardize [cols] columns based on the [window]-game rolling mean and stdev across all teams
        '''
        df_feat = self.features
        cols = ['eFG%', 'eFG%_against', 'TS%', 'TS%_against',
                'OffRat', 'DefRat', 'Poss',
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
            df_feat[(col+'_z')] = ((df_feat[col] -
                                    (df_feat[col].transform(lambda x: x.rolling(window=window).mean()))) /
                                df_feat[col].transform(lambda x: x.rolling(window=window).std()))

        self.features = df_feat

    def previous_games_vs_opponent(self, ngames=6):
        '''
        get the stats from the previous n games of a given matchup
        these stats are not rolling averaged, but some are league-standardized
        '''
        df_feat = self.features
        prev_matchup_cols = self.prev_cols

        print('determining previous matchup stats...')
        rolling = (
            df_feat.groupby(
                [df_feat.index.get_level_values('TEAM_ABBREVIATION'), df_feat['oppAbv']],
                group_keys=False
            )[prev_matchup_cols]
            .apply(lambda g: g.rolling(ngames).mean())
            .add_suffix('_prev')
        )
        for col in rolling.columns:
            df_feat[col] = rolling[col]

        self.features = df_feat
        print('...complete')

    def tune_arima_orders(self, endog_cols=ARIMA_ENDOG_COLS, save_path=ARIMA_ORDERS_PATH):
        '''
        grid-search the best ARIMA(p,d,q) order per endog column (by average one-step-ahead
        forecast error across teams), and cache the winning orders to save_path for
        compute_arima() to consume. not called from run() - this is a standalone tuning
        step to be run manually whenever the feature set changes meaningfully.
        '''
        df = self.features.sort_index()
        print('tuning ARIMA orders...')
        p_values = [0, 1, 6, 20]
        d_values = [0, 1, 2, 5]
        q_values = [0, 1, 10, 40]
        param_grid = list(product(p_values, d_values, q_values))

        arima_orders = {}
        for endog_col in endog_cols:
            best_params = None
            best_error = float('inf')

            for p, d, q in param_grid:
                total_error = 0
                num_teams = 0

                for team, group in df.groupby('TEAM_ABBREVIATION', group_keys=False):
                    try:
                        series = group[endog_col].dropna().reset_index(drop=True)
                        model = ARIMA(series, order=(p, d, q)).fit()
                        forecast = model.forecast()
                        residual_error = np.abs(series.iloc[-1] - forecast.iloc[0])
                        total_error += residual_error
                        num_teams += 1
                    except Exception:
                        continue  # Skip if model fails

                avg_error = total_error / num_teams if num_teams > 0 else float('inf')

                if avg_error < best_error:
                    best_error = avg_error
                    best_params = (p, d, q)

            print(f'best ARIMA order for {endog_col}: {best_params} (avg residual error: {best_error})')
            arima_orders[endog_col] = best_params

        with open(save_path, 'wb') as f:
            pickle.dump(arima_orders, f)
        print(f'saved ARIMA orders to {save_path}')
        return arima_orders

    def compute_rolling_avg_exog(self, df, window=41):
        '''
        compute [window]-game rolling average for all teams for exogenous features
        used internally by compute_dfm as the DFM's exogenous input
        '''
        df = df.sort_index()
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv']

        # exogenous features rolling average
        # define the exogenous (opponent) features for the time series model below
        cols_to_transform = ['OER', 'DER', 'eFG%', 'eFG%_against', 'TS%', 'TS%_against',
                                'DaysElapsed', 'DaysRest', 'WL', 'Poss', 'OffRat', 'DefRat',
                                'OREB%', 'DREB%', 'TOV%', 'TOV_forced%', 'STL%', 'AST%', 'BB%',
                                'eFG%_z', 'eFG%_against_z', 'TS%_z', 'TS%_against_z',
                                'Poss_z', 'OffRat_z', 'DefRat_z',	'roadtrip', #'Home',
                                'OREB%_z', 'DREB%_z', 'TOV%_z', 'TOV_forced%_z', 'STL%_z', 'AST%_z']


        # Apply the transformation only to the specified columns
        df_transformed = (
            df.groupby('TEAM_ABBREVIATION', group_keys=False)
            .apply(lambda group: group[cols_to_transform].rolling(window=window, min_periods=window).mean())
        )

        df_transformed = df_transformed[cols_to_transform].add_suffix('_exog')

        # Keep the excluded columns intact
        if 'oppAbv' in df.columns:
            df_excluded = df[list_of_cols_to_exclude]
        else:
            list_of_cols_to_exclude.remove('oppAbv')
            df_excluded = df[list_of_cols_to_exclude]

        # Combine the transformed and excluded columns
        return pd.concat([df_excluded, df_transformed], axis=1)

    def compute_rolling_avg_optimal(self, df):
        '''
        compute [window]-game rolling average for all teams for [features] columns
        skips entirely if the optimal windows CSV hasn't been provided
        '''
        try:
            optimal_windows = pd.read_csv(r'catalogs/parameters/rolling_average_windows.csv', index_col=0)
        except FileNotFoundError:
            print('no optimal rolling window file found, skipping optimal-window features')
            return df

        df = df.sort_index()
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv']
        list_of_cols_to_exclude.extend([col for col in df.columns if '_prev' in col])

        for feature, window in optimal_windows.iterrows():
            window = int(window.values[0])

            if not feature in list_of_cols_to_exclude:
                # Apply the transformation only to the specified columns
                df_transformed = (
                    df.groupby('TEAM_ABBREVIATION', group_keys=False)
                    .apply(lambda group: group[feature].rolling(window=window, min_periods=window).mean())
                )

                new_label = feature + '_raop'
                df[new_label] = df_transformed

        return df

    def compute_rolling_avg(self, df, window=None):
        '''
        compute [window]-game rolling average for all teams for [features] columns
        '''
        df = df.sort_index()
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv', 'GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION']
        list_of_cols_to_exclude.extend([col for col in df.columns if '_prev' in col])
        list_of_cols_to_exclude.extend([col for col in df.columns if '_ra' in col])
        cols_to_transform = df.columns.difference(list_of_cols_to_exclude)

        # Apply the transformation only to the specified columns
        df_transformed = (
            df.groupby('TEAM_ABBREVIATION', group_keys=False)
            .apply(lambda group: group[cols_to_transform].rolling(window=window, min_periods=window).mean())
        )
        new_label = '_ra' +str(window)
        df_transformed = df_transformed[cols_to_transform].add_suffix(new_label)

        # Combine the transformed and excluded columns
        return pd.concat([df, df_transformed], axis=1)

    def _flip_to_opponent(self, df, exclude_cols=('oppAbv',)):
        '''
        for each column (except exclude_cols), add a same-named _opp column holding
        that row's opponent's value for the same game (assumes exactly two rows per
        GAME_ID) - used by compute_dfm to build opponent-matched exog inputs
        '''
        cols_to_flip = [col for col in df.columns if col not in exclude_cols]
        opp_df = (
            df[cols_to_flip]
            .groupby('GAME_ID')
            .transform(lambda x: x[::-1].values)
            .add_suffix('_opp')
        )
        return pd.concat([df, opp_df], axis=1)

    def compute_dfm(self, df):
        '''
        apply the fitted DFM time series models to relevant feature columns
        skips entirely if the DFM parameter pickles haven't been provided

        exog inputs are lagged by one game per team before fitting, so the resulting
        _dfm columns already reflect only information available before the game they're
        attached to - they must NOT be shifted again downstream (see build_datasets.py)

        the exog inputs also include opponent-matched (_opp) versions of some trailing
        stats - fitted DFMs may condition a team's factors on their opponent's form
        '''
        dfm_paths = ['catalogs/parameters/dfm_1.pkl', 'catalogs/parameters/dfm_2.pkl']
        dfms = []
        for dfm_path in dfm_paths:
            try:
                with open(dfm_path, 'rb') as file:
                    dfms.append(pickle.load(file))
            except FileNotFoundError:
                print(f'no DFM parameter file found at {dfm_path}, skipping DFM features')
                return df

        df_exog = self.compute_rolling_avg_exog(df)

        # lag the rolling-average exog columns by one game per team so they only ever
        # reflect information available before the game they're attached to - the raw
        # pass-through columns (Home, roadtrip, DaysRest, DaysElapsed, oppAbv) stay
        # contemporaneous, matching how they're treated everywhere else in the pipeline
        exog_cols_to_shift = [c for c in df_exog.columns if c.endswith('_exog')]
        df_exog[exog_cols_to_shift] = (
            df_exog.groupby('TEAM_ABBREVIATION', group_keys=False)[exog_cols_to_shift]
            .apply(lambda g: g.shift(1))
        )
        df_exog = self._flip_to_opponent(df_exog)

        for dfm in dfms:
            endog_1 = dfm['ATL']['endog']
            exog_1 = dfm['ATL']['exog']
            exog_ra8 = [x for x in exog_1 if '_ra8' in x]
            exog_ra32 = [x for x in exog_1 if '_ra32' in x]
            exog_1 = [x for x in exog_1 if not '_ra' in x]

            for team, group in df.groupby('TEAM_ABBREVIATION', group_keys=False):
                if team not in dfm:
                    continue

                endog_series = group[endog_1].apply(pd.to_numeric, errors='coerce').dropna().sort_index()
                exog_series = (
                    df_exog.xs(team, level='TEAM_ABBREVIATION', drop_level=False)[exog_1]
                    .dropna()
                    .sort_index()
                )

                for feature in exog_ra8:
                    exog_series[feature] = endog_series[feature.split('_ra')[0]].rolling(window=8).mean().shift(1)
                for feature in exog_ra32:
                    exog_series[feature] = endog_series[feature.split('_ra')[0]].rolling(window=32).mean().shift(1)

                exog_series = exog_series.dropna()
                endog_series = endog_series.dropna()
                common_indices = exog_series.index.intersection(endog_series.index)

                exog_series = exog_series.loc[common_indices]
                endog_series = endog_series.loc[common_indices]

                exog_series = exog_series.reset_index(drop=True)
                endog_series = endog_series.reset_index(drop=True)

                k_factors = dfm[team]["k_factors"]
                factor_order = dfm[team]["factor_order"]
                params = dfm[team]['params']

                # Apply the fitted model's parameters to the new data using filtering
                mod = sm.tsa.DynamicFactor(
                    endog=endog_series,
                    exog=exog_series,
                    k_factors=k_factors,
                    factor_order=factor_order
                )

                transformed_result = mod.filter(params)
                filtered_df = transformed_result.fittedvalues.copy()
                filtered_df.index = common_indices
                filtered_df = filtered_df.add_suffix('_dfm')

                for col in filtered_df.columns:
                    if col not in df:
                        df[col] = np.nan

                df.update(filtered_df)
        print('...complete')
        return df

    def compute_arima(self, df, endog_cols=ARIMA_ENDOG_COLS, params_path=ARIMA_ORDERS_PATH):
        '''
        fit an ARIMA(p,d,q) per team per endog column and add its one-step-ahead
        in-sample forecast as a {col}_arima feature. statsmodels' state-space ARIMA
        fittedvalues are one-step-ahead in-sample predictions (using data through the
        prior game), so this is already leakage-safe and must NOT be shifted again
        downstream (see build_datasets.py).

        skips entirely if the ARIMA orders haven't been tuned/provided yet (see
        tune_arima_orders).
        '''
        try:
            with open(params_path, 'rb') as f:
                arima_orders = pickle.load(f)
        except FileNotFoundError:
            print(f'no ARIMA order file found at {params_path}, skipping ARIMA features')
            return df

        df = df.sort_index()
        for col in endog_cols:
            order = arima_orders.get(col)
            if order is None:
                continue

            print(f'fitting ARIMA{order} for {col}...')
            forecasts = []
            for team, group in df.groupby('TEAM_ABBREVIATION', group_keys=False):
                series = group[col].dropna()
                if len(series) < 20:
                    continue
                try:
                    fitted = ARIMA(series.reset_index(drop=True), order=order).fit().fittedvalues
                    fitted.index = series.index
                    forecasts.append(fitted)
                except Exception as e:
                    print(f'ARIMA fit failed for {team}/{col}: {e}')

            if forecasts:
                df[f'{col}_arima'] = pd.concat(forecasts)
        return df

    def save_features(self):
        upsert_to_sql(self.features, self.conn, 'team_features', pk_cols=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

class ComputePlayerFeatures(ComputeTeamFeatures):
    def __init__(self, conn, purpose='predict', refresh=False):
        self.conn = conn
        self.refresh = refresh
        self.meta_features = ['gameId', 'teamId', 'teamCity', 'teamName', 'teamTricode', 'teamSlug',
                                    'personId', 'firstName', 'familyName', 'nameI', 'playerSlug',
                                    'position', 'comment', 'jerseyNum', 'minutes']

    def run(self):
        if self.refresh:
            self.raw_data = self.load_data()
        else:
            self.raw_data = self.load_required_formatted_data()
            print('data loaded')

        self.compute_features()
        self.save_features()
        print('features computed')

    def load_data(self):
        '''
        load data from sql db with multi-index and datetime GAME_DATE
        '''
        return pd.read_sql(f"SELECT * FROM player_gamelogs", self.conn).set_index(['gameId', 'teamId', 'personId']).sort_index()

    def compute_features(self):
        '''
        compute various features for model training
        offensive rating, defensing rating, assist rate, consecutive road games (roadtrip) etc
        '''
        print('computing features...')
        df = self.raw_data.copy()
        df = self.format_raw_player_gamelogs(df)
        df = self.rolling_average(df=df, window=8)
        self.features = df
        self.rotation_features = self.build_rotation_table(df=df, n=4)
        print('...complete')

    def format_raw_player_gamelogs(self, df):
        '''
        turn raw player gamelogs (as stored by pull_data_db_v2.py) into the formatted data:
        per-48 features and opponent stats added, multi-index and datetime GAME_DATE
        '''
        df['minutes_decimal'] = df['minutes'].apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1])/60 if x is not None and x != '' else 0)

        return df

    def rolling_average(self, df, window=8):
        '''
        compute [window]-game rolling average for all players for [features] columns
        '''
        cols_to_transform = df.columns.difference(self.meta_features)

        # get rolling average for all players, for team's last n games played (not excluding games where they did not play (i.e. minutes = 0))
        df_transformed = (
            df.groupby('personId', group_keys=False)
            .apply(lambda group: group[cols_to_transform].rolling(window=window, min_periods=window).mean())
        )
        new_label = '_ra' +str(window)
        df_transformed = df_transformed[cols_to_transform].add_suffix(new_label)

        # get rolling average for all players, for player's last n games played (yes excluding games where they did not play (i.e. minutes = 0))
        transformed_groups = []
        for _, group in df.groupby('personId', group_keys=False):
            group = group[group['minutes_decimal'] > 0]
            group_transformed = group[cols_to_transform].rolling(window=window, min_periods=window).mean()
            transformed_groups.append(group_transformed)
        df_transformed_no_zero = pd.concat(transformed_groups).reindex(df.index)
        new_label = '_ra-excl' +str(window)
        df_transformed_no_zero = df_transformed_no_zero.add_suffix(new_label)

        # Combine the transformed and excluded columns
        return pd.concat([df, df_transformed, df_transformed_no_zero], axis=1)

    def build_rotation_table(self, df, n=4, usage_stat_col='usagePercentage_ra8'):
        '''
        compact (gameId, teamId)-indexed table of the top n rotation players' stats
        (by rolling usage) for each team in each game, plus their aggregate mean -
        these are static, per-team-per-game features derived from player gamelogs,
        distinct from the full per-player rolling feature table
        '''
        cols_to_transform = df.columns.difference(self.meta_features)
        df = df.sort_index()
        df = df.loc[df['teamTricode'].isin(TEAM_ABVS), :]  # removes random pre season games

        def top_n_row(game_group):
            game_group = game_group.loc[game_group['comment'] == '']  # exclude players who didn't play
            top_n_players = game_group.nlargest(n, usage_stat_col)

            row = {}
            for player_num, (_, player_row) in enumerate(top_n_players.iterrows(), start=1):
                for col in cols_to_transform:
                    row[f'{col}_top{player_num}'] = player_row[col]
            for col in cols_to_transform:
                row[f'{col}_top{n}_agg'] = top_n_players[col].mean()
            return pd.Series(row)

        print(f'computing top {n} rotation stats...')
        rotation = df.groupby(['gameId', 'teamId'], group_keys=True).apply(top_n_row)
        print('...complete')
        return rotation

    def save_features(self):
        upsert_to_sql(self.features, self.conn, 'player_features', pk_cols=['gameId', 'teamId', 'personId'])
        upsert_to_sql(self.rotation_features, self.conn, 'team_rotation_features', pk_cols=['gameId', 'teamId'])

if __name__ == '__main__':
    print('computing features for player data...')
    PlayerFeatures = ComputePlayerFeatures(conn=sqlite3.connect('nba_database_test.db'),
                                       purpose='predict',
                                       refresh=True)
    PlayerFeatures.run()
    print('player features complete')

    print('computing features for team data...')
    TeamFeatures = ComputeTeamFeatures(conn=sqlite3.connect('nba_database_test.db'),
                                   purpose='predict',
                                   refresh=True)
    TeamFeatures.run()
    print('team features complete')
    print('all features computed and saved to database')
