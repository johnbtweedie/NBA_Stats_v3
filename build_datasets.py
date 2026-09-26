import numpy as np
import pandas as pd
import pickle
import sqlite3
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
import warnings

from db_utils import upsert_to_sql
from process_data_v2 import ARIMA_ORDERS_PATH, ARIMA_ENDOG_COLS

warnings.filterwarnings('ignore')

# same list used by ComputeTeamFeatures.previous_games_vs_opponent in process_data_v2.py -
# needed here to know which _prev (rolling avg stats of previous n games vs opponent) columns get 
# shifted within (team, oppAbv) groups rather than within team groups
PREV_COLS = ['OREB_per48', 'OREB%_z', 'DREB_per48', 'DREB%_z', 'REB_per48',
             'FGA_per48', 'FG3A_per48', 'FTA_per48',
             'WL', 'OffRat', 'DefRat', 'OffRat_z', 'DefRat_z',
             'PTS_per48_z',
             'FGM_per48_z', 'FGA_per48_z', 'FG3M_per48_z', 'FG3A_per48_z',
             'FTM_per48_z', 'FTA_per48_z']


class BuildDatasets:
    '''
    dataset construction: turns the contemporaneous, unshifted team_features table
    (produced by process_data_v2.py) into the datasets actually ingested downstream:

      - modeling_dataset: leakage-free (shifted one game back), opponent-matched,
        response-joined, with a dataset_split column (train/test/validation) - for
        train_models.py
      - deployment_team_data / deployment_prev_matchup_data: unshifted, most-recent-game
        snapshot per team (and per team/opponent) - for predicting the next, unplayed
        game in make_predictions.py

    see process_data_v2.py's compute_dfm/compute_arima docstrings for why _dfm/_arima
    columns are excluded from the shift here - they're already leakage-safe forecasts
    of the row they're attached to.
    '''

    def __init__(self, conn):
        self.conn = conn

    def run(self):
        self.team_features = (
            pd.read_sql('SELECT * FROM team_features', self.conn)
            .assign(GAME_DATE=lambda d: pd.to_datetime(d['GAME_DATE']))
            .set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
            .sort_index()
        )
        self.response = (
            pd.read_sql('SELECT * FROM response_table', self.conn)
            .assign(GAME_DATE=lambda d: pd.to_datetime(d['GAME_DATE']))
            .set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        )
        self.rotation_features = (
            pd.read_sql('SELECT * FROM team_rotation_features', self.conn)
        )
        self.build_deployment_datasets()
        self.build_modeling_dataset()
        print('datasets built')
    
    def merge_features(self, team_features=None):
        '''
        merge team_features and rotation_features into a single table - used for both the
        modeling and deployment datasets, so they always have the same feature columns
        '''
        print('merging team_features and rotation_features...')
        if team_features is None:
            team_features = self.team_features
        df = team_features.reset_index().merge(self.rotation_features, left_on=['GAME_ID', 'TEAM_ABBREVIATION'], right_on=['GAME_ID', 'TEAM_ABBREVIATION'], how='left').set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION']).sort_index()
        print('...complete')
        return df

    def shift_observations(self, df):
        '''
        shift features back one game per team so a row's features only reflect
        information available before that row's game (no look-ahead leakage)

        _prev columns are shifted within (team, oppAbv) groups instead, since they're
        already only updated when that particular matchup recurs. _dfm/_arima columns
        are excluded entirely - they're already one-step-ahead forecasts of the row
        they're attached to (see process_data_v2.py), so shifting them again would
        make them stale by two games instead of one.
        '''
        print('shifting features by one game...')
        df = df.sort_index()

        # WL is deliberately NOT excluded: unshifted it is the row's own game result
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv']
        list_of_cols_to_exclude.extend([col for col in df.columns if '_prev' in col])
        list_of_cols_to_exclude.extend([col for col in df.columns if '_dfm' in col])
        list_of_cols_to_exclude.extend([col for col in df.columns if '_arima' in col])

        cols_to_transform = df.columns.difference(list_of_cols_to_exclude)

        df_shift_by_team = (
            df.groupby(['TEAM_ABBREVIATION'], group_keys=False)
            .apply(lambda group: group[cols_to_transform].shift(1))
        )

        df_shift_by_team_opp = df.loc[:, df.columns.str.contains('_prev|oppAbv')]
        df_shift_by_team_opp = (
            df_shift_by_team_opp.groupby(['TEAM_ABBREVIATION', 'oppAbv'], group_keys=False)
            .apply(lambda group: group.shift(1))
        )
        df.update(df_shift_by_team)
        df.update(df_shift_by_team_opp)

        print('...complete')
        return df

    def match_opponent_stats(self, df_feat):
        print('matching opponent stats...')
        df_feat = df_feat.dropna().sort_index()

        # if only one team's row survived the dropna, the flip below would "match" it
        # against itself and hand it its own features as the opponent's
        game_ids = df_feat.index.get_level_values('GAME_ID')
        df_feat = df_feat[game_ids.map(game_ids.value_counts()) == 2]

        # every game now has exactly two rows and sorting keeps them adjacent, so a row's
        # opponent is its neighbour: swap each pair with one vectorized take
        game_ids = df_feat.index.get_level_values('GAME_ID').to_numpy()
        swap = np.arange(len(df_feat)) ^ 1
        assert (game_ids == game_ids[swap]).all(), 'game rows are not adjacent pairs'

        cols_to_flip = [col for col in df_feat.columns if col not in {'WL', 'oppAbv'}]
        opp_df = df_feat[cols_to_flip].iloc[swap].add_suffix('_opp')
        opp_df.index = df_feat.index
        df_feat = pd.concat([df_feat, opp_df], axis=1)

        last_missing_idx = df_feat[df_feat.isnull().any(axis=1)].index.max()
        df_feat_truncated = df_feat.loc[last_missing_idx:] if last_missing_idx is not None else df_feat
        print('...complete')
        return df_feat_truncated.dropna()

    def assign_dataset_split(self, df, n_validation=1000, test_frac=0.2, random_state=100):
        '''
        replicate train_models.py's existing holdout semantics: the most recent
        n_validation rows (chronological) are held out as 'validation' (lookahead out of sample) 
        the remainder gets a random train_test_split into 'train'/'test'
        '''
        df = df.sort_index(level='GAME_ID')
        split = pd.Series('train', index=df.index, name='dataset_split')

        if n_validation >= len(df):
            print(f'only {len(df)} modeling rows available, clamping validation holdout size')
            n_validation = len(df) // 2

        validation_idx = df.index[-n_validation:]
        split.loc[validation_idx] = 'validation'

        remaining_idx = df.index[:-n_validation]
        _, test_idx = train_test_split(remaining_idx, test_size=test_frac, random_state=random_state)
        split.loc[test_idx] = 'test'

        return df.assign(dataset_split=split)

    def build_modeling_dataset(self):
        print('building modeling dataset...')
        df = self.merge_features()
        df = self.shift_observations(df)
        df = self.match_opponent_stats(df)
        df = df.join(self.response, how='inner')
        df = self.assign_dataset_split(df)

        upsert_to_sql(df, self.conn, 'modeling_dataset', pk_cols=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        print('...complete')

    def build_deployment_datasets(self):
        '''
        unshifted, most-recent-game snapshot per team/matchup, for predicting each
        team's next (not yet played) game
        '''
        print('building deployment datasets...')
        # only the latest game per team and the latest meeting per pair of teams are
        # needed, so trim the history before doing anything expensive with it
        df = self.merge_features(self.recent_history(self.team_features))

        df_current = df.groupby(level='TEAM_ABBREVIATION').tail(1).copy()
        df_current = self.forecast_next_game_arima(df_current, df)
        missing_rotation = df_current.loc[:, df_current.columns.str.contains('_top')].isnull().any(axis=1)
        if missing_rotation.any():
            print(f'warning: no rotation features for the latest game of '
                  f'{list(df_current.index.get_level_values("TEAM_ABBREVIATION")[missing_rotation])}')
        # the snapshot is replaced wholesale each run (not upserted), so a change in the
        # feature set can never leave stale columns or stale teams behind
        df_current.reset_index().to_sql('deployment_team_data', self.conn, if_exists='replace', index=False)

        df_prev = df.loc[:, df.columns.str.contains('_prev|oppAbv')]
        df_prev = df_prev.groupby(['TEAM_ABBREVIATION', 'oppAbv']).tail(1)
        n_teams = df_current.index.get_level_values('TEAM_ABBREVIATION').nunique()
        if len(df_prev) < n_teams * (n_teams - 1):
            print(f'warning: only {len(df_prev)} of {n_teams * (n_teams - 1)} team/opponent pairs have a '
                  f'recent meeting - widen recent_history(years=...)')
        upsert_to_sql(df_prev, self.conn, 'deployment_prev_matchup_data', pk_cols=['TEAM_ABBREVIATION', 'oppAbv'])
        print('...complete')

    def recent_history(self, df, years=2):
        '''
        the last `years` years of games. teams meet at least twice a season, so two years
        (conservatively two full seasons) always contains every pair's latest meeting
        '''
        latest = df.index.get_level_values('GAME_DATE').max()
        return df[df.index.get_level_values('GAME_DATE') >= latest - pd.DateOffset(years=years)]

    def forecast_next_game_arima(self, df_current, df_history, endog_cols=ARIMA_ENDOG_COLS, params_path=ARIMA_ORDERS_PATH):
        '''
        the historical *_arima columns in team_features are one-step-ahead forecasts of
        the game they're attached to (already played) - not useful for predicting a
        team's next, unplayed game. replace them here with a genuine forecast one step
        beyond each team's history in df_history (the recent window, not the full history -
        the ARIMA coefficients are re-estimated on it). skipped if no tuned ARIMA orders exist.
        '''
        try:
            with open(params_path, 'rb') as f:
                arima_orders = pickle.load(f)
        except FileNotFoundError:
            print(f'no ARIMA order file found at {params_path}, skipping next-game ARIMA forecast')
            return df_current

        df = df_history.sort_index()
        for col in endog_cols:
            order = arima_orders.get(col)
            feature_col = f'{col}_arima'
            if order is None or feature_col not in df_current.columns:
                continue

            for team in df_current.index.get_level_values('TEAM_ABBREVIATION'):
                series = df.xs(team, level='TEAM_ABBREVIATION')[col].dropna()
                if len(series) < 20:
                    continue
                try:
                    forecast = ARIMA(series.reset_index(drop=True), order=order).fit().forecast(1).iloc[0]
                    df_current.loc[df_current.index.get_level_values('TEAM_ABBREVIATION') == team, feature_col] = forecast
                except Exception as e:
                    print(f'next-game ARIMA forecast failed for {team}/{col}: {e}')

        return df_current


if __name__ == '__main__':
    conn = sqlite3.connect('nba_database_test.db')
    BuildDatasets(conn).run()
