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
# needed here to know which _prev columns get shifted within (team, oppAbv) groups rather
# than within team groups
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

        self.build_deployment_datasets()
        self.build_modeling_dataset()
        print('datasets built')

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

        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv', 'WL']
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
        df_feat = df_feat.dropna().sort_index()

        cols_to_flip = [col for col in df_feat.columns if col not in {'WL', 'oppAbv'}]
        opp_df = (
            df_feat[cols_to_flip]
            .groupby('GAME_ID')
            .transform(lambda x: x[::-1].values)
            .add_suffix('_opp')
        )
        df_feat = pd.concat([df_feat, opp_df], axis=1)

        last_missing_idx = df_feat[df_feat.isnull().any(axis=1)].index.max()
        df_feat_truncated = df_feat.loc[last_missing_idx:] if last_missing_idx is not None else df_feat
        return df_feat_truncated.dropna()

    def assign_dataset_split(self, df, n_validation=1000, test_frac=0.2, random_state=100):
        '''
        replicate train_models.py's existing holdout semantics: the most recent
        n_validation rows (chronological) are held out as 'validation'; the remainder
        gets a random train_test_split into 'train'/'test'
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
        df = self.shift_observations(self.team_features)
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
        df = self.team_features

        df_current = df.groupby(level='TEAM_ABBREVIATION').tail(1).copy()
        df_current = self.forecast_next_game_arima(df_current)
        upsert_to_sql(df_current, self.conn, 'deployment_team_data', pk_cols=['TEAM_ABBREVIATION'])

        df_prev = df.loc[:, df.columns.str.contains('_prev|oppAbv')]
        df_prev = df_prev.groupby(['TEAM_ABBREVIATION', 'oppAbv']).tail(1)
        upsert_to_sql(df_prev, self.conn, 'deployment_prev_matchup_data', pk_cols=['TEAM_ABBREVIATION', 'oppAbv'])
        print('...complete')

    def forecast_next_game_arima(self, df_current, endog_cols=ARIMA_ENDOG_COLS, params_path=ARIMA_ORDERS_PATH):
        '''
        the historical *_arima columns in team_features are one-step-ahead forecasts of
        the game they're attached to (already played) - not useful for predicting a
        team's next, unplayed game. replace them here with a genuine forecast one step
        beyond each team's full history. skipped if no tuned ARIMA orders are available.
        '''
        try:
            with open(params_path, 'rb') as f:
                arima_orders = pickle.load(f)
        except FileNotFoundError:
            print(f'no ARIMA order file found at {params_path}, skipping next-game ARIMA forecast')
            return df_current

        df = self.team_features.sort_index()
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
