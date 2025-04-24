import pandas as pd
import numpy as np
import sqlite3
import time
from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
import pickle
import statsmodels.api as sm
# import warnings

class ComputeFeatures:
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
        
        # load data from db connection passed
        self.conn = conn
        if refresh:
            self.raw_data = self.load_data()
        else:
            self.raw_data = self.load_required_formatted_data()
            print('data loaded')
        self.refresh = refresh
        
        
        # process the data to create features
        # self.store_response_vars()
        # self.compute_features()
        # self.leaguewide_standardization()
        # self.previous_games_vs_opponent()
        # self.features = self.get_opponent_abv(self.features)
        # self.features.to_csv('regular_features_no_rolling.csv')

        # self.exog_features = self.compute_rolling_avg_exog(self.features)
        # self.exog_features = self.shift_observations(self.exog_features)
        # self.exog_features = self.match_opponent_stats(self.exog_features) # not working for exog features? cehck
        # self.exog_features.to_csv('exog_features.csv')
        
        self.exog_features = pd.read_csv('exog_features.csv') # need to set multi-index*************************************
        self.features = pd.read_csv('regular_features_no_rolling.csv') #debug checkpoint
        self.features = self.compute_rolling_avg_optimal(df=self.features)
        self.features = self.compute_rolling_avg(df=self.features, window=8)
        self.features = self.compute_rolling_avg(df=self.features, window=16)
        self.features = self.compute_rolling_avg(df=self.features, window=32)
        self.features = self.compute_dfm(df=self.features)
        self.save_current_observations()
        self.features = self.shift_observations(self.features)
        self.features = self.match_opponent_stats(self.features)
        self.save_features()

        print('features computed')

    def load_required_formatted_data(self):
        try:
            df_feat_existing = pd.read_sql('SELECT * FROM feature_table', self.conn)
            df_feat_existing = df_feat_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
            # df_feat_existing = df_feat_existing.sort_index(level=['GAME_ID'])

            unique_teams = df_feat_existing.index.get_level_values('TEAM_ABBREVIATION').unique()
            # try:
            #     unique_teams = unique_teams.drop('NOH')

            # get the past 5 games rolling average stats vs the opponent
            print('determining required data for each matchup...')
            required_retrieval_index = []
            cutoff_index_for_saving = []
            for i, team_1 in enumerate(unique_teams):
                # print('Processing', i + 1, 'of', len(unique_teams), 'teams')

                for team_2 in unique_teams[i + 1:]:
                    # print(team_1, 'vs.', team_2)
                    df_team_1 = df_feat_existing.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)
                    df_team_2 = df_feat_existing.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)

                    shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

                    # store 6th most recent games index value
                    required_retrieval_index.append(shared_game_ids[-13]) # we take 13 to ensure enough games are there for each team to populate _prev 6 games rolling avg and all DaysRest correctly
                    cutoff_index_for_saving.append(shared_game_ids[-6]) # we will get rid of this extra game data before resaving to the db, this removes DaysRest diff() artifacts
            print('...complete')
            
            print('loading required data from database...')
            max_required_retrieval_index = pd.to_numeric(
                required_retrieval_index).min()  # index of the least recent game we have to grab data from
            df = pd.read_sql(
                f"SELECT * FROM formatted_data_table WHERE CAST(GAME_ID AS INTEGER) >= {max_required_retrieval_index}",
                self.conn)
            features_exist = True
            print('...complete')
        except:
            print('no existing features detected, computing for full dataset')
            df = pd.read_sql(f"SELECT * FROM formatted_data_table", self.conn)
            features_exist = False

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        df['WL'] = df['WL'].map({'W': 1, 'L': 0})

        self.cutoff_index_for_saving = pd.to_numeric(cutoff_index_for_saving).min()
        return df
    
    def load_data(self):
        '''
        load data from sql db with multi-index and datetime GAME_DATE
        '''

        df = pd.read_sql(f"SELECT * FROM formatted_data_table", self.conn)

        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        df['WL'] = df['WL'].map({'W': 1, 'L': 0})
        self.cutoff_index_for_saving = pd.to_numeric(df.index.get_level_values(1).min())

        return df
    
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

        # clip this to index from final feature set to prevent leakage in training
        print('...complete')

        self.response = df
        self.save_features(db_table_name='response_table')
        
        # return df

    def compute_features(self):
        '''
        compute various features for model training
        offensive rating, defensing rating, assist rate, consecutive road games (roadtrip) etc
        '''
        print('computing features...')

        df_feat = self.raw_data.copy()
        df_feat = df_feat[self.ra_cols]

        # # if enabled, shift by one game so we can produce suitable training data (i.e. data to predict current game is from the game before/pre-current game)
        # if self.shift:
        #     df_feat[self.ra_cols] = df_feat[self.ra_cols].shift(1) 
        # will do this later instead

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

        df_results = pd.DataFrame()
        for i, group in enumerate(df_roadtrip):
            df_group = pd.DataFrame(df_roadtrip.iloc[i],
                                    index=self.raw_data.loc[:,:,df_roadtrip.index[i]].index,
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
            # print(col+'_z')
            df_feat[(col+'_z')] = ((df_feat[col] -
                                    (df_feat[col].transform(lambda x: x.rolling(window=window).mean()))) /
                                df_feat[col].transform(lambda x: x.rolling(window=window).std()))
            
        self.features = df_feat
        # return df_feat

    def previous_games_vs_opponent(self, ngames=6):
        '''
        get the stats from the previous n games of a given matchup
        these stats are not rolling averaged, but some are league-standardized
        '''
        
        # df_feat = self.features[25000:]
        df_feat = self.features
        prev_matchup_cols = self.prev_cols

        suffixed_cols = []
        # initialize empty columns
        for col in prev_matchup_cols:
            prev_col = col + '_prev'
            suffixed_cols.append(prev_col)
            if prev_col not in df_feat.columns:
                df_feat[prev_col] = np.nan

        unique_teams = df_feat.index.get_level_values('TEAM_ABBREVIATION').unique()

        # get the past 5 games rolling average stats vs the opponent
        print('determining previous matchup stats...')
        for i, team_1 in enumerate(unique_teams):
            print('Processing', i + 1, 'of', len(unique_teams), 'teams')

            for team_2 in unique_teams[i + 1:]:
                start_time = time.time()  # Start time of the loop

                print(team_1, 'vs.', team_2)
                # grab data for each team
                df_team_1 = df_feat.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)[prev_matchup_cols]
                df_team_2 = df_feat.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)[prev_matchup_cols]

                # cut to shared games between teams 
                shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))
                df_team_1 = df_team_1.loc[:,shared_game_ids,:]
                df_team_2 = df_team_2.loc[:,shared_game_ids,:]

                # calc rolling avg
                df_team_1_rolling = df_team_1.rolling(ngames).mean().add_suffix('_prev')
                df_team_2_rolling = df_team_2.rolling(ngames).mean().add_suffix('_prev')

                # swap indexes
                # df_team_1_rolling_opp = df_team_2_rolling.set_index(df_team_1.index).add_suffix('_prev')
                # df_team_2_rolling_opp = df_team_1_rolling.set_index(df_team_2.index).add_suffix('_prev')

                # df_team_1_rolling_opp = df_team_1_rolling_opp.add_suffix('_prev')
                # df_team_2_rolling_opp = df_team_2_rolling_opp.add_suffix('_prev')

                df_feat.update(df_team_1_rolling.loc[:,shared_game_ids,team_1])
                df_feat.update(df_team_2_rolling.loc[:,shared_game_ids,team_2])

                # End time of the loop and print the time taken
                elapsed_time = time.time() - start_time
                print(f"Time taken for {team_1} vs {team_2}: {elapsed_time:.4f} seconds")
        
        self.features = df_feat
        # return df_feat
    
    def tune_rolling_avg(self):
        df = self.features
        df = df.sort_index()
        print('tuning rolling average')
        # Define parameter grid
        p_values = [0, 1, 6, 20]
        d_values = [0, 1, 2, 5]
        q_values = [0, 1, 10, 40]
        param_grid = list(product(p_values, d_values, q_values))

        best_params = None
        best_error = float('inf')

        # Iterate over all (p,d,q) combinations
        for p, d, q in param_grid:
            total_error = 0
            num_teams = 0

            for team, group in df.groupby('TEAM_ABBREVIATION', group_keys=False):
                try:
                    # Convert index to datetime if necessary
                    group = group.copy()
                    if not isinstance(group.index, pd.DatetimeIndex):
                        group.index = pd.to_datetime(group.index, errors='coerce')  # Ensure datetime format
                    
                    model = ARIMA(group['DefRat'], order=(p, d, q)).fit()
                    forecast = model.forecast()
                    residual_error = np.abs(group['DefRat'].iloc[-1] - forecast.iloc[0])
                    total_error += residual_error
                    num_teams += 1
                except Exception as e:
                    continue  # Skip if model fails

            avg_error = total_error / num_teams if num_teams > 0 else float('inf')

            if avg_error < best_error:
                best_error = avg_error
                best_params = (p, d, q)

        print(f"Best ARIMA parameters: {best_params} with average residual error: {best_error}")
        # return best_params
    
    def compute_rolling_avg_exog(self, df, window=41):
        '''
        compute [window]-game rolling average for all teams for exogenous features
        this is passed to the rolling average tuning script
        '''
        df = df.sort_index()
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv']
        
        # exogenous features rolling average
        # define the exogenous (opponent) features for the time series model below
        cols_to_transform = ['OER', 'DER', 'eFG%', 'eFG%_against', 'TS%', 'TS%_against',	
                                'DaysElapsed', 'DaysRest', 'WL', 'Poss', 'OffRat', 'DefRat',	
                                'OREB%', 'DREB%', 'TOV%', 'TOV_forced%', 'STL%', 'AST%', 'BB%',
                                'eFG%_z', 'eFG%_against_z', 'TS%_z', 'TS%_against_z',	
                                'Poss_z', 'OffRat_z', 'DefRat_z',	#'Home', 'roadtrip',
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
        '''
        df = df.sort_index()
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv']
        list_of_cols_to_exclude.extend([col for col in df.columns if '_prev' in col])
        cols_to_transform = df.columns.difference(list_of_cols_to_exclude)


        # compute optimal window rolling averages
        optimal_windows = pd.read_csv(r'catalogs/parameters/rolling_average_windows.csv', index_col=0)
        for feature, window in optimal_windows.iterrows():
            window = int(window.values[0])
            print(feature, window)
            
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

        # # Keep the excluded columns intact
        # if 'oppAbv' in df.columns:
        #     df_excluded = df[list_of_cols_to_exclude]
        # else:
        #     list_of_cols_to_exclude.remove('oppAbv')
        #     df_excluded = df[list_of_cols_to_exclude]
        
        # Combine the transformed and excluded columns
        return pd.concat([df, df_transformed], axis=1)
    
    def compute_dfm(self, df):
        '''
        apply the fitted DFM time series models to relevant feature columns
        '''
        df = df.copy().set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        # df = df.sort_index()
        df_exog = self.exog_features.copy().set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

        dfm_1_path = 'dfm_1.pkl'

        # Open and load the pickle file
        with open(dfm_1_path, 'rb') as file:
            dfm_1 = pickle.load(file)
        endog_1 = dfm_1['ATL']['endog']
        exog_1 = dfm_1['ATL']['exog']
        exog_ra8 = [x for x in exog_1 if '_ra8' in x]
        exog_ra32 = [x for x in exog_1 if '_ra32' in x]
        exog_1 = [x for x in exog_1 if not '_ra' in x]

        # unique_teams = df['TEAM_ABBREVIATION'].unique()
        for team, group in df.groupby('TEAM_ABBREVIATION', group_keys=False):
            if team != 'NOH': #in ['ATL', 'OKC', 'MIL']:

                print(team)
                endog_series = group[endog_1].apply(pd.to_numeric, errors='coerce').dropna().sort_index()  
                exog_series = df_exog.xs(team, level='TEAM_ABBREVIATION', drop_level=False)[exog_1].dropna().sort_index() 

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

                k_factors = dfm_1[team]["k_factors"]
                factor_order = dfm_1[team]["factor_order"]
                params = dfm_1[team]['params']

                # Apply the fitted model's parameters to the new data using filtering
                mod = sm.tsa.DynamicFactor(
                    endog=endog_series,
                    exog=exog_series,
                    k_factors=k_factors,
                    factor_order=factor_order
                )

                transformed_result = mod.filter(params)
                filtered_df = transformed_result.fittedvalues.copy()
                filtered_df.index = common_indices  # if needed
                filtered_df = filtered_df.add_suffix('_dfm')

                for col in filtered_df.columns:
                    if col not in df:
                        df[col] = np.nan

                df.update(filtered_df)

                print(team, 'complete')

        return df

    def save_current_observations(self):
        '''
        store the unshifted observations to use in next game predictions
        '''   
        print('storing data for current predictions...')

        # generate column for opponents abbreviation
        df = self.features#.reset_index()
        # df['oppAbv'] = (
        #     df.groupby('GAME_ID')['TEAM_ABBREVIATION']
        #     .transform(lambda x: x[::-1].values)
        # )
        # df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
        # self.features = df
        
        # groupby teams and get their current stats
        # includes their home/away status, roadtrip status, days of rest, and the rolling averages of the stats (including the most recent game played)
        # df_current = self.features.loc[:, ~self.features.columns.str.contains('_prev')] # enable to exclude prev columns
        # df_current = df_current.groupby(level='TEAM_ABBREVIATION').tail(1)
        
        df_current = df.groupby(level='TEAM_ABBREVIATION').tail(1)
        df_current.to_sql('current_team_data', self.conn, if_exists='replace', index=True)

        # get most recent prev stats for all teams and their opponents
        df_prev = df.loc[:,df.columns.str.contains('_prev|oppAbv')]
        df_prev = (
            df_prev.groupby(['TEAM_ABBREVIATION', 'oppAbv']).tail(1)
        )
        df_prev.to_sql('prev_team_data', self.conn, if_exists='replace', index=True)

        # Home and roadtrip are not applicable to current night predictions, we must determine this at that stage
        print('...complete')
    
    def shift_observations(self, df):
        '''
        shift befiore or after opponent matching????
        '''
        print('shifting features by one game...')
        # df = self.features

        # group by teams, shift features by one game
        df_shift_by_team = df.sort_index()
        list_of_cols_to_exclude = ['Home', 'roadtrip', 'DaysRest', 'DaysElapsed', 'oppAbv']
        list_of_cols_to_exclude.extend([col for col in df.columns if '_prev' in col])

        # Columns to apply the transformation
        cols_to_transform = df.columns.difference(list_of_cols_to_exclude)
        
        df_shift_by_team = (
            df_shift_by_team.groupby(['TEAM_ABBREVIATION'], group_keys=False)
            .apply(lambda group: group[cols_to_transform].shift(1))
        )
        
        # group by teams and each unique opponent, shift '_prev' and by 1
        df_shift_by_team_opp = df.loc[:,df.columns.str.contains('_prev|oppAbv')]
        df_shift_by_team_opp = (
            df_shift_by_team_opp.groupby(['TEAM_ABBREVIATION', 'oppAbv'], group_keys=False)
            .apply(lambda group: group.shift(1))
        )
        df.update(df_shift_by_team)
        df.update(df_shift_by_team_opp)

        # self.features = df
        print('...complete')
        return df
    
    def match_opponent_stats(self, df_feat):
        
        # df_feat = self.features[25000:] #debug
        # df_feat = self.features.dropna().sort_index()
        # cols = self.ra_cols.to_list()
        # prev_matchup_cols = self.prev_cols#[col for col in df_feat.columns if '_prev' in col]

        # # remove unecessary data
        # df_feat = df_feat.sort_index()
        # # need to remove excess games here...taking 100 extra in case, to avoid opponent matching artifacts
        # # each team should
        # df_feat = df_feat.loc[(slice(None), slice(self.cutoff_index_for_saving-2460, None), slice(None))]

        df_feat = df_feat.dropna().sort_index()

        cols = df_feat.columns.to_list()
        opp_cols = []

        for col in tqdm(cols, desc="Processing columns"):
            opp_col = col + '_opp'
            if opp_col not in {'WL_opp', 'oppAbv_opp'}:
                opp_cols.append(opp_col)
                df_feat[opp_col] = np.nan
                df_feat[opp_col] = (
                    df_feat[col].groupby('GAME_ID')
                    .transform(lambda x: x[::-1].values)
                )


        # opp_cols = list(set(opp_cols))

        # unique_teams = df_feat.index.get_level_values('TEAM_ABBREVIATION').unique()
        

        # # match opponent stats
        # print('matching opponent stats...')
        # for i, team_1 in enumerate(unique_teams):
        #     print('processing ', i + 1, ' of ', len(unique_teams), ' teams')
        #     for team_2 in unique_teams[i + 1:]:
        #         print(team_1, ' vs. ', team_2)

        #         # get dataframe of each teams features/stats
        #         df_team_1 = df_feat.xs(team_1, level='TEAM_ABBREVIATION', drop_level=False)[cols]
        #         df_team_2 = df_feat.xs(team_2, level='TEAM_ABBREVIATION', drop_level=False)[cols]

        #         # find shared game ids
        #         shared_game_ids = df_team_1.index.get_level_values(1).intersection(df_team_2.index.get_level_values(1))

        #         # truncate each dataframe to the shared game ids
        #         df_team_1 = df_team_1.loc[:,shared_game_ids,:]
        #         df_team_2 = df_team_2.loc[:,shared_game_ids,:]

                
        #         df_team_1_opp = df_team_2.add_suffix('_opp')
        #         df_team_2_opp = df_team_1.add_suffix('_opp')
                
        #         # ------------------------------------------------------------------------
        #         ''' this section needs to be fixed'''
        #         team_1_index = df_team_1_opp.index.intersection(df_feat.index)
        #         team_2_index = df_team_2_opp.index.intersection(df_feat.index) 

        #         df_feat.loc[team_1_index, opp_cols] = df_team_1_opp.loc[team_1_index, opp_cols]
        #         df_feat.loc[team_2_index, opp_cols] = df_team_2_opp.loc[team_2_index, opp_cols]
        #         #########------------------------------------------------#######
        #         print('complete')

        # Find the index of the last row with missing values
        last_missing_idx = df_feat[df_feat.isnull().any(axis=1)].index.max()

        # Truncate the DataFrame up to the last row with missing values
        df_feat_truncated = df_feat.loc[last_missing_idx:] if last_missing_idx is not None else df_feat

        # Optionally, drop the row with missing values if you don't want it in the result
        df_feat_truncated = df_feat_truncated.dropna()

        return df_feat_truncated

    def save_features(self, db_table_name='feature_table'):
        if self.refresh:
            print('no existing data detected, creating database table and saving..')
            if db_table_name == 'feature_table':
                self.features.to_sql(f'{db_table_name}', self.conn, if_exists='replace', index=True)
            elif db_table_name == 'response_table':
                self.response.to_sql(f'{db_table_name}', self.conn, if_exists='replace', index=True)
            else:
                print('database table name not recognized')
        else:
            try:
                df_existing = pd.read_sql(f'SELECT * FROM {db_table_name}', self.conn)
                # df_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])

                df_existing['GAME_ID'] = df_existing['GAME_ID'].astype('int')
                df_existing.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'], inplace=True)

                if 'level_0' in df_existing.columns:
                    df_existing.drop('level_0', axis=1, inplace=True)
                if 'index' in df_existing.columns:
                    df_existing.drop('index', axis=1, inplace=True)

                if db_table_name == 'feature_table':
                    df_existing.update(self.features)
                elif db_table_name == 'response_table':
                    df_existing.update(self.response)
                else:
                    print('database table name not recognized')

                df_existing = df_existing[~df_existing.index.duplicated(keep='last')]
                df_existing = df_existing.sort_index(level=['GAME_ID'])
                # combined_data = combined_data.sort_values(by='GAME_ID')

                print('saving to database...')
                df_existing.to_sql(f'{db_table_name}', self.conn, if_exists='replace', index=True)
                print('..complete')

            except pd.io.sql.DatabaseError:
                print('no existing data detected, creating database table and saving..')
                if db_table_name == 'feature_table':
                    self.features.to_sql(f'{db_table_name}', self.conn, if_exists='replace', index=True)
                elif db_table_name == 'response_table':
                    self.response.to_sql(f'{db_table_name}', self.conn, if_exists='replace', index=True)
                else:
                    print('database table name not recognized')

                print('..complete')
            
all_features = ComputeFeatures(conn=sqlite3.connect('nba_database_2025-01-18.db'), 
                               purpose='predict',
                               refresh=True)

print('complete')