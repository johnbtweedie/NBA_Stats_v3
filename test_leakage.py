'''
look-ahead leakage tests for the team feature -> modeling dataset pipeline.

the core test is a perturbation test: change the box score / outcome of ONE game G and
re-run the whole pipeline. every training feature for any row played on or before G's date
(including G's own rows) must be bit-for-bit unchanged, because those features are
supposed to describe a team *before tip-off*. the only things allowed to change for G's
rows are the response columns (*_r) and the schedule-known columns (Home, DaysRest, ...).

everything runs on synthetic gamelogs in a temp directory (so no real catalogs/parameters
files are picked up and results are deterministic).
'''
import os
import pickle
import sqlite3

import numpy as np
import pandas as pd
import pytest

from build_datasets import BuildDatasets
from process_data_v2 import ARIMA_ORDERS_PATH, ComputePlayerFeatures, ComputeTeamFeatures

TEAMS = ['ATL', 'BOS', 'BKN', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND']
GAMES_PER_TEAM = 220
KEY = ['GAME_ID', 'TEAM_ABBREVIATION']

# columns that are legitimately known before tip-off (schedule) or are the response itself
SCHEDULE_COLS = {'Home', 'roadtrip', 'DaysRest', 'DaysElapsed'}


def make_team_gamelogs(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    remaining = {t: GAMES_PER_TEAM for t in TEAMS}
    game_id = 22400001
    date = pd.Timestamp('2023-10-24')
    played_today = set()
    while sum(v > 0 for v in remaining.values()) >= 2:
        available = [t for t, v in remaining.items() if v > 0 and t not in played_today]
        if len(available) < 2:  # like the real schedule: several games a day, but a team plays once
            date += pd.Timedelta(days=1)
            played_today = set()
            continue
        home, away = rng.choice(available, size=2, replace=False)
        margin = int(rng.integers(-25, 25)) or 1
        for team, opp, is_home, plus_minus in [(home, away, True, margin), (away, home, False, -margin)]:
            fgm = int(rng.integers(30, 50))
            fga = fgm + int(rng.integers(20, 40))
            fg3m = int(rng.integers(5, 18))
            fg3a = fg3m + int(rng.integers(5, 15))
            ftm = int(rng.integers(5, 25))
            fta = ftm + int(rng.integers(0, 8))
            oreb, dreb = int(rng.integers(5, 15)), int(rng.integers(20, 40))
            rows.append({
                'GAME_DATE': date.strftime('%Y-%m-%d'), 'GAME_ID': game_id, 'TEAM_ABBREVIATION': team,
                'MATCHUP': f'{team} {"vs." if is_home else "@"} {opp}',
                'WL': 'W' if plus_minus > 0 else 'L', 'MIN': 240 + int(rng.integers(-5, 5)),
                'FGM': fgm, 'FGA': fga, 'FG_PCT': fgm / fga, 'FG3M': fg3m, 'FG3A': fg3a, 'FG3_PCT': fg3m / fg3a,
                'FTM': ftm, 'FTA': fta, 'FT_PCT': ftm / fta, 'OREB': oreb, 'DREB': dreb, 'REB': oreb + dreb,
                'AST': int(rng.integers(15, 30)), 'TOV': int(rng.integers(8, 20)),
                'STL': int(rng.integers(4, 12)), 'BLK': int(rng.integers(2, 8)), 'BLKA': int(rng.integers(2, 8)),
                'PF': int(rng.integers(10, 22)), 'PFD': int(rng.integers(10, 22)),
                'PTS': 2 * (fgm - fg3m) + 3 * fg3m + ftm, 'PLUS_MINUS': plus_minus,
            })
        remaining[home] -= 1
        remaining[away] -= 1
        played_today.update([home, away])
        game_id += 1
    return pd.DataFrame(rows)


def make_player_gamelogs(team_gamelogs, seed=1):
    '''8 players per team per game; the top 5 always play, the bench sometimes sits'''
    rng = np.random.default_rng(seed)
    rows = []
    for game_id, team in zip(team_gamelogs['GAME_ID'], team_gamelogs['TEAM_ABBREVIATION']):
        team_id = 1610612000 + TEAMS.index(team)
        for p in range(8):
            played = p < 5 or rng.random() > 0.15
            minutes = int(36 - 4 * p + rng.integers(-3, 4))
            rows.append({
                'gameId': f'00{game_id}', 'teamId': team_id, 'teamTricode': team, 'personId': team_id * 100 + p,
                'comment': '' if played else 'DNP', 'minutes': f'{minutes}:00' if played else '',
                'usagePercentage': rng.uniform(0.1, 0.35) if played else 0.0,
                'offensiveRating': rng.uniform(90, 120), 'defensiveRating': rng.uniform(90, 120),
            })
    return pd.DataFrame(rows)


def perturb_players(players, game_id, team):
    '''bench player has a monster game (jumps into the rotation), starter sits out'''
    df = players.copy()
    in_game = (df['gameId'] == f'00{game_id}') & (df['teamTricode'] == team)
    team_id = 1610612000 + TEAMS.index(team)
    bench = in_game & (df['personId'] == team_id * 100 + 7)
    df.loc[bench, ['usagePercentage', 'offensiveRating', 'comment', 'minutes']] = [0.99, 300.0, '', '25:00']
    starter = in_game & (df['personId'] == team_id * 100)
    df.loc[starter, ['usagePercentage', 'comment', 'minutes']] = [0.0, 'DNP', '']
    return df


def perturb_game(gamelogs, game_id, team):
    '''make one team's performance in one game wildly different, and flip its result'''
    df = gamelogs.copy()
    m = (df['GAME_ID'] == game_id) & (df['TEAM_ABBREVIATION'] == team)
    for col in ['PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF']:
        df.loc[m, col] = df.loc[m, col] * 3 + 25
    df.loc[m, 'FG_PCT'] = 0.99
    df.loc[m, 'FG3_PCT'] = 0.99
    df.loc[m, 'FT_PCT'] = 0.99
    df.loc[m, 'PLUS_MINUS'] = -df.loc[m, 'PLUS_MINUS']
    df.loc[m, 'WL'] = df.loc[m, 'WL'].map({'W': 'L', 'L': 'W'})
    return df


def run_pipeline(gamelogs, players, workdir, arima_orders=None):
    '''player + team features -> datasets, against a fresh sqlite file; returns tables as dataframes'''
    workdir = str(workdir)
    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        if arima_orders is not None:
            os.makedirs(os.path.dirname(ARIMA_ORDERS_PATH), exist_ok=True)
            with open(ARIMA_ORDERS_PATH, 'wb') as f:
                pickle.dump(arima_orders, f)

        conn = sqlite3.connect('test.db')
        gamelogs.to_sql('team_gamelogs', conn, index=False)
        players.to_sql('player_gamelogs', conn, index=False)
        ComputePlayerFeatures(conn=conn, refresh=True).run()
        ComputeTeamFeatures(conn=conn, refresh=True).run()
        BuildDatasets(conn).run()
        tables = {t: pd.read_sql(f'SELECT * FROM {t}', conn) for t in
                  ['team_features', 'response_table', 'modeling_dataset', 'deployment_team_data',
                   'team_rotation_features']}
        conn.close()
        return tables
    finally:
        os.chdir(old_cwd)


def leaking_columns(base, perturbed, cutoff_date):
    '''training feature columns that changed for rows on/before cutoff_date after the perturbation'''
    b = base['modeling_dataset'].set_index(KEY)
    p = perturbed['modeling_dataset'].set_index(KEY)
    common = b.index.intersection(p.index)
    b, p = b.loc[common], p.loc[common]
    early = pd.to_datetime(b['GAME_DATE']) <= cutoff_date
    assert early.sum() > 0, 'no modeling rows on/before the perturbed game - test would be vacuous'

    def is_feature(col):
        stem = col[:-4] if col.endswith('_opp') else col
        return stem not in SCHEDULE_COLS and not stem.endswith('_r')

    cols = [c for c in b.columns
            if is_feature(c) and pd.api.types.is_numeric_dtype(b[c]) and c not in ('GAME_DATE',)]
    changed = ~np.isclose(b.loc[early, cols].to_numpy(dtype=float),
                          p.loc[early, cols].to_numpy(dtype=float), equal_nan=True)
    return [c for c, bad in zip(cols, changed.any(axis=0)) if bad]


@pytest.fixture(scope='module')
def gamelogs():
    return make_team_gamelogs()


@pytest.fixture(scope='module')
def players(gamelogs):
    return make_player_gamelogs(gamelogs)


@pytest.fixture(scope='module')
def baseline(gamelogs, players, tmp_path_factory):
    return run_pipeline(gamelogs, players, tmp_path_factory.mktemp('baseline'))


@pytest.fixture(scope='module')
def perturbed_game(baseline):
    '''(game_id, team, date) of a row halfway through the modeling dataset, so there are
    modeling rows both before and after it'''
    m = baseline['modeling_dataset'].sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    row = m.iloc[len(m) // 2]
    return int(row['GAME_ID']), row['TEAM_ABBREVIATION'], pd.Timestamp(row['GAME_DATE'])


@pytest.fixture(scope='module')
def perturbed(gamelogs, players, perturbed_game, tmp_path_factory):
    game_id, team, _ = perturbed_game
    return run_pipeline(perturb_game(gamelogs, game_id, team), perturb_players(players, game_id, team),
                        tmp_path_factory.mktemp('perturbed'))


# ---------------------------------------------------------------------------
# training features: nothing from game G (or later) may reach a row on/before G
# ---------------------------------------------------------------------------

def test_no_lookahead_leakage_into_training_features(baseline, perturbed, perturbed_game):
    _, _, game_date = perturbed_game
    leaks = leaking_columns(baseline, perturbed, game_date)
    assert not leaks, f'{len(leaks)} training feature columns changed when a later/same game changed: {leaks[:15]}'


def test_perturbation_is_detectable(baseline, perturbed, perturbed_game):
    '''guards against a vacuous test: the perturbed team's NEXT game must see the change'''
    game_id, team, game_date = perturbed_game
    b = baseline['modeling_dataset'].set_index(KEY)
    p = perturbed['modeling_dataset'].set_index(KEY)
    later = [i for i in b.index.intersection(p.index)
             if i[1] == team and pd.to_datetime(b.loc[i, 'GAME_DATE']) > game_date]
    assert later, 'perturbed team has no later modeling rows'
    assert any(not np.isclose(b.loc[i, 'PTS_per48_ra8'], p.loc[i, 'PTS_per48_ra8']) for i in later)
    rotation_cols = [c for c in b.columns if '_top' in c and not c.endswith('_opp')]
    assert rotation_cols, 'rotation features were not merged into the modeling dataset'
    assert any(not np.allclose(b.loc[i, rotation_cols].astype(float), p.loc[i, rotation_cols].astype(float))
               for i in later)


def test_training_features_are_previous_game_values(baseline):
    '''a row's rolling/z-scored features equal the team's unshifted values at its PREVIOUS game'''
    tf = baseline['team_features'].copy()
    tf['GAME_DATE'] = pd.to_datetime(tf['GAME_DATE'])
    tf = tf.sort_values(['GAME_DATE', 'GAME_ID'])
    cols = ['PTS_per48_ra8', 'OffRat_ra32', 'eFG%_ra16', 'DefRat_z']

    prev = tf.groupby('TEAM_ABBREVIATION')[cols].shift(1)
    prev[KEY] = tf[KEY]
    prev = prev.set_index(KEY)

    m = baseline['modeling_dataset']
    own = prev.loc[list(zip(m['GAME_ID'], m['TEAM_ABBREVIATION']))]
    opp = prev.loc[list(zip(m['GAME_ID'], m['oppAbv']))]
    for col in cols:
        np.testing.assert_allclose(m[col].to_numpy(), own[col].to_numpy(), err_msg=f'{col}')
        np.testing.assert_allclose(m[f'{col}_opp'].to_numpy(), opp[col].to_numpy(), err_msg=f'{col}_opp')


# ---------------------------------------------------------------------------
# responses: outcomes must describe the row's own game
# ---------------------------------------------------------------------------

def test_response_aligned_with_row_game(baseline, gamelogs):
    raw = gamelogs.set_index(KEY)
    for table in ['modeling_dataset', 'response_table']:
        df = baseline[table].set_index(KEY)
        r = raw.loc[df.index]
        np.testing.assert_array_equal(df['WL_r'].to_numpy(), (r['WL'] == 'W').astype(int).to_numpy(), err_msg=table)
        np.testing.assert_allclose(df['PTS_per48_r'].to_numpy(), (r['PTS'] / r['MIN'] * 48).to_numpy(), atol=0.01,
                                   err_msg=table)


def test_each_game_has_exactly_one_winner(baseline):
    per_game = baseline['modeling_dataset'].groupby('GAME_ID')['WL_r'].agg(['count', 'sum'])
    assert (per_game['count'] == 2).all()
    assert (per_game['sum'] == 1).all()


def test_responses_are_not_shifted_like_features(baseline, perturbed, perturbed_game):
    '''the perturbed game's own row must reflect the flipped result in the response'''
    game_id, team, _ = perturbed_game
    b = baseline['modeling_dataset'].set_index(KEY)
    p = perturbed['modeling_dataset'].set_index(KEY)
    assert b.loc[(game_id, team), 'WL_r'] != p.loc[(game_id, team), 'WL_r']


# ---------------------------------------------------------------------------
# deployment: must include the most recent game (deliberately NOT shifted)
# ---------------------------------------------------------------------------

def test_deployment_snapshot_includes_latest_game(baseline):
    tf = baseline['team_features'].copy()
    tf['GAME_DATE'] = pd.to_datetime(tf['GAME_DATE'])
    latest = tf.sort_values(['GAME_DATE', 'GAME_ID']).groupby('TEAM_ABBREVIATION').tail(1).set_index('TEAM_ABBREVIATION')
    dep = baseline['deployment_team_data'].set_index('TEAM_ABBREVIATION')
    assert set(dep.index) == set(latest.index)
    np.testing.assert_array_equal(dep.loc[latest.index, 'GAME_ID'], latest['GAME_ID'])
    np.testing.assert_allclose(dep.loc[latest.index, 'PTS_per48_ra8'], latest['PTS_per48_ra8'])


def test_deployment_has_every_modeling_feature(baseline):
    '''models are trained on modeling_dataset, so deployment must offer the same feature columns
    (the _opp columns are built from the opponent's deployment row at prediction time)'''
    modeling = baseline['modeling_dataset']
    expected = {c for c in modeling.columns
                if not c.endswith('_opp') and not c.endswith('_r') and c != 'dataset_split'}
    missing = expected - set(baseline['deployment_team_data'].columns)
    assert not missing, f'deployment is missing {len(missing)} modeling features: {sorted(missing)[:10]}'
    assert any('_top' in c for c in expected), 'no rotation features in the modeling dataset'


def test_deployment_rotation_features_are_from_latest_game(baseline):
    dep = baseline['deployment_team_data'].set_index('TEAM_ABBREVIATION')
    rot = baseline['team_rotation_features'].set_index(KEY)
    cols = [c for c in rot.columns if '_top1' in c or '_top4_agg' in c][:6]
    assert cols
    assert not dep[cols].isnull().any().any()
    for team, row in dep.iterrows():
        np.testing.assert_allclose(row[cols].astype(float).to_numpy(),
                                   rot.loc[(int(row['GAME_ID']), team), cols].astype(float).to_numpy())


# ---------------------------------------------------------------------------
# ARIMA forecast features
# ---------------------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    'compute_arima estimates (p,d,q) coefficients on each team\'s FULL history, so a later game '
    'changes the coefficients and therefore earlier one-step-ahead fitted values (parameter '
    'look-ahead). fix needs an expanding-window / frozen-parameter fit.'))
def test_arima_features_have_no_lookahead(gamelogs, players, perturbed_game, tmp_path_factory):
    game_id, team, game_date = perturbed_game
    orders = {'OffRat': (1, 0, 0), 'DefRat': (1, 0, 0)}
    base = run_pipeline(gamelogs, players, tmp_path_factory.mktemp('arima_base'), arima_orders=orders)
    pert = run_pipeline(perturb_game(gamelogs, game_id, team), perturb_players(players, game_id, team),
                        tmp_path_factory.mktemp('arima_pert'), arima_orders=orders)
    leaks = [c for c in leaking_columns(base, pert, game_date) if 'arima' in c]
    assert not leaks, f'ARIMA features leak: {leaks}'


def test_recent_history_preserves_deployment_snapshot():
    '''trimming to two years must not change the latest game per team or latest meeting per pair'''
    teams = ['AAA', 'BBB', 'CCC', 'DDD']
    pairs = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:]]
    rows, gid = [], 0
    for month in range(60):  # five years, every pair meets every ~2 months
        for k, (a, b) in enumerate(pairs):
            gid += 1
            date = pd.Timestamp('2019-10-01') + pd.DateOffset(months=month) + pd.Timedelta(days=k)
            rows += [(date, gid, a, b, month), (date, gid, b, a, month)]
    df = pd.DataFrame(rows, columns=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION', 'oppAbv', 'x'])
    df = df.set_index(['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION']).sort_index()

    recent = BuildDatasets(conn=None).recent_history(df)
    assert len(recent) < len(df)
    assert (recent.index.get_level_values('GAME_DATE') >= df.index.get_level_values('GAME_DATE').max()
            - pd.DateOffset(years=2)).all()

    pd.testing.assert_frame_equal(df.groupby(level='TEAM_ABBREVIATION').tail(1),
                                  recent.groupby(level='TEAM_ABBREVIATION').tail(1))
    pd.testing.assert_frame_equal(df.groupby(['TEAM_ABBREVIATION', 'oppAbv']).tail(1).sort_index(),
                                  recent.groupby(['TEAM_ABBREVIATION', 'oppAbv']).tail(1).sort_index())
