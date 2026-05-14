import pandas as pd
import pytest
from process_data import ComputeFeatures


def make_df(rows, data):
    """Build a minimal multi-index DataFrame for testing."""
    idx = pd.MultiIndex.from_tuples(rows, names=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
    return pd.DataFrame(data, index=idx)


def run_match(df):
    cf = ComputeFeatures.__new__(ComputeFeatures)
    return cf.match_opponent_stats(df.copy())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def two_team_game():
    """One game, two teams."""
    rows = [
        ('2024-01-01', 1001, 'BOS'),
        ('2024-01-01', 1001, 'LAL'),
    ]
    data = {
        'PTS_per48': [110.0, 105.0],
        'AST_per48': [25.0,  22.0],
    }
    return make_df(rows, data)


@pytest.fixture
def multi_game():
    """Three games across three teams."""
    rows = [
        ('2024-01-01', 1001, 'BOS'), ('2024-01-01', 1001, 'LAL'),
        ('2024-01-03', 1002, 'BOS'), ('2024-01-03', 1002, 'MIA'),
        ('2024-01-05', 1003, 'LAL'), ('2024-01-05', 1003, 'MIA'),
    ]
    data = {
        'PTS_per48': [110.0, 105.0, 112.0, 108.0,  99.0, 115.0],
        'AST_per48': [ 25.0,  22.0,  28.0,  20.0,  18.0,  30.0],
    }
    return make_df(rows, data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_opp_values_are_swapped(two_team_game):
    result = run_match(two_team_game)
    bos = result.loc[('2024-01-01', 1001, 'BOS')]
    lal = two_team_game.loc[('2024-01-01', 1001, 'LAL')]
    assert bos['PTS_per48_opp'] == lal['PTS_per48']
    assert bos['AST_per48_opp'] == lal['AST_per48']


def test_swap_is_symmetric(two_team_game):
    """If A gets B's stats, B must also get A's stats."""
    result = run_match(two_team_game)
    bos = two_team_game.loc[('2024-01-01', 1001, 'BOS')]
    lal = result.loc[('2024-01-01', 1001, 'LAL')]
    assert lal['PTS_per48_opp'] == bos['PTS_per48']
    assert lal['AST_per48_opp'] == bos['AST_per48']


def test_opp_cols_created_for_all_columns(two_team_game):
    result = run_match(two_team_game)
    for col in two_team_game.columns:
        assert f'{col}_opp' in result.columns, f'{col}_opp missing'


def test_original_columns_preserved(two_team_game):
    result = run_match(two_team_game)
    for col in two_team_game.columns:
        assert col in result.columns, f'{col} was dropped'


def test_correct_across_multiple_games(multi_game):
    result = run_match(multi_game)
    # BOS vs MIA in game 1002
    bos = result.loc[('2024-01-03', 1002, 'BOS')]
    mia = multi_game.loc[('2024-01-03', 1002, 'MIA')]
    assert bos['PTS_per48_opp'] == mia['PTS_per48']
    # LAL vs MIA in game 1003
    lal = result.loc[('2024-01-05', 1003, 'LAL')]
    mia = multi_game.loc[('2024-01-05', 1003, 'MIA')]
    assert lal['PTS_per48_opp'] == mia['PTS_per48']


def test_no_nulls_in_output(multi_game):
    result = run_match(multi_game)
    assert not result.isnull().any().any(), "Unexpected nulls in output"


def test_wl_and_oppabv_excluded_if_present():
    """WL and oppAbv columns should not produce double-suffixed _opp_opp columns."""
    rows = [
        ('2024-01-01', 1001, 'BOS'),
        ('2024-01-01', 1001, 'LAL'),
    ]
    data = {
        'PTS_per48': [110.0, 105.0],
        'WL':        [1, 0],
        'oppAbv':    ['LAL', 'BOS'],
    }
    df = make_df(rows, data)
    result = run_match(df)
    assert 'WL_opp' not in result.columns
    assert 'oppAbv_opp' not in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
