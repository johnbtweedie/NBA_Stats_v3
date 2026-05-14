import pandas as pd
import numpy as np
import pytest
from process_data import ComputeFeatures


def make_df(rows, data):
    idx = pd.MultiIndex.from_tuples(rows, names=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
    return pd.DataFrame(data, index=idx).sort_index()


def run_prev(df, prev_cols, ngames=2):
    cf = ComputeFeatures.__new__(ComputeFeatures)
    cf.features = df.copy()
    cf.prev_cols = prev_cols
    cf.previous_games_vs_opponent(ngames=ngames)
    return cf.features


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def matchup_df():
    """
    BOS plays LAL in games 1001, 1003, 1005.
    BOS plays MIA in games 1002, 1004.
    LAL plays MIA in game 1006.
    oppAbv already populated (as get_opponent_abv would do).
    """
    rows = [
        ('2024-01-01', 1001, 'BOS'), ('2024-01-01', 1001, 'LAL'),
        ('2024-01-03', 1002, 'BOS'), ('2024-01-03', 1002, 'MIA'),
        ('2024-01-05', 1003, 'BOS'), ('2024-01-05', 1003, 'LAL'),
        ('2024-01-07', 1004, 'BOS'), ('2024-01-07', 1004, 'MIA'),
        ('2024-01-09', 1005, 'BOS'), ('2024-01-09', 1005, 'LAL'),
        ('2024-01-11', 1006, 'LAL'), ('2024-01-11', 1006, 'MIA'),
    ]
    data = {
        'PTS_per48': [100, 90, 105, 80, 110, 85, 115, 88, 120, 95, 92, 75],
        'oppAbv':    ['LAL', 'BOS', 'MIA', 'BOS', 'LAL', 'BOS', 'MIA', 'BOS', 'LAL', 'BOS', 'MIA', 'LAL'],
    }
    return make_df(rows, data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_prev_cols_created(matchup_df):
    result = run_prev(matchup_df, prev_cols=['PTS_per48'])
    assert 'PTS_per48_prev' in result.columns


def test_original_cols_preserved(matchup_df):
    result = run_prev(matchup_df, prev_cols=['PTS_per48'])
    assert 'PTS_per48' in result.columns
    assert 'oppAbv' in result.columns


def test_rolling_is_head_to_head_only(matchup_df):
    """BOS vs LAL rolling should not include BOS vs MIA games."""
    result = run_prev(matchup_df, prev_cols=['PTS_per48'], ngames=2)

    # BOS in game 1003 (vs LAL, 2nd BOS-LAL game):
    #   BOS-LAL history: game 1001 (100), game 1003 (110)
    #   rolling(2).mean() at game 1003 = (100 + 110) / 2 = 105.0
    bos_1003 = result.loc[('2024-01-05', 1003, 'BOS'), 'PTS_per48_prev']
    assert bos_1003 == pytest.approx(105.0), f"Expected 105.0, got {bos_1003}"


def test_first_game_is_nan(matchup_df):
    """First head-to-head game has no prior history — should be NaN."""
    result = run_prev(matchup_df, prev_cols=['PTS_per48'], ngames=2)
    bos_1001 = result.loc[('2024-01-01', 1001, 'BOS'), 'PTS_per48_prev']
    assert pd.isna(bos_1001)


def test_rolling_advances_correctly(matchup_df):
    """Third BOS-LAL game rolling should use games 1003 and 1005."""
    result = run_prev(matchup_df, prev_cols=['PTS_per48'], ngames=2)

    # BOS in game 1005 (vs LAL, 3rd BOS-LAL game):
    #   rolling(2) over games 1003 (110) and 1005 (120) = 115.0
    bos_1005 = result.loc[('2024-01-09', 1005, 'BOS'), 'PTS_per48_prev']
    assert bos_1005 == pytest.approx(115.0), f"Expected 115.0, got {bos_1005}"


def test_independent_matchup_unaffected(matchup_df):
    """BOS vs MIA rolling should be independent of BOS vs LAL games."""
    result = run_prev(matchup_df, prev_cols=['PTS_per48'], ngames=2)

    # BOS in game 1004 (vs MIA, 2nd BOS-MIA game):
    #   BOS-MIA history: game 1002 (105), game 1004 (115)
    #   rolling(2).mean() = (105 + 115) / 2 = 110.0
    bos_1004 = result.loc[('2024-01-07', 1004, 'BOS'), 'PTS_per48_prev']
    assert bos_1004 == pytest.approx(110.0), f"Expected 110.0, got {bos_1004}"


def test_multiple_prev_cols(matchup_df):
    """Works with more than one column in prev_cols."""
    matchup_df['AST_per48'] = [20, 18, 22, 16, 24, 19, 26, 17, 28, 21, 23, 15]
    result = run_prev(matchup_df, prev_cols=['PTS_per48', 'AST_per48'], ngames=2)
    assert 'PTS_per48_prev' in result.columns
    assert 'AST_per48_prev' in result.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
