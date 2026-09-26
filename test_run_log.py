import json
import logging
import sqlite3
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from run_log import RunLog, recent_runs, run_results


@pytest.fixture
def conn():
    return sqlite3.connect(':memory:')


def frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=n)
    idx = pd.MultiIndex.from_arrays([dates, np.arange(n), ['ATL'] * n],
                                    names=['GAME_DATE', 'GAME_ID', 'TEAM_ABBREVIATION'])
    X = pd.DataFrame({'a': rng.normal(size=n), 'b': rng.normal(size=n)}, index=idx)
    return X, pd.Series((X['a'] > 0).astype(int), index=idx)


def test_successful_run_is_recorded(conn, tmp_path, capsys):
    X, y = frame()
    original_fit, original_stdout, n_filters = GridSearchCV.fit, sys.stdout, len(warnings.filters)

    with RunLog(conn, 'WL', 'classification', {'scale': True}, log_dir=str(tmp_path)) as log:
        print('a printed progress line')
        warnings.warn('something looks off')
        logging.getLogger('nba.train').warning('a logged issue')
        search = GridSearchCV(LogisticRegression(), {'C': [0.1, 1]}, cv=3, scoring='accuracy').fit(X, y)
        log.log_models({'WL_clf_logit': {'best model': search.best_estimator_,
                                         'test results': pd.DataFrame([{'Model': 'm', 'Call': 'c', 'Accuracy': 0.6}]),
                                         'holdout results': None}})
        log.set_artifact('models.pkl')

    # everything is restored afterwards
    assert GridSearchCV.fit is original_fit and sys.stdout is original_stdout
    assert len(warnings.filters) == n_filters

    text = open(log.log_file).read()
    assert 'a printed progress line' in text          # stdout is teed into the log
    assert 'something looks off' in text              # warnings that were silenced are captured
    assert 'a logged issue' in text
    assert 'grid search' in text

    run = recent_runs(conn).iloc[0]
    assert run['status'] == 'completed' and run['artifact'] == 'models.pkl' and run['n_warnings'] >= 2

    result = run_results(conn).iloc[0]
    assert result['model_name'] == 'WL_clf_logit' and result['estimator'] == 'LogisticRegression'
    assert json.loads(result['best_params'])['C'] in (0.1, 1)
    assert result['cv_best_score'] == pytest.approx(search.best_score_)
    assert result['test_Accuracy'] == 0.6


def test_failed_run_is_recorded_and_reraised(conn, tmp_path):
    with pytest.raises(ValueError, match='boom'):
        with RunLog(conn, 'WL', 'classification', log_dir=str(tmp_path)) as log:
            raise ValueError('boom')

    run = recent_runs(conn).iloc[0]
    assert run['status'] == 'failed' and 'ValueError: boom' in run['error']
    assert 'Traceback' in open(log.log_file).read()


def test_dataset_problems_are_flagged(conn, tmp_path):
    X, y = frame()
    X['constant'] = 1.0
    X.iloc[0, 0] = np.inf
    y = pd.Series(np.r_[np.ones(180), np.zeros(20)], index=X.index)       # 90/10 class imbalance
    split = pd.Series(['train'] * 150 + ['test'] * 30 + ['validation'] * 20, index=X.index)

    with RunLog(conn, 'WL', 'classification', log_dir=str(tmp_path)) as log:
        log.log_dataset(X, y, split, rows_dropped_na=7)

    text = open(log.log_file).read()
    assert 'constant features' in text and 'constant' in text
    assert 'inf/NaN' in text
    assert 'imbalanced' in text
    assert '7 rows dropped' in text
    assert 'validation split has only 20 rows' in text

    dataset = json.loads(pd.read_sql('SELECT dataset FROM training_runs', conn)['dataset'][0])
    assert dataset['splits']['train']['rows'] == 150
    assert dataset['constant_features'] == ['constant']


def test_train_model_run_end_to_end(conn, tmp_path, monkeypatch):
    '''the real trainModel logit path, on a small synthetic modeling_dataset'''
    monkeypatch.chdir(tmp_path)
    src = open(__file__.replace('test_run_log.py', 'train_models.py')).read()
    ns = {'__name__': 'train_models_under_test'}
    exec(src[:src.index('WL_models = trainModel(')], ns)     # class definitions only (no training on import)
    class SmallGrid(GridSearchCV):
        '''the real logit grid is ~10,000 fits - keep the test fast, the logging is what is under test'''
        def __init__(self, estimator, param_grid, **kwargs):
            super().__init__(estimator, {'C': [0.1, 1.0]}, **kwargs)
    ns['GridSearchCV'] = SmallGrid
    monkeypatch.setattr(pd, 'read_excel', lambda *a, **k: pd.DataFrame({'feature_cols': ['a', 'b']}))

    rng = np.random.default_rng(1)
    n_games = 400
    a = rng.normal(size=(n_games, 2))
    rows = []
    for g in range(n_games):
        win = int(a[g, 0] + rng.normal() > 0)
        for k, team in enumerate(['ATL', 'BOS']):
            rows.append({'GAME_DATE': str(pd.Timestamp('2024-01-01') + pd.Timedelta(days=g // 5)), 'GAME_ID': g,
                         'TEAM_ABBREVIATION': team, 'a': a[g, 0] * (1 - 2 * k), 'b': a[g, 1],
                         'WL_r': win if k == 0 else 1 - win})
    df = pd.DataFrame(rows)
    df['dataset_split'] = ['train'] * 480 + ['test'] * 160 + ['validation'] * 160
    df.to_sql('modeling_dataset', conn, index=False)

    Model = ns['trainModel']
    def logit_then_save(self, dense_grid):
        self.models = {}
        self.tune_model_logit()
        self.ensemble = None
        self.save_best_models()
    monkeypatch.setattr(Model, 'train_classification_models', logit_then_save)

    Model(conn=conn, scale=True, pca=False)

    run = recent_runs(conn).iloc[0]
    assert run['status'] == 'completed', run['error']
    assert run['artifact'].startswith('best_models_WL_')

    result = run_results(conn).iloc[0]
    assert result['estimator'] == 'LogisticRegression'
    assert result['test_Accuracy'] > 0.5 and result['cv_best_score'] > 0.5
    dataset = json.loads(pd.read_sql('SELECT dataset FROM training_runs', conn)['dataset'][0])
    assert dataset['splits']['validation']['rows'] == 160 and dataset['n_features'] == 2
    assert 'best_params' in open(run['log_file']).read()
