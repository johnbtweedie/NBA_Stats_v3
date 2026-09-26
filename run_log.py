'''
training run log: what was fit, on what data, what went wrong, and how it performed.

each run of `RunLog` produces:
  * logs/train_<response>_<run_id>.log - everything printed during the run, plus log
    messages, captured warnings and any traceback, in order
  * a row in the `training_runs` table - settings, dataset summary, environment, status
  * a row per model in the `model_results` table - best params, cv score, fit time, test
    and holdout metrics

review with `recent_runs(conn)` and `run_results(conn)`.
'''
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV

LOGGER_NAME = 'nba.train'
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())

NON_METRIC_KEYS = {'Model', 'Call', 'model', 'params'}


def _json(obj):
    return json.dumps(obj, default=lambda o: float(o) if isinstance(o, np.floating)
                      else int(o) if isinstance(o, np.integer) else str(o))


class _Tee:
    '''copy everything written to stdout into the log file too'''

    def __init__(self, original, stream):
        self.original, self.stream = original, stream

    def write(self, text):
        self.original.write(text)
        self.stream.write(text)
        return len(text)

    def flush(self):
        self.original.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class _CountHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record):
        self.count += 1


class RunLog:
    def __init__(self, conn, response, model_type, settings=None, log_dir='logs'):
        self.conn = conn
        self.response = response
        self.model_type = model_type
        self.settings = settings or {}
        self.log_dir = log_dir
        self.logger = logging.getLogger(LOGGER_NAME)
        self.grid_searches = []
        self._logged_models = set()
        self.dataset_summary = None
        self.artifact = None

    # ---- lifecycle -------------------------------------------------------------------

    def __enter__(self):
        now = datetime.now()
        self.run_id = now.strftime('%Y%m%d_%H%M%S')
        self.started = now.isoformat(timespec='seconds')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'train_{self.response}_{self.run_id}.log')

        self._file_handler = logging.FileHandler(self.log_file)
        self._file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-7s %(message)s'))
        self._console_handler = logging.StreamHandler(sys.__stderr__)
        self._console_handler.setLevel(logging.WARNING)
        self._console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        self._counter = _CountHandler()
        self._loggers = [self.logger, logging.getLogger('py.warnings')]
        for lg in self._loggers:
            lg.addHandler(self._file_handler)
            lg.addHandler(self._counter)
        self.logger.addHandler(self._console_handler)
        self._old_levels = [lg.level for lg in self._loggers]
        for lg in self._loggers:
            lg.setLevel(logging.INFO)

        # train_models.py silences all warnings at import - during a run, log each distinct one instead
        self._old_filters = warnings.filters[:]
        warnings.simplefilter('default')
        logging.captureWarnings(True)

        self._old_stdout = sys.stdout
        sys.stdout = _Tee(sys.stdout, self._file_handler.stream)

        self._patch_grid_search()
        self._t0 = time.time()

        self.logger.info(f'run {self.run_id} started: response={self.response} model_type={self.model_type} '
                         f'settings={_json(self.settings)}')
        self.environment = self._environment()
        self.logger.info(f'environment: {_json(self.environment)}')
        self._write_run(status='running')
        return self

    def __exit__(self, exc_type, exc, tb):
        status, error = 'completed', None
        if exc_type is not None:
            status = 'failed'
            error = ''.join(traceback.format_exception(exc_type, exc, tb))
            self.logger.error(f'run failed with {exc_type.__name__}: {exc}\n{error}')
        elapsed = time.time() - self._t0
        self.logger.info(f'run {status} in {elapsed:.0f}s, {self._counter.count} warnings/errors logged')

        self._write_run(status=status, error=error, finished=datetime.now().isoformat(timespec='seconds'),
                        n_warnings=self._counter.count)

        GridSearchCV.fit = self._orig_fit
        sys.stdout = self._old_stdout
        logging.captureWarnings(False)
        warnings.filters[:] = self._old_filters
        for lg, level in zip(self._loggers, self._old_levels):
            lg.removeHandler(self._file_handler)
            lg.removeHandler(self._counter)
            lg.setLevel(level)
        self.logger.removeHandler(self._console_handler)
        self._file_handler.close()
        return False  # never swallow the exception

    # ---- what gets recorded ----------------------------------------------------------

    def log_dataset(self, features, responses, split_labels, rows_dropped_na=0):
        '''summarize the data going into training and flag anything suspicious'''
        summary = {'n_features': features.shape[1], 'rows_dropped_for_nan': int(rows_dropped_na), 'splits': {}}
        dates = features.index.get_level_values('GAME_DATE') if 'GAME_DATE' in features.index.names else None
        for label in ['train', 'test', 'validation']:
            mask = (split_labels == label).to_numpy()
            entry = {'rows': int(mask.sum())}
            if dates is not None and mask.any():
                entry['first_date'], entry['last_date'] = str(dates[mask].min().date()), str(dates[mask].max().date())
            summary['splits'][label] = entry
            self.logger.info(f'{label}: {entry}')
            if entry['rows'] < 100:
                self.logger.warning(f'{label} split has only {entry["rows"]} rows')

        if rows_dropped_na:
            self.logger.warning(f'{rows_dropped_na} rows dropped because of missing values')

        train = features[(split_labels == 'train').to_numpy()]
        constant = [c for c in features.columns if train[c].nunique(dropna=False) <= 1]
        non_finite = [c for c in features.columns if not np.isfinite(features[c].to_numpy(dtype=float)).all()]
        summary['constant_features'], summary['non_finite_features'] = constant, non_finite
        if constant:
            self.logger.warning(f'{len(constant)} constant features in train (no information): {constant[:10]}')
        if non_finite:
            self.logger.warning(f'{len(non_finite)} features contain inf/NaN: {non_finite[:10]}')

        if responses.nunique() == 2:
            balance = (responses[(split_labels == 'train').to_numpy()].value_counts(normalize=True)
                       .reindex(responses.unique(), fill_value=0.0))   # a class absent from train counts as 0
            summary['train_class_balance'] = {str(k): float(v) for k, v in balance.items()}
            self.logger.info(f'train class balance: {summary["train_class_balance"]}')
            if balance.min() < 0.35:
                self.logger.warning(f'training classes are imbalanced (smallest class is {balance.min():.0%} of train)')
        else:
            summary['train_response_mean_std'] = [float(responses.mean()), float(responses.std())]

        self.logger.info(f'{features.shape[1]} features, {len(features)} rows in total')
        self.dataset_summary = summary
        self._write_run(status='running')

    def log_models(self, models):
        '''record best params, cv score, fit time and test/holdout metrics for each fitted model'''
        used = set()
        for name, entry in models.items():
            if name in self._logged_models or not isinstance(entry, dict):
                continue
            estimator = type(entry.get('best model')).__name__
            grid = next((g for g in reversed(self.grid_searches)
                         if g['estimator'] == estimator and id(g) not in used), {})
            used.add(id(grid))
            test, hold = self._metrics(entry.get('test results')), self._metrics(entry.get('holdout results'))
            self.conn.execute(
                'INSERT INTO model_results (run_id, model_name, estimator, best_params, cv_best_score, cv_scoring, '
                'cv_folds, n_candidates, fit_seconds, test_metrics, holdout_metrics) VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                (self.run_id, name, estimator, _json(grid.get('best_params')), grid.get('best_score'),
                 grid.get('scoring'), grid.get('cv'), grid.get('n_candidates'), grid.get('seconds'),
                 _json(test), _json(hold)))
            self.conn.commit()
            self._logged_models.add(name)
            self.logger.info(f'{name}: best_params={grid.get("best_params")} cv={grid.get("best_score")} '
                             f'fit={grid.get("seconds")}s test={test} holdout={hold}')

    def set_artifact(self, path):
        self.artifact = path
        self.logger.info(f'saved model artifact: {path}')
        self._write_run(status='running')

    # ---- internals -------------------------------------------------------------------

    def _patch_grid_search(self):
        self._orig_fit = GridSearchCV.fit
        recorder = self

        def fit(search, X, y=None, **kwargs):
            start = time.time()
            result = recorder._orig_fit(search, X, y, **kwargs)
            record = {'estimator': type(search.best_estimator_).__name__ if hasattr(search, 'best_estimator_')
                      else type(search.estimator).__name__,
                      'best_params': search.best_params_, 'best_score': float(search.best_score_),
                      'scoring': str(search.scoring), 'cv': str(search.cv if not hasattr(search.cv, 'n_splits')
                                                                 else search.cv.n_splits),
                      'n_candidates': len(search.cv_results_['params']), 'seconds': round(time.time() - start, 1)}
            recorder.grid_searches.append(record)
            recorder.logger.info(f'grid search {record}')
            return result

        GridSearchCV.fit = fit

    @staticmethod
    def _metrics(frame):
        if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
            return None
        return {k: v for k, v in frame.iloc[0].to_dict().items() if k not in NON_METRIC_KEYS}

    def _environment(self):
        info = {}
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            info['git_commit'] = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=here, capture_output=True,
                                                text=True, timeout=5).stdout.strip() or None
            info['git_dirty'] = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=here, capture_output=True,
                                                    text=True, timeout=5).stdout.strip())
        except Exception:
            pass
        import sklearn
        info.update(python=sys.version.split()[0], pandas=pd.__version__, numpy=np.__version__,
                    sklearn=sklearn.__version__)
        return info

    def _write_run(self, status, error=None, finished=None, n_warnings=None):
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS training_runs (
                run_id TEXT PRIMARY KEY, started TEXT, finished TEXT, status TEXT, response TEXT, model_type TEXT,
                settings TEXT, environment TEXT, dataset TEXT, artifact TEXT, log_file TEXT, n_warnings INTEGER,
                error TEXT);
            CREATE TABLE IF NOT EXISTS model_results (
                run_id TEXT, model_name TEXT, estimator TEXT, best_params TEXT, cv_best_score REAL, cv_scoring TEXT,
                cv_folds TEXT, n_candidates INTEGER, fit_seconds REAL, test_metrics TEXT, holdout_metrics TEXT);
        ''')
        self.conn.execute(
            'INSERT OR REPLACE INTO training_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (self.run_id, self.started, finished, status, self.response, self.model_type, _json(self.settings),
             _json(self.environment), _json(self.dataset_summary), self.artifact, self.log_file, n_warnings, error))
        self.conn.commit()


def recent_runs(conn, n=10):
    '''one row per training run, newest first'''
    return pd.read_sql('SELECT run_id, started, status, response, model_type, n_warnings, artifact, log_file, error '
                       'FROM training_runs ORDER BY run_id DESC LIMIT ?', conn, params=(n,))


def run_results(conn, run_id=None):
    '''per-model results for a run (default: the latest), with metrics expanded into columns'''
    if run_id is None:
        run_id = pd.read_sql('SELECT MAX(run_id) AS r FROM training_runs', conn)['r'][0]
    df = pd.read_sql('SELECT * FROM model_results WHERE run_id = ?', conn, params=(run_id,))
    for col in ['test_metrics', 'holdout_metrics']:
        expanded = pd.DataFrame([json.loads(v) if v and v != 'null' else {} for v in df[col]])
        df = pd.concat([df.drop(columns=col), expanded.add_prefix(col.split('_')[0] + '_')], axis=1)
    return df
