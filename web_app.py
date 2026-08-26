from flask import Flask, jsonify, render_template, request, send_from_directory
import joblib
import os

app = Flask(__name__)

LOGO_DIR = os.path.join(app.root_path, 'catalogs', 'logos')

TEAM_NAMES = {
    'ATL': 'Atlanta Hawks',
    'BKN': 'Brooklyn Nets',
    'BOS': 'Boston Celtics',
    'CHA': 'Charlotte Hornets',
    'CHI': 'Chicago Bulls',
    'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets',
    'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors',
    'HOU': 'Houston Rockets',
    'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers',
    'LAL': 'Los Angeles Lakers',
    'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks',
    'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans',
    'NYK': 'New York Knicks',
    'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic',
    'PHI': 'Philadelphia 76ers',
    'PHX': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings',
    'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors',
    'UTA': 'Utah Jazz',
    'WAS': 'Washington Wizards',
}

predictions = joblib.load('data/WL_predictions.pkl')
teams_list = sorted(TEAM_NAMES.items(), key=lambda item: item[1])


def calculate_decimal_odds(win_prob):
    return round(1 / win_prob, 2)


def calculate_american_odds(win_prob):
    if win_prob >= 0.5:
        return str(round(-100 * (win_prob / (1 - win_prob))))
    else:
        return '+' + str(round(100 * ((1 - win_prob) / win_prob)))


@app.route('/')
def index():
    return render_template('index.html', teams_list=teams_list)


@app.route('/predict')
def predict():
    home = request.args.get('home', '').upper()
    away = request.args.get('away', '').upper()

    if home not in TEAM_NAMES or away not in TEAM_NAMES:
        return jsonify({'error': 'unknown team abbreviation'}), 400
    if home == away:
        return jsonify({'error': 'home and away teams must be different'}), 400

    row = predictions[(predictions['home'] == home) & (predictions['away'] == away)]
    if row.empty:
        return jsonify({'error': 'no prediction available for this matchup'}), 404
    row = row.iloc[0]

    home_win_prob = row['home_ensemble']
    away_win_prob = row['away_ensemble']

    return jsonify({
        'home': home,
        'away': away,
        'home_full_name': TEAM_NAMES[home],
        'away_full_name': TEAM_NAMES[away],
        'home_win_pct': round(home_win_prob * 100, 2),
        'away_win_pct': round(away_win_prob * 100, 2),
        'home_decimal_odds': calculate_decimal_odds(home_win_prob),
        'away_decimal_odds': calculate_decimal_odds(away_win_prob),
        'home_american_odds': calculate_american_odds(home_win_prob),
        'away_american_odds': calculate_american_odds(away_win_prob),
        'home_logo': f'/logo/{home}',
        'away_logo': f'/logo/{away}',
    })


@app.route('/logo/<abbreviation>')
def logo(abbreviation):
    return send_from_directory(LOGO_DIR, f'{abbreviation.upper()}.png')


if __name__ == '__main__':
    app.run(debug=True, port=5001)
