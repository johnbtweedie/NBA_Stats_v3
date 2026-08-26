const homeSelect = document.getElementById('home-select');
const awaySelect = document.getElementById('away-select');
const errorMessage = document.getElementById('error-message');

const els = {
    homeLogo: document.getElementById('home-logo'),
    homeName: document.getElementById('home-name'),
    homeWinPct: document.getElementById('home-win-pct'),
    homeDecimalOdds: document.getElementById('home-decimal-odds'),
    homeAmericanOdds: document.getElementById('home-american-odds'),
    awayLogo: document.getElementById('away-logo'),
    awayName: document.getElementById('away-name'),
    awayWinPct: document.getElementById('away-win-pct'),
    awayDecimalOdds: document.getElementById('away-decimal-odds'),
    awayAmericanOdds: document.getElementById('away-american-odds'),
};

async function updatePrediction() {
    const home = homeSelect.value;
    const away = awaySelect.value;
    errorMessage.textContent = '';

    if (home === away) {
        errorMessage.textContent = 'Home and away teams must be different.';
        return;
    }

    const response = await fetch(`/predict?home=${home}&away=${away}`);
    const data = await response.json();

    if (!response.ok) {
        errorMessage.textContent = data.error || 'Unable to fetch prediction.';
        return;
    }

    els.homeLogo.src = data.home_logo;
    els.homeName.textContent = data.home_full_name;
    els.homeWinPct.textContent = `${data.home_win_pct}%`;
    els.homeDecimalOdds.textContent = data.home_decimal_odds;
    els.homeAmericanOdds.textContent = data.home_american_odds;

    els.awayLogo.src = data.away_logo;
    els.awayName.textContent = data.away_full_name;
    els.awayWinPct.textContent = `${data.away_win_pct}%`;
    els.awayDecimalOdds.textContent = data.away_decimal_odds;
    els.awayAmericanOdds.textContent = data.away_american_odds;
}

homeSelect.addEventListener('change', updatePrediction);
awaySelect.addEventListener('change', updatePrediction);

updatePrediction();
