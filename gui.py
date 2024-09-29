import tkinter as tk
from PIL import Image, ImageTk  # Pillow library for handling images
import pandas as pd

# Load the DataFrame from the pickle file
df = pd.read_pickle('games_data.pkl')

# Initialize the Tkinter window
root = tk.Tk()
root.title("Game Matchups")
root.configure(bg='light grey')  # Set background color to light grey

# Group the DataFrame by 'GAME_ID'
grouped_games = df.groupby('GAME_ID')


# Function to load and resize team logos
def load_logo(path, size=(100, 100)):
    img = Image.open(path)
    img = img.resize(size, Image.LANCZOS)  # Resize the image to fit the window
    return ImageTk.PhotoImage(img)


# Function to calculate decimal odds from win probability
def calculate_decimal_odds(win_prob):
    return round(1 / win_prob, 2)


# Function to calculate American odds from win probability
def calculate_american_odds(win_prob):
    if win_prob >= 0.5:
        return str(round(-100 * (win_prob / (1 - win_prob))))
    else:
        return str('+' + str(round(100 * ((1 - win_prob) / win_prob))))


# Iterate through each unique GAME_ID
for game_id, group in grouped_games:
    home_row = group[group['Home'] == 'Home'].iloc[0]
    away_row = group[group['Home'] == 'Away'].iloc[0]

    home_team = home_row['Team']
    away_team = away_row['Team']

    # Path to logos
    home_logo_path = 'catalogs/logos/' + home_team + '.png'
    away_logo_path = 'catalogs/logos/' + away_team + '.png'
    at_logo_path = 'catalogs/logos/at.png'  # Path to "at" logo

    # Load the logos
    home_logo = load_logo(home_logo_path)
    away_logo = load_logo(away_logo_path)
    at_logo = load_logo(at_logo_path, size=(50, 50))  # Smaller logo for the middle

    # Create a frame for each game with a thick white border
    frame = tk.Frame(root, bg='light grey', highlightbackground="white", highlightthickness=5)
    frame.pack(padx=10, pady=10, fill='x')

    # Column 1: Away team information
    away_frame = tk.Frame(frame, bg='light grey')
    away_frame.grid(row=0, column=0, padx=20)

    # Away team logo
    away_logo_label = tk.Label(away_frame, image=away_logo, bg='light grey')
    away_logo_label.image = away_logo  # Keep a reference to avoid garbage collection
    away_logo_label.grid(row=0, column=0)

    # Away team name label
    away_label = tk.Label(away_frame, text=f"Away: {away_team}", font=('Helvetica', 12), bg='light grey')
    away_label.grid(row=1, column=0)

    # Away team win probability
    away_win_prob_percent = round(away_row['Win Probability'] * 100, 2)
    away_decimal_odds = calculate_decimal_odds(away_row['Win Probability'])
    away_american_odds = calculate_american_odds(away_row['Win Probability'])

    away_win_prob_value = tk.Label(away_frame, text=f"{away_win_prob_percent}%", font=('Helvetica', 10),
                                   bg='light grey')
    away_win_prob_value.grid(row=3, column=0)

    away_decimal_odds_value = tk.Label(away_frame, text=f"{away_decimal_odds}", font=('Helvetica', 10), bg='light grey')
    away_decimal_odds_value.grid(row=4, column=0)

    away_american_odds_value = tk.Label(away_frame, text=f"{away_american_odds}", font=('Helvetica', 10),
                                        bg='light grey')
    away_american_odds_value.grid(row=5, column=0)

    # Column 2: Middle section with "at" logo and row labels
    middle_frame = tk.Frame(frame, bg='light grey')
    middle_frame.grid(row=0, column=1, padx=20)

    # Middle "at" logo
    at_logo_label = tk.Label(middle_frame, image=at_logo, bg='light grey')
    at_logo_label.image = at_logo  # Keep a reference to avoid garbage collection
    at_logo_label.grid(row=0, column=1, pady=10)

    # Blank row (to create space between the logo and the labels)
    blank_label = tk.Label(middle_frame, text=" ", bg='light grey')
    blank_label.grid(row=1, column=1)

    blank_label2 = tk.Label(middle_frame, text=" ", bg='light grey')
    blank_label2.grid(row=2, column=1)

    # Row labels
    win_prob_label = tk.Label(middle_frame, text="Win Probability", font=('Helvetica', 10), bg='light grey')
    win_prob_label.grid(row=3, column=1)

    decimal_odds_label = tk.Label(middle_frame, text="Decimal Odds", font=('Helvetica', 10), bg='light grey')
    decimal_odds_label.grid(row=4, column=1)

    american_odds_label = tk.Label(middle_frame, text="American Odds", font=('Helvetica', 10), bg='light grey')
    american_odds_label.grid(row=5, column=1)

    # Column 3: Home team information
    home_frame = tk.Frame(frame, bg='light grey')
    home_frame.grid(row=0, column=2, padx=20)

    # Home team logo
    home_logo_label = tk.Label(home_frame, image=home_logo, bg='light grey')
    home_logo_label.image = home_logo  # Keep a reference to avoid garbage collection
    home_logo_label.grid(row=0, column=2)

    # Home team name label
    home_label = tk.Label(home_frame, text=f"Home: {home_team}", font=('Helvetica', 12), bg='light grey')
    home_label.grid(row=1, column=2)

    # Home team win probability
    home_win_prob_percent = round(home_row['Win Probability'] * 100, 2)
    home_decimal_odds = calculate_decimal_odds(home_row['Win Probability'])
    home_american_odds = calculate_american_odds(home_row['Win Probability'])

    home_win_prob_value = tk.Label(home_frame, text=f"{home_win_prob_percent}%", font=('Helvetica', 10),
                                   bg='light grey')
    home_win_prob_value.grid(row=3, column=2)

    home_decimal_odds_value = tk.Label(home_frame, text=f"{home_decimal_odds}", font=('Helvetica', 10), bg='light grey')
    home_decimal_odds_value.grid(row=4, column=2)

    home_american_odds_value = tk.Label(home_frame, text=f"{home_american_odds}", font=('Helvetica', 10),
                                        bg='light grey')
    home_american_odds_value.grid(row=5, column=2)

# Start the Tkinter event loop
root.mainloop()
