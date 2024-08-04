from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('nn_model.pkl')


# Sample list of teams
teams_list = ["Team A", "Team B", "Team C", "Team D"]  # Replace this with your actual list of teams

@app.route('/')
def index():  # Rename the endpoint function to 'index'
    return render_template('index.html', teams_list=teams_list)

if __name__ == '__main__':
    app.run(debug=True)