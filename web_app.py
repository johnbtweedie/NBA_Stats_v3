from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('nn_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        # Process team names, perform any necessary preprocessing
        # Make predictions using your sklearn model
        prediction = model.predict([[team1_feature1, team1_feature2, ..., team2_feature1, team2_feature2, ...]])
        # Display prediction result
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)