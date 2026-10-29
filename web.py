from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load saved files
model = pickle.load(open('model.pkl', 'rb'))
feature_columns = pickle.load(open('feature_columns.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Get input values from the form
    school = request.form['school']
    sex = request.form['sex']
    age = int(request.form['age'])
    address = request.form['address']
    famsize = request.form['famsize']
    Pstatus = request.form['Pstatus']

    Medu = int(request.form['Medu'])
    Fedu = int(request.form['Fedu'])
    studytime = int(request.form['studytime'])
    failures = int(request.form['failures'])
    schoolsup = request.form['schoolsup']
    famsup = request.form['famsup']
    paid = request.form['paid']
    activities = request.form['activities']
    nursery = request.form['nursery']
    higher = request.form['higher']
    internet = request.form['internet']
    romantic = request.form['romantic']

    # Create input dataframe
    input_data = pd.DataFrame({
        'school': [school],
        'sex': [sex],
        'age': [age],
        'address': [address],
        'famsize': [famsize],
        'Pstatus': [Pstatus],
        'Medu': [Medu],
        'Fedu': [Fedu],
        'studytime': [studytime],
        'failures': [failures],
        'schoolsup': [schoolsup],
        'famsup': [famsup],
        'paid': [paid],
        'activities': [activities],
        'nursery': [nursery],
        'higher': [higher],
        'internet': [internet],
        'romantic': [romantic]
    })

    # Label encoding binary columns
    binary_cols = [
        'school', 'sex', 'address', 'famsize', 'Pstatus',
        'schoolsup', 'famsup', 'paid', 'activities',
        'nursery', 'higher', 'internet', 'romantic'
    ]

    for col in binary_cols:
        input_data[col] = input_data[col].map({
            'no': 0, 'yes': 1,
            'GP': 0, 'MS': 1,
            'F': 0, 'M': 1,
            'U': 0, 'R': 1,
            'GT3': 0, 'LE3': 1,
            'T': 0, 'A': 1
        })

    # One-hot encoding
    input_data = pd.get_dummies(
        input_data,
        dtype='int',
        drop_first=True
    )

    # Match training columns and their order
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "The student is likely to pass."
    else:
        result = "The student is unlikely to pass."

    return render_template(
        'result.html',
        prediction_text=result,
        age=age,
        school=school,
        sex=sex
    )


if __name__ == '__main__':
   app.run(port=5010, debug=True)