from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('hearing_loss_model.pkl')

gender_map = {'Male': 1, 'Female': 0}
age_map = {'<25': 0, '25-34': 1, '35-44': 2, '45-54': 3, '55-64': 4, '65+': 5}
region_map = {'MA': 0}  

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    inputs = {}
    if request.method == 'POST':
        gender = request.form['gender']
        age_group = request.form['age_group']
        region = request.form.get('region', 'MA')
        naics = int(request.form['naics'])

        l3k = float(request.form['l3k'])
        l4k = float(request.form['l4k'])
        l6k = float(request.form['l6k'])
        r3k = float(request.form['r3k'])
        r4k = float(request.form['r4k'])
        r6k = float(request.form['r6k'])

        gender_encoded = gender_map.get(gender, -1)
        age_encoded = age_map.get(age_group, -1)
        region_encoded = region_map.get(region, -1)

        input_data = np.array([[age_encoded, gender_encoded, region_encoded, naics,
                                l3k, l4k, l6k, r3k, r4k, r6k]])

        prediction = model.predict(input_data)[0]
        prediction = 'Hearing Loss' if prediction == 1 else 'No Hearing Loss'

        inputs = {
            'Gender': gender,
            'Age Group': age_group,
            'Region': region,
            'NAICS': naics,
            'L3k': l3k, 'L4k': l4k, 'L6k': l6k,
            'R3k': r3k, 'R4k': r4k, 'R6k': r6k
        }

    return render_template('index.html', prediction=prediction, inputs=inputs)

if __name__ == '__main__':
    app.run(debug=True)
