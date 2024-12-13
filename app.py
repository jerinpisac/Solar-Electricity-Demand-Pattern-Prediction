from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained RandomForestRegressor model
model = joblib.load("solar_electricity_demand_model.pkl")  # Replace with the actual model file name

@app.route('/')
def index():
    return render_template('index.html')  # HTML form for input

@app.route('/home1')
def home1():
    return render_template('home1.html')

@app.route('/home2')
def home2():
    return render_template('home2.html')

@app.route('/predict1', methods=['POST'])
def predict1():
    try:
        # Get user inputs from the form
        date = request.form['date']
        time = request.form['time']
        global_reactive_power = float(request.form['global_reactive_power'])
        voltage = float(request.form['voltage'])
        global_intensity = float(request.form['global_intensity'])
        sub_metering_1 = float(request.form['sub_metering_1'])
        sub_metering_2 = float(request.form['sub_metering_2'])
        sub_metering_3 = float(request.form['sub_metering_3'])

        # Prepare input data
        data = {
            'Date': [pd.to_datetime(date).timestamp()],
            'Time': [pd.to_timedelta(time).total_seconds()],
            'Global_reactive_power': [global_reactive_power],
            'Voltage': [voltage],
            'Global_intensity': [global_intensity],
            'Sub_metering_1': [sub_metering_1],
            'Sub_metering_2': [sub_metering_2],
            'Sub_metering_3': [sub_metering_3]
        }
        input_data = pd.DataFrame(data)

        # Predict using the model
        prediction = model.predict(input_data)[0]

        return render_template('result1.html', prediction=prediction)

    except Exception as e:
        return f"Error: {str(e)}", 400
    
@app.route('/predict2', methods=['POST'])
def predict2():
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']

    # Load the CSV file into a DataFrame
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return f"Error reading file: {str(e)}", 400

    # Ensure the file has the required columns
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert 'Date' to datetime
    data['Date'] = data['Date'].astype('int64') / 10**9  # Convert 'Date' to Unix timestamp (seconds)
    data['Time'] = pd.to_timedelta(data['Time']).dt.total_seconds()
    required_columns = ['Date', 'Time',
                        'Global_reactive_power', 'Voltage', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    if not all(col in data.columns for col in required_columns):
        return f"File is missing required columns: {required_columns}", 400

    # Make predictions
    features = data[required_columns]
    predictions = model.predict(features)

    # Add predictions to the DataFrame
    data['Predicted_GLobal_Active_Power'] = predictions

    # Convert DataFrame to HTML for display
    results_html = data.to_html(classes='table table-striped')

    return render_template('result2.html', table=results_html)

if __name__ == '__main__':
    app.run(debug=True)
