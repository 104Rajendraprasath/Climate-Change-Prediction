🌍 Climate Change Prediction App

This project is a machine learning & deep learning based web application that predicts key climate parameters and provides a human-readable weather summary.
It uses ANN, LSTM, GRU, and RNN models trained on climate datasets to forecast temperature, CO₂ levels, sea level, precipitation, humidity, and wind speed.

📑 Features

Input climate indicators through a simple web form.

Dynamically select ML/DL model (ANN, LSTM, GRU, RNN).

Predicts future values of climate parameters.

Converts raw predictions into human-readable summaries (e.g. “Hot, Light Rain, Breezy”).

Interactive frontend built with HTML, CSS, JavaScript.

Backend powered by Python Flask and TensorFlow/Keras models.

🛠️ Tools and Technologies Used

Programming Language: Python

Backend Framework: Flask

Frontend: HTML, CSS, JavaScript (Fetch API)

Data Handling & Analysis: Pandas, NumPy

Visualization: Matplotlib, Seaborn, Plotly

Machine Learning / Deep Learning: Scikit-learn, TensorFlow / Keras

Models Implemented: ANN, LSTM, GRU, RNN

Data Sources: NASA GISTEMP, NOAA Climate Data, Kaggle datasets

Preprocessing: Missing value handling, normalization, time-series formatting

Evaluation Metrics: RMSE, MAE

Deployment: Localhost (can be extended to Heroku/AWS/Azure)

📂 Project Structure
├── app.py                  # Flask backend
├── templates/
│   └── index.html          # Frontend HTML template
├── static/
│   └── style.css           # Custom CSS styles
├── models/
│   ├── ann_model.h5
│   ├── lstm_model.h5
│   ├── gru_model.h5
│   └── rnn_model.h5
├── climate_change_data.csv # (optional) dataset
└── README.md

🚀 Getting Started
1. Clone the repository
git clone https://github.com/yourusername/ClimateChangePredictionApp.git
cd ClimateChangePredictionApp

2. Install dependencies
pip install flask numpy pandas tensorflow scikit-learn matplotlib seaborn

3. Add your models

Place your trained models (ANN, LSTM, GRU, RNN) in the models directory as .h5 files.

4. Run the Flask app
python app.py


The app runs at http://127.0.0.1:5000/
.
Open this in your browser.

📝 How It Works

User enters climate features (Temperature, CO₂, Sea Level, Precipitation, Humidity, Wind Speed).

User selects which model to use (ANN/LSTM/GRU/RNN).

Backend loads the selected model dynamically and predicts outputs.

Predictions are converted into a readable weather summary.

Frontend displays both raw predictions and the summary.

📊 Model Training (Optional)

Use Jupyter Notebooks to train models on datasets such as NASA GISTEMP or Kaggle Climate datasets.

Preprocess the data (handle missing values, normalize, format for time series).

Split into training and testing sets.

Evaluate models with RMSE or MAE.

Save models as .h5 and put them in the models/ folder.

🖼️ Screenshots

(Include screenshots of your web app showing the form and predictions)

📝 Future Scope

Integrate real-time IoT sensor feeds.

Deploy on cloud platforms like AWS, Azure, or Heroku.

Add a database to store historical predictions.

Implement NLP to generate full weather reports.

🔗 GitHub Link

[Insert your GitHub repo link here]

📜 License

This project is licensed under the MIT License — see the LICENSE file for details.
