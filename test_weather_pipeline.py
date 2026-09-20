import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression


# TEST DATA INTERGITY: Verify that the local CSV data asset exists and is populated
def test_csv_file_exists():
    filename = 'austin_weather_history.csv'
    # Assert that the file is physically present in the directory
    assert os.path.exists(filename), f"{filename} missing from directory structure!"

    # Assert that the file is not empty
    df = pd.read_csv(filename)
    assert not df.empty, "The saved weather database CSV is completely empty!"


# TEST FEATURE ENGINEERING: Verify that our .shift() logic computes correctly
def test_pandas_lag_features():
    # Construct a mini mock dataframe to test transformation shapes
    mock_data = pd.DataFrame({
        'Hour': ['12 AM', '01 AM', '02 AM', '03 AM'],
        'Temperature (°F)': [70.0, 72.0, 75.0, 73.0],
        'Year': [2026, 2026, 2026, 2026]
    })

    # Mirror the shift transformations from Predict_Weather.py
    mock_data['Temp_Lag_1'] = mock_data['Temperature (°F)'].shift(1)
    mock_data['Temp_Lag_2'] = mock_data['Temperature (°F)'].shift(2)
    mock_data = mock_data.dropna()

    # Assert that dropping NaNs leaves exactly 2 testable rows from our 4 inputs
    assert len(mock_data) == 2, "dropna() dropped too many rows during feature engineering!"

    # Verify the math: For the '02 AM' row, lag_1 should be 01 AM (72.0) and lag_2 should be 12 AM (70.0)
    row_02am = mock_data[mock_data['Hour'] == '02 AM'].iloc[0]
    assert row_02am['Temp_Lag_1'] == 72.0, "Temp_Lag_1 calculation index is incorrect!"
    assert row_02am['Temp_Lag_2'] == 70.0, "Temp_Lag_2 calculation index is incorrect!"


# EST MACHINE LEARNING SYSTEM: Verify the scikit-learn model outputs valid data shapes
def test_model_prediction_outputs():
    # Mock clean arrays for training inputs
    X_train = np.array([[70, 68], [72, 70], [74, 72], [76, 74]])
    y_train = np.array([72, 74, 76, 78])

    # Train a quick test regression instance
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Ask it to predict for a 12-hour sequence matrix shape
    X_test_mock = np.random.rand(12, 2)
    predictions = model.predict(X_test_mock)

    # Assert that the model outputs exactly 12 predictions matching our 12-hour forecast shape
    assert len(predictions) == 12, "Model failed to output a matching 12-hour forecast array shape!"
    # Assert that the output values are valid numeric floats and not broken NaNs
    assert not np.isnan(predictions).any(), "Machine Learning model generated broken NaN calculations!"
