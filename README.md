# 🚲 Bike Sharing Prediction with Gradient Boosting

A machine learning application that predicts bike rental demand using Gradient Boosting regression. This project includes a user-friendly Streamlit interface for making real-time predictions.

## Project Overview

This project addresses the bike-sharing demand prediction problem using advanced machine learning techniques. The model analyzes various factors such as weather, season, temperature, humidity, and wind speed to predict the number of bikes that will be rented at any given hour.

## Model Performance

- **R² Score**: 0.9521 (95.21% variance explained)
- **RMSE**: 38.96 
- **MAE**: 24.65 

These metrics indicate excellent predictive performance with minimal prediction errors.

##  Features Used

- **Temporal Features**: Year, Month, Hour, Day of Week
- **Weather Features**: Temperature, Feels-Like Temperature, Humidity, Wind Speed, Weather Situation
- **Categorical Features**: Season, Holiday, Working Day

## Model Details

- **Algorithm**: Gradient Boosting Regressor
- **Hyperparameters**:
  - Estimators: 300
  - Learning Rate: 0.2
  - Max Depth: 5
- **Data Processing**: StandardScaler for feature normalization

##  Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd PROJET_Regression
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Train the Model
To retrain the model with the latest data:
```bash
python train_model.py
```
This will:
- Load and preprocess the bike-sharing dataset
- Train the Gradient Boosting model
- Save the trained model and scaler as pickle files
- Display performance metrics

#### Launch the Web Application
Start the interactive Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

##  How to Use the Application

1. **Input Features**: Use the form to select or adjust:
   - Season (1-4)
   - Year and Month
   - Hour (0-23)
   - Holiday and Working Day status
   - Weather situation and conditions
   - Temperature, humidity, and wind speed

2. **Click "Predict Bike Count"**: The model will instantly predict the expected number of bike rentals

3. **View Results**: The prediction appears as a success message

## Project Structure

```
PROJET_Regression/
├── app.py                          # Streamlit web application
├── train_model.py                  # Model training script
├── hour.csv                        # Bike-sharing dataset
├── requirements.txt                # Python dependencies
├── gradient_boosting_model.pkl    # Trained model (generated)
├── scaler.pkl                      # Feature scaler (generated)
├── README.md                       # This file
└── assets/
    ├── style.css                   # Custom CSS styling
    └── bike.png                    # Application logo
```

## Dependencies

All dependencies are listed in `requirements.txt`:
- **streamlit**: Web framework for interactive applications
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **joblib**: Model serialization
- **pillow**: Image processing

## Dataset

The project uses the Bike-Sharing-Demand dataset with hourly information including:
- Temporal patterns
- Weather conditions
- Bike rental counts

## Customization

You can easily customize the model by modifying `train_model.py`:
- Adjust hyperparameters (n_estimators, learning_rate, max_depth)
- Add or remove features
- Change the train-test split ratio
- Implement cross-validation

## License

This project is open source and available for educational purposes.


---

**Last Updated**: May 2026
