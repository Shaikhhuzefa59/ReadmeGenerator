# NYC Airbnb Room Type Prediction Service

## High Level Overview
The NYC Airbnb Room Type Prediction Service is a FastAPI-based web application designed to predict the room type of an Airbnb listing based on various features such as location, price, review metrics, and availability. It provides both a static web interface for user interaction and a RESTful API for programmatic access.

## Features & Highlights
- Predicts Airbnb room types using a pre-trained machine learning model.
- User-friendly web interface for easy data input and result viewing.
- RESTful API endpoint for integration with other applications.
- Utilizes Pydantic for data validation and Joblib for model serialization.

## Tech Stack
- **FastAPI**: Web framework for building APIs.
- **Pandas**: Data manipulation and analysis.
- **Pydantic**: Data validation and settings management.
- **Scikit-learn**: Machine learning tools for predictive data analysis.
- **Uvicorn**: ASGI server to run FastAPI applications.
- **Joblib**: Lightweight pipelining for loading the trained model.

## Project Directory Structure
```
.gitignore
.python-version
main.py
Model_Pipeline.pkl.gz
nyc_airbnb_room_type_classification.ipynb
pyproject.toml
README.md
requirements.txt
uv.lock
static/
    index.html
    script.js
    style.css
```

## Getting Started

### Prerequisites
- Python 3.7+
- Poetry or pip for dependency management

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-repo/nyc-airbnb-room-type-prediction.git
   cd nyc-airbnb-room-type-prediction
   ```

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```
   *Or using pip:*
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
1. **Activate the virtual environment (if using Poetry):**
   ```bash
   poetry shell
   ```

2. **Run the application using Uvicorn:**
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the web interface:**
   Open your browser and go to `http://127.0.0.1:8000/`.

4. **Use the API endpoint:**
   You can send POST requests to `http://127.0.0.1:8000/predict` with a JSON payload containing the features of an Airbnb listing to get predictions.

## Contributing
Contributions are welcome! Please follow these steps to contribute to the project:
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.