# Heart Risk Perdictor

## 📌 Overview
Heart Risk Perdictor is a deep learning-based project that analyzes patient health data to determine the likelihood of heart disease. Using advanced neural network models built with TensorFlow/Keras, this project enables early risk assessment for medical professionals.

## 🚀 Features
- **Deep Learning Model** trained with medical data
- **Data Preprocessing** including normalization and feature scaling
- **Saved Model Deployment** for real-time predictions
- **User-friendly Interface** for easy interaction

## 🏗 Tech Stack
- Python 🐍
- TensorFlow/Keras
- Scikit-learn
- Pandas & NumPy
- Joblib (for saving the scaler)

## 📂 Project Structure

- `data/` → Contains the dataset used for training.
- `models/` → Stores trained models (`.h5` files).
- `src/` → Includes Python scripts for model training and preprocessing.
- `templates/` → Holds HTML files for the web interface.
- `app.py` → Main file to run the web app.
- `README.md` → Documentation for usage and setup.

## 🔧 Installation & Usage
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/heart-disease-predictor.git
   cd heart-disease-predictor
2. ** For creating virtual Environment**
   
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
       pip install -r requirements.txt

4. ** Running training Script**

       python src/train_model.py

5. **Deploy the Model**

       python app.py
