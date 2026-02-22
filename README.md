# Crime Rate Prediction Using Machine Learning

A predictive policing system that classifies crime cases as **Open** or **Closed** using supervised machine learning models.

## 🚀 Project Overview
- Evaluated 4 ML models for classification performance
- Random Forest achieved highest accuracy
- Built interactive Flask dashboard for real-time predictions
- Integrated Gradio interface for trend visualization

## 🛠 Tech Stack
- Python
- Scikit-learn
- Pandas
- Flask
- Gradio

## 📁 Project Structure
```
Crime-Rate-Prediction-ML/
│── app.py
│── requirements.txt
│── README.md
│
├── model/
│   └── crime_prediction_rf_model.pkl
│
├── data/
│   └── crime_dataset_india.csv
│
└── templates/
    └── index.html
```

## ▶️ How to Run

1. Clone the repository:
   ```
   git clone <your-repo-link>
   cd Crime-Rate-Prediction-ML
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Flask app:
   ```
   python app.py
   ```

4. Open in browser:
   ```
   http://127.0.0.1:5000/
   ```

---

## 📊 Model Performance
Random Forest Classifier achieved the highest accuracy among tested models.

---

## 👨‍💻 Author
Your Name
