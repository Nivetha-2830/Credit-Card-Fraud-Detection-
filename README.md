
# 💳 Credit Card Fraud Detection

This project is a web-based application developed to detect fraudulent credit card transactions using machine learning algorithms. It leverages user-friendly interfaces for login, registration, and prediction, and is built with Python, Flask, and essential data science libraries.

---

## 📁 Project Structure

```
FINAL YEAR PROJECT/
│
├── backgrounds/
│   ├── home_bg.png
│   ├── login_bg.png
│   ├── predict_bg.png
│   └── register_bg.png
│
├── model_files/
│   └── [Model-related files, if any]
│
├── app.py                  # Main Flask application
├── train_model.py         # ML model training script
├── predictions_log.csv    # Stores prediction history
├── user.csv               # Stores registered user data
├── fraudTest.csv          # Dataset used for predictions
├── package-lock.json      # Dependency lock file
└── README.md              # You're here!
```

---

## 🚀 How to Run the Project

### 1. Clone the repository or download the folder:
```bash
git clone <repo-url>
cd "Final Year Project"
```

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```
If there's no `requirements.txt`, manually install:
```bash
pip install flask pandas scikit-learn
```

### 3. Train the Machine Learning Model:
```bash
python train_model.py
```

### 4. Run the Flask App:
```bash
python app.py
```

### 5. Open in Browser:
Visit: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 ML Model

- Algorithms used: Logistic Regression, Decision Tree, Random Forest
- Techniques: Data preprocessing, SMOTE for class imbalance
- Dataset: `fraudTest.csv`
- Output: Trained model to classify transactions as "Fraud" or "Genuine"

---

## 🖼️ UI Design

The application includes the following pages with custom background images:

- **Home** - `home_bg.png`
- **Login** - `login_bg.png`
- **Register** - `register_bg.png`
- **Prediction Page** - `predict_bg.png`

---

## 🗃️ Logs and Storage

- **predictions_log.csv**: Stores prediction results.
- **user.csv**: Manages user credentials and roles.

---

## 🔒 Features

- User Authentication (Login/Register)
- Upload transaction data or predict manually
- Fraud detection results using ML
- User-friendly interface

---

## 🛠️ Technologies Used

- Python
- Flask
- Pandas, NumPy, Scikit-learn
- HTML/CSS (via Flask templates)
- VS Code

---

## ✍️ Author

- **Name**: Nivetha A
- **Degree**: MCA, Dhanalakshmi Srinivasan University
- **Project Guide**: [If applicable, add guide name]

---

## 📌 Notes

- Make sure `fraudTest.csv` is present and clean for model predictions.
- Background images should remain in the `backgrounds/` folder for UI to work correctly.

---

## 📷 Preview

*(Include screenshots of your web app UI here if needed)*

---

## 📃 License

This project is for educational purposes and academic submissions.
