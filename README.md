# census-income-predictor
# 💰 Adult Census Income Prediction Platform

An end-to-end Machine Learning web application that predicts whether an individual's annual income exceeds **\$50,000** based on census and demographic data. This platform utilizes a scikit-learn pipeline paired with a modern, responsive Streamlit dashboard interface.

---

## 🌐 Working Demo

You can interact with the fully deployed, live version of this artificial intelligence application directly in your web browser:

👉 **[Launch Live Web Application](https://census-income-predictor-bah.streamlit.app/)** 🚀

*(Note: Replace the link above with your actual live URL once deployed to Streamlit Community Cloud!)*

---

## 🚀 Key Features

* **Complete ML Pipeline:** Handles missing data cleaning, categorical feature engineering (Label Encoding), and robust numeric scaling (`StandardScaler`).
* **Optimized Classification Core:** Powered by a `DecisionTreeClassifier` engineered on thousands of census records.
* **Premium Dashboard Layout:** Features an emerald-green interactive control panel containing runtime session trackers, developer biographies, and quick test cases.
* **Production Ready Asset Exporting:** Bundles interdependent system objects cleanly using `joblib` serialization to guarantee zero feature leakage during real-time inference.

---

## 🛠️ Built With

* **Core Engine:** Python 3.10+
* **Data Processing:** Pandas, NumPy
* **Analytics Framework:** Scikit-Learn
* **Data Visualization:** Seaborn, Matplotlib
* **Web UI Development:** Streamlit

---

## 📥 Project Layout Structure

```text
├── adult.csv          # Base Census Dataset
├── train.py           # ML Model Pipeline Training Script
├── adult_app.py       # Streamlit Web Application Interface Script
├── model_assets.pkl   # Serialized Model Matrix Variables (Auto-generated)
├── requirements.txt   # System Package Configuration File
├── logo.png           # Brand Logo (Optional Placement)
└── profile.jpg        # Developer Profile Picture (Optional Placement)
```

---

## ⚙️ Quick Installation & Setup Checklist

### Step 1: Clone or Open Your Project Directory
Ensure all required files (`train.py`, `adult_app.py`, and `adult.csv`) are stored together in your current workspace directory.

### Step 2: Install System Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Training Engine
Generate the consolidated `model_assets.pkl` pipeline file by running the training script:
```bash
python train.py
```

### Step 4: Boot Up the Interactive UI Dashboard
Launch your web server using Streamlit:
```bash
streamlit run adult_app.py
```

---

## 🧑‍💻 Developer Profile

* **Developer Name:** Sulayman Bah
* **Professional Track:** Machine Learning Engineer / Python Developer
* **Primary Focus:** Building scalable, deployment-ready Data Science products.
