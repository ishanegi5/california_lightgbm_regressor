# California Housing Price Prediction using LightGBM 🏡

This project demonstrates the use of **LightGBM Regressor** on the **California Housing dataset** for predicting median house values.  

---

## 📌 Project Overview
- Dataset: California Housing (from `sklearn.datasets`)
- Task: **Regression** (predict continuous target variable - median house value)
- Model: LightGBM Regressor (`LGBMRegressor`)
- Evaluation Metrics: R², MSE, RMSE, MAE

---

## 📂 Dataset
The dataset contains **20,640 samples** with features like:
- Median income
- Average number of rooms
- Population
- Latitude, Longitude  
Target: Median house value in California districts.

---

## ⚙️ Model
```python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=55500,
    max_depth=2,
    num_leaves=3,
    objective='regression',
    random_state=42
)

model.fit(X_train, y_train)
📊 Results
Evaluation on the test set:

R² Score: 0.93

MSE: 0.067

RMSE: 0.259

MAE: 0.183

These results show that LightGBM can effectively capture non-linear relationships for housing price prediction.

🔑 Key Learnings
Increasing the number of boosting iterations (n_estimators) improved model performance, but required more computation time.

LightGBM is efficient for tabular regression tasks and handles large datasets well.

Metrics like R², RMSE, MAE provide deeper insights than accuracy.

🚀 How to Run
Clone this repo:


git clone https://github.com/ishanegi5/california_lightgbm_regressor.git
cd california_lightgbm_regressor
Install requirements:


pip install -r requirements.txt
Run the Jupyter Notebook or Python script.

📦 Requirements
Python 3.x

scikit-learn

lightgbm

pandas

numpy

Install all with:


pip install lightgbm scikit-learn pandas numpy


👩‍💻 Author
Isha Negi
