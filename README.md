# California Housing Price Prediction

Predicting median house values across California using a **Linear Regression model built from scratch** with NumPy — no sklearn for the model itself.

---

##  Overview

This project implements gradient descent-based linear regression from the ground up to predict housing prices based on features like location, income, proximity to ocean, and household stats.

The goal was to deeply understand what happens *under the hood* of a regression model rather than just calling a library function.

---

## Dataset

- **Source:** California Housing Dataset from kaggle
- **Samples:** 20,640 houses
- **Features:** 10 (longitude, latitude, housing age, rooms, bedrooms, population, households, income, ocean proximity)
- **Target:** Median house value

---

## What I Built

- Manual train/test split (80/20) using NumPy
- Z-score normalization (fit on train, applied to test — no data leakage)
- Linear regression class with bias term, gradient descent optimizer, and loss history tracking
- All evaluation metrics from scratch (MSE, RMSE, MAE, R²)

---

## Feature Engineering

Added 3 engineered features that significantly improved performance:

| Feature | Formula | Why |
|---|---|---|
| `rooms_per_household` | total_rooms / households | Raw room count is misleading without household context |
| `bedrooms_per_room` | total_bedrooms / total_rooms | Captures density of bedrooms |
| `population_per_household` | population / households | Better measure of crowding |

Also encoded `ocean_proximity` (5 categories → binary columns) instead of dropping it — ocean proximity is one of the strongest price predictors.

---

## Results

| Metric | Train | Test |
|---|---|---|
| R² | ~0.63 | ~0.62 |
| RMSE | ~$68,000 | ~$70,000 |
| MAE | ~$50,000 | ~$52,000 |

---

## 🧰 Tools Used

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-white?style=flat&logo=matplotlib&logoColor=black)

---

## 💡 What I Learned

- How gradient descent actually works mathematically
- Why feature engineering often matters more than model choice
- The importance of normalizing both X *and* y
- How to identify and fix data leakage in preprocessing
- Why linear regression has a performance ceiling on non-linear data

---

## 👩‍💻 Author

**Aditi Singh Pradhan**  
First Year CS @ BITS Pilani Goa × RMIT University  
[LinkedIn](https://www.linkedin.com/in/aditi-singh-pradhan-b4ab92351)

