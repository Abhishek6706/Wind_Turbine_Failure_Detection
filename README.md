# Wind Turbine Generator Failure Prediction using Machine Learning

This project uses sensor data from wind turbines to predict generator failure. By employing a deep learning model, it achieves **92% accuracy**, enabling proactive maintenance, reducing downtime, and maximizing energy production.

---

## Table of Contents
- [About The Project](#about-the-project)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results & Conclusion](#results--conclusion)

---

## About The Project

The primary goal of this project is to build a reliable machine learning model that can predict the failure of a wind turbine's generator based on 40 different sensor readings. Early detection of potential failures is crucial for scheduling maintenance, preventing catastrophic damage, and ensuring the continuous operation of the wind farm.

This predictive model serves as a powerful tool for shifting from reactive to **proactive maintenance strategies**.

---

## Dataset

The model was trained and evaluated on a dataset containing 20,000 samples. Each sample represents a snapshot of a wind turbine's operational state and includes:
* **40 Features (V1-V40):** Numerical data from various sensors monitoring the turbine's components.
* **1 Target Variable (Target):** A binary indicator where `1` signifies a generator failure and `0` indicates normal operation.

An initial exploratory data analysis revealed a class imbalance, with significantly fewer instances of failure than normal operation. This was a key consideration in the modeling process.

---

## Methodology

The project follows a standard machine learning workflow from data preprocessing to model evaluation.

### 1. Data Preprocessing
* **Handling Missing Values:** Missing data points in features 'V1' and 'V2' were handled by filling them with the mean value of their respective columns.
* **Feature Scaling:** All 40 features were standardized using `StandardScaler` from scikit-learn. This ensures that all features contribute equally to the model's performance without being skewed by differences in their scales.
* **Handling Class Imbalance:** To address the imbalance in the target variable, the **Synthetic Minority Over-sampling Technique (SMOTE)** was applied. This technique generates new synthetic samples for the minority class (failures) to create a balanced dataset for training.

### 2. Model Architecture
A **Sequential Deep Learning Model** was built using TensorFlow/Keras. The architecture is designed to capture complex patterns in the sensor data and consists of:
* An input layer and several hidden `Dense` layers with `ReLU` activation.
* `Dropout` layers to prevent overfitting by randomly setting a fraction of input units to 0 during training.
* A final `Dense` output layer with a `Sigmoid` activation function, which outputs a probability score for the binary classification task (failure or no failure).

The model was compiled with the `adam` optimizer and `binary_crossentropy` loss function, which are well-suited for binary classification problems.

---

## Results & Conclusion

The model's performance was evaluated on a held-out test set, demonstrating strong predictive capabilities.

* **Accuracy:** The model achieved an overall **accuracy of 92%**.
* **Classification Report:** The report showed a high `precision`, `recall`, and `f1-score` for both classes, indicating that the model is effective at identifying both normal operations and generator failures.
* **Confusion Matrix:** The confusion matrix confirmed the model's ability to correctly classify a high number of true positives (actual failures) and true negatives (normal operations).

In conclusion, this project successfully demonstrates that a deep learning model can be used to predict wind turbine generator failures with a high degree of confidence. This enables a shift to a more efficient and cost-effective predictive maintenance schedule, ultimately maximizing energy output and operational lifespan.
