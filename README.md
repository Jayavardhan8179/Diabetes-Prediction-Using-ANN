# Diabetes Prediction Using Artificial Neural Network (ANN)

---

## 📖 Introduction
Diabetes is a chronic medical condition where the body is unable to regulate blood glucose levels effectively. Early detection of diabetes is critical to prevent long-term complications such as heart disease, kidney failure, and nerve damage.

This project uses an **Artificial Neural Network (ANN)** to predict whether a person has diabetes based on clinical and lifestyle-related attributes. ANN is well suited for this task because medical data often contains **non-linear and complex relationships** that traditional machine learning models may not fully capture.

---

## 🎯 Objective of the Project
The main objectives of this project are:

- To build an accurate **diabetes prediction model**
- To apply **Artificial Neural Networks** on large-scale healthcare data
- To perform proper **data preprocessing and feature selection**
- To evaluate the model using standard classification metrics
- To demonstrate a **real-world healthcare application** of ANN

---

## 📊 Dataset Description
- **Dataset Name:** Diabetes Clinical Dataset
- **Source:** Kaggle
- **Number of Records:** 100,000+
- **Data Type:** Structured tabular data
- **Domain:** Healthcare / Medical Analytics

### 🎯 Target Variable
- `diabetes`
  - `0` → Non-Diabetic
  - `1` → Diabetic

---

## 🧾 Feature Explanation

### ✅ Features Used in the Model
The following features were selected because they have **direct medical or lifestyle relevance** to diabetes prediction:

| Feature | Description |
|------|------------|
| `age` | Age of the patient |
| `gender` | Biological gender |
| `bmi` | Body Mass Index – obesity indicator |
| `hbA1c_level` | Average blood glucose over past 2–3 months |
| `blood_glucose_level` | Current blood glucose level |
| `hypertension` | Indicates high blood pressure (0/1) |
| `heart_disease` | Indicates heart disease history (0/1) |
| `smoking_history` | Smoking behavior of the patient |

---

### ❌ Features Removed
The following columns were removed because they either add noise, bias, or are not directly useful for medical diagnosis:

- `year`
- `location`
- `race:AfricanAmerican`
- `race:Asian`
- `race:Caucasian`
- `race:Hispanic`
- `race:Other`

> **Reason for removal:**  
> These attributes may introduce ethical bias and do not directly influence the biological diagnosis of diabetes.

---

## 🧹 Data Preprocessing (Step-by-Step)

Data preprocessing is one of the most important stages in an ANN project.

### 1️⃣ Dropping Unnecessary Columns
- Removed non-medical and sensitive demographic features.

### 2️⃣ Handling Invalid Values
Certain medical features cannot logically have zero values.  
Zero values were replaced with `NaN` for:
- `age`
- `bmi`
- `hbA1c_level`
- `blood_glucose_level`

### 3️⃣ Missing Value Treatment
- Missing values were handled using **median imputation**.
- Median was chosen to avoid the impact of outliers.

### 4️⃣ Encoding Categorical Variables
- `gender`
- `smoking_history`

These were converted into numerical format using **One-Hot Encoding** so they could be processed by the ANN.

### 5️⃣ Feature Scaling
- Numerical features were normalized using **StandardScaler**.
- Scaling is essential for ANN to ensure faster convergence and better performance.

### 6️⃣ Train-Test Split
- Dataset was split into training and testing sets.
- This helps evaluate the model’s ability to generalize to unseen data.

---

## 🧠 Why Artificial Neural Network (ANN)?
ANN was selected instead of traditional machine learning models because:

- Medical data involves **non-linear relationships**
- ANN automatically learns hidden patterns
- Better performance for complex classification tasks
- Mimics human brain learning through neurons and weights

---

## 🏗 ANN Model Architecture

### 🔹 Model Structure
- **Input Layer:** Receives processed feature inputs
- **Hidden Layers:**
  - Dense layers with ReLU activation
  - Dropout layers to reduce overfitting
- **Output Layer:**
  - Single neuron with Sigmoid activation for binary classification

### ⚙️ Model Configuration
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam
- **Activation Functions:** ReLU, Sigmoid

---

## 📈 Model Evaluation
The model performance was evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **Confusion Matrix**
- **ROC-AUC Curve**

### 🏆 Performance Result
- Achieved **approximately 85–90% accuracy**
- Improved detection of diabetic patients
- Controlled overfitting through dropout and scaling

---

## 🛠 Technologies & Tools Used
- **Programming Language:** Python
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - TensorFlow / Keras
  - Matplotlib
  - Seaborn
- **Environment:** Jupyter Notebook

---

## ▶️ How to Run This Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/Jayavardhan8179/Diabetes-Prediction-Using-ANN.git

