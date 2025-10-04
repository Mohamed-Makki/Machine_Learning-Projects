# 🩺 Disease Prediction using Machine Learning

## 📌 Overview

This project aims to **predict potential diseases** based on the symptoms provided by the user.
 It leverages multiple **machine learning algorithms** to ensure reliable results and provides a user-friendly **Tkinter-based GUI** for interaction.

------

## 🛠️ Project Workflow

1. **Data Collection**
   - Utilized `Training.csv` and `Testing.csv` datasets containing symptoms and their corresponding diseases.
2. **Data Preprocessing**
   - Encoded disease labels into numerical values for model training.
   - Used symptom indicators as features (binary representation: presence or absence of each symptom).
3. **Machine Learning Models**
    Implemented and compared three different ML classifiers:
   - 🌳 **Decision Tree Classifier**
   - 🌲 **Random Forest Classifier**
   - 📊 **Naive Bayes Classifier**
4. **Evaluation**
   - Accuracy scores were computed on the testing dataset.
   - Models were compared to identify strengths and weaknesses.
5. **User Interface**
   - A graphical user interface (GUI) built with **Tkinter** allows users to:
     - Select symptoms from dropdown menus.
     - Run predictions using different classifiers.
     - View the predicted disease instantly.

------

## 📊 Visualizations & Insights

- **Distribution Analysis**: Explored the frequency of symptoms and class imbalance in diseases.
- **Charts**: Used bar plots and pie charts to visualize target class distributions.
- **Key Insights**:
  - Random Forest showed higher stability and accuracy.
  - Naive Bayes performed efficiently with limited computation but slightly lower accuracy.
  - Decision Tree provided interpretable results, useful for understanding feature importance.

------

## 🚀 How to Run

1. **Install Dependencies**

   ```
   pip install pandas numpy scikit-learn matplotlib seaborn
   ```

   *(Tkinter is pre-installed with most Python distributions.)*

2. **Run the Application**

   ```
   python clean_code.py
   ```

   - Select symptoms from the dropdown.
   - Click the classifier button to predict the disease.

------

## 📂 Project Structure

```
├── Project/Disease_Predictor.ipynb   # Jupyter Notebook with analysis & visualizations
├── Project/clean_code.py             # Main application with ML models & Tkinter GUI
├── Datasets/Training.csv             # Training dataset
├── Datasets/Testing.csv              # Testing dataset
├── README.md                 		  # Project documentation
```

------

## 💡 Future Improvements

- Expand dataset with more diseases and symptoms.
- Integrate deep learning models for enhanced accuracy.
- Deploy as a **web application** for broader accessibility.
- Add explainability (e.g., SHAP values) to understand model decisions.
