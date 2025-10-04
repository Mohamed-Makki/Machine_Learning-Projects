# Diabetes Prediction: SVM Classification System

A machine learning project that predicts diabetes risk using the PIMA Indian Diabetes Dataset and Support Vector Machine (SVM) algorithm.

## Overview

This project implements a binary classification system for diabetes prediction based on medical diagnostic measurements. The model analyzes 8 key health indicators to accurately predict whether a patient is diabetic or non-diabetic.

## Dataset

- **Samples**: 768 patients from PIMA Indian population
- **Features**: 8 medical measurements (Glucose, BMI, Age, Insulin, etc.)
- **Target**: Binary classification (0 = Non-Diabetic, 1 = Diabetic)
- **Split**: 80% training, 20% testing

## Requirements

```
numpy
pandas
scikit-learn
```

## Installation

1. Clone the repository
2. Install required packages:
```bash
pip install numpy pandas scikit-learn
```
3. Ensure the `diabetes.csv` file is in the project directory

## Usage

Run the main script:
```bash
python diabetes_classification.ipynb
```

The script will:
- Load and analyze the diabetes dataset
- Perform data standardization for optimal SVM performance
- Train an SVM model with linear kernel
- Evaluate model performance on test data
- Demonstrate prediction on sample patient data

## Results

- **Training Accuracy**: ~77-85%
- **Testing Accuracy**: ~75-82%
- **Model**: Support Vector Machine with linear kernel and data standardization

## Key Features

- ✅ Data standardization for SVM optimization
- ✅ Comprehensive data analysis and exploration
- ✅ SVM model training with linear kernel
- ✅ Performance evaluation and metrics
- ✅ Patient risk prediction system
- ✅ Medical interpretation of results

## Dataset Features

1. **Pregnancies**: Number of pregnancies
2. **Glucose**: Plasma glucose concentration
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (mu U/ml)
6. **BMI**: Body mass index (weight in kg/(height in m)²)
7. **DiabetesPedigree**: Diabetes pedigree function
8. **Age**: Age in years

## Applications

- Early diabetes detection systems
- Healthcare risk assessment tools
- Preventive medicine applications
- Medical decision support systems

## File Structure

```
├── Project/Diabetes Prediction Classification.ipynb    # Main project script
├── Dataset/diabetes.csv                			    # PIMA diabetes dataset
└── README.md                 						    # Project documentation
```

## Conclusion

This project successfully demonstrates the effectiveness of Support Vector Machine algorithm in medical diagnosis prediction. The SVM model with proper data standardization achieved reliable performance in predicting diabetes risk based on medical diagnostic measurements.

The implementation showcases the critical importance of data preprocessing, particularly feature standardization, which significantly improves SVM algorithm performance. This approach proves valuable for healthcare applications where accurate risk assessment can enable early intervention and better patient outcomes.

The project serves as a practical example of applying machine learning techniques to real-world healthcare challenges and demonstrates how AI can assist medical professionals in making informed decisions.
