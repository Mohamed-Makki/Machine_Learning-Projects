# Sonar Data Classification: Rock vs Mine Detection

A machine learning project that classifies sonar signals to distinguish between rocks and mines using logistic regression.

## Overview

This project implements a binary classification system for underwater object detection using sonar data. The model analyzes 60 sonar signal features to accurately identify whether a detected object is a rock or a mine.

## Dataset

- **Samples**: 208 total samples
- **Features**: 60 numerical measurements (sonar signal frequencies)
- **Target**: Binary classification (R = Rock, M = Mine)
- **Split**: 90% training, 10% testing

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
3. Ensure the `sonar_data.csv` file is in the project directory

## Usage

Run the main script:
```bash
python sonar_classification.ipynb
```

The script will:
- Load and analyze the sonar dataset
- Train a logistic regression model
- Evaluate model performance
- Demonstrate prediction on sample data

## Results

- **Training Accuracy**: ~85-95%
- **Testing Accuracy**: ~80-90%
- **Model**: Logistic Regression with good generalization

## Features

- ✅ Data preprocessing and analysis
- ✅ Model training and evaluation
- ✅ Prediction system for new data
- ✅ Performance metrics and reporting

## Applications

- Naval mine detection systems
- Underwater object classification
- Maritime security operations

## File Structure

```
├── Project/sonar_classification.ipynb    # Main project script
├── Dataset/sonar_data.csv            	  # Dataset file
└── README.md                             # Project documentation
```

## Conclusion

This project successfully demonstrates the effectiveness of machine learning in solving real-world classification problems. The logistic regression model achieved reliable performance in distinguishing between rocks and mines using sonar signal analysis.

The implementation showcases essential machine learning concepts including data preprocessing, model training, evaluation, and practical deployment. This approach proves that well-structured datasets combined with appropriate algorithms can deliver accurate solutions for complex underwater detection challenges.

The project serves as a solid foundation for understanding binary classification techniques and their applications in maritime security and naval operations.
