- # Cancer Classification using Machine Learning

  ## 📌 Project Overview

  This project focuses on **Biomedical Text Classification** for cancer research publications. The task is to classify long medical research papers into one of three cancer categories:

  - **Thyroid Cancer**
  - **Colon Cancer**
  - **Lung Cancer**

  The dataset contains **7,569 scientific publications**, collected and labeled for supervised machine learning. Each document is represented as raw text (abstracts and full papers), which is transformed into numerical features using Natural Language Processing (NLP) techniques.

  ------
  
  ## 📊 Dataset Description
  
  - **Total Samples**: 7,569 documents
  - **Classes**: 3 (Thyroid, Colon, Lung)
  - **Class Distribution**:
    - Colon Cancer: 2,579 documents
    - Lung Cancer: 2,180 documents
    - Thyroid Cancer: 2,810 documents
  
  ------
  
  ## 🔎 Data Preprocessing
  
  1. **Text Cleaning**: Removing punctuation, lowercasing, removing special characters.
  2. **Tokenization**: Splitting text into words/sentences using NLTK.
  3. **Stopword Removal**: Filtering common English stopwords.
  4. **Stemming**: Applying Porter Stemmer to reduce words to their base form.
  5. **Feature Extraction**:
     - **CountVectorizer**
     - **TF-IDF (Term Frequency - Inverse Document Frequency)**

  ------

  ## 🤖 Machine Learning Models
  
  The following classifiers were implemented and compared:
  
  - **Logistic Regression**
  - **Random Forest**
  - **Gradient Boosting**
  - **Naïve Bayes** (Gaussian, Multinomial, Bernoulli)
  - **Decision Tree**
  - **K-Nearest Neighbors (KNN)**
  - **Support Vector Machine (SVM)**
  - **Stochastic Gradient Descent (SGDClassifier)**
  - **Linear Discriminant Analysis (LDA)**
  - **Quadratic Discriminant Analysis (QDA)**
  
  ------
  
  ## 📈 Model Evaluation
  
  - **Accuracy Score**
  - **Confusion Matrix**
  - **Classification Report (Precision, Recall, F1-score)**
  
  Dimensionality reduction (PCA) was used for visualization of document clusters in 2D.
  
  ------

  ## 📊 Visualizations
  
  - **Class Distribution Plots**
  - **WordClouds** for each cancer type to identify common keywords.
  - **PCA Projections** to visualize document separability in reduced dimensions.

  ------
  
  ## 🛠️ Tech Stack

  - **Python** (Pandas, NumPy, Scikit-learn, NLTK)
  - **Visualization**: Matplotlib, Seaborn, WordCloud
  - **NLP**: Tokenization, Stopword Removal, Stemming, TF-IDF
  - **Models**: Logistic Regression, Random Forest, Gradient Boosting, Naive Bayes, SVM, etc.
  
  ------
  
  ## 🚀 Conclusion
  
  This project demonstrates the power of **NLP + Machine Learning** in classifying biomedical documents. Among all models tested, TF-IDF combined with linear classifiers (e.g., Logistic Regression, SVM) showed strong performance, making them suitable for large-scale text classification tasks.
  
  Future improvements could include:
  
  - Using advanced deep learning models (e.g., BERT, BioBERT).
  - Expanding dataset with more cancer research categories.
  - Building a real-time classification pipeline for biomedical applications.
  
  ------
  
  ## 📂 Project Structure
  
  ```
  ├── Project/Cancer classification.ipynb     # Jupyter notebook with analysis and modeling
  ├── Dataset/alldata_1_for_kaggle.csv        # Dataset used for classification
  ├── README.md                       # Project documentation
  ```
