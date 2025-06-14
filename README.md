
# 📩 Spam vs Ham Classification using NLTK and Bag of Words (BoW)

This project focuses on building a **Spam Detection Model** using **Natural Language Processing (NLP)** techniques with the **Bag of Words** approach. The model classifies SMS messages into either **Spam** or **Ham (Not Spam)** using **NLTK** and **Scikit-learn**. Development was done entirely on **Google Colab** for convenience and reproducibility.

## 🔍 Project Overview

- **Objective**: Classify SMS messages as spam or ham.
- **Dataset**: SMS Spam Collection Dataset.
- **Platform**: Developed and tested on **Google Colab**.
- **Libraries Used**: `NLTK`, `sklearn`, `pandas`, `matplotlib`, `seaborn`.

## 🧠 Key Concepts

- Natural Language Processing (NLP)
- Text Preprocessing (tokenization, stemming, stopword removal)
- Feature Extraction using **Bag of Words**
- Model Building with **Multinomial Naive Bayes**
- Performance Metrics: Accuracy, Confusion Matrix, Precision, Recall

## 🧹 Data Preprocessing Pipeline

1. **Lowercasing & Punctuation Removal**
2. **Tokenization**
3. **Stopword Removal** using NLTK
4. **Stemming** using `PorterStemmer`
5. **Feature Vectorization** using `CountVectorizer` (BoW)

## 🧪 Model Training & Evaluation

- **Algorithm**: Multinomial Naive Bayes
- **Text Representation**: CountVectorizer (BoW)
- **Evaluation**: Accuracy score, Confusion matrix, Classification report

### ✅ Results

- **Training Accuracy**: ~98%
- **Test Accuracy**: ~96%
- Balanced performance across precision and recall.

## 🚀 How to Use This Notebook

1. Open in Google Colab:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. Upload the notebook `Spam_Ham_NLTK_Project_using_BOW.ipynb` to your Colab environment.

3. Run each cell in order to:
   - Load and preprocess the dataset
   - Train the spam detection model
   - Evaluate and visualize performance

## 📁 File Structure

```
📦 Spam_Ham_NLTK_Project_using_BOW
 ┣ 📜 Spam_Ham_NLTK_Project_using_BOW.ipynb
 ┗ 📄 README.md
```

## 📚 References

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [SMS Spam Collection Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

## 🙌 Acknowledgements

- Developed using **Google Colab**
- Inspired by educational tutorials on text classification
- Thanks to the open-source community for tools and resources!
