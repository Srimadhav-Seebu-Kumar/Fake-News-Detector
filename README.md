﻿# Fake News Detection using NLP

This project focuses on detecting fake news using Natural Language Processing (NLP) techniques. A supervised machine learning model is trained on a labeled dataset consisting of real (`True.csv`) and fake (`Fake.csv`) news articles.

## Dataset

The dataset includes two files:

- `True.csv`: Contains news articles labeled as **real**.
- `Fake.csv`: Contains news articles labeled as **fake**.

Each file includes typical news metadata like:
- Title
- Text
- Subject
- Date

## Project Workflow

The notebook `train.ipynb` performs the following steps:

1. **Data Loading & Cleaning**: Combines `True.csv` and `Fake.csv` into a single labeled dataset.
2. **Preprocessing**: 
   - Removing punctuation and stopwords
   - Tokenization
   - Lemmatization
3. **Text Vectorization**: Uses methods like TF-IDF or CountVectorizer.
4. **Model Building**: Trains classification models (e.g., Logistic Regression, Naive Bayes, or deep learning models).
5. **Evaluation**: Reports accuracy, confusion matrix, precision, recall, and F1-score.
6. **Model Saving**: Optionally saves the trained model for deployment.

## How to Run

### Requirements

Make sure the following are installed:

- Python 3.7+
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- nltk
- matplotlib
- seaborn

You can install them using:

```bash
pip install -r requirements.txt

```

Or mannualy:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn
```

# Running the Notebook
Place True.csv and Fake.csv in the project directory.

# Outputs

Trained model (optional .pkl or .h5 file)

Performance metrics and plots

Cleaned and preprocessed dataset (if saved)

# Example Use Case

After training, you can use the model to classify whether new or unseen news articles are likely to be real or fake based on their content.

# License

This project is licensed under the MIT License.
