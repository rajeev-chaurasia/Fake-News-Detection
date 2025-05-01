# Fake News Detection Project

This project aims to detect fake news using machine learning techniques. It involves combining multiple datasets, preprocessing text data, extracting features, training various classification models, and evaluating their performance.

## Datasets

Fake and Real News Dataset - https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset 

WelFake Dataset - https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

The project uses the following datasets, which should be placed in a `dataset/` directory relative to the notebook:

* `Fake.csv`
* `True.csv`
* `WELFake_Dataset.csv`

The `Fake.csv` and `True.csv` datasets are combined and labelled (0 for fake, 1 for real), and then concatenated with the `WELFake_Dataset.csv`, which already contains labels. Duplicates and missing values are handled during the data loading and preparation phase.

## Dependencies

The following Python libraries are required to run the notebook:

* `pandas`
* `os`
* `spacy`
* `re`
* `numpy`
* `matplotlib.pyplot`
* `seaborn`
* `collections`
* `wordcloud`
* `nltk`
* `sklearn.feature_extraction.text.TfidfVectorizer`
* `scipy.sparse.hstack`
* `tqdm.auto.tqdm`
* `pandarallel`
* `sklearn.model_selection.train_test_split`
* `sklearn.naive_bayes.MultinomialNB`
* `sklearn.linear_model.LogisticRegression`
* `sklearn.metrics.accuracy_score`
* `sklearn.metrics.precision_score`
* `sklearn.metrics.recall_score`
* `sklearn.metrics.f1_score`
* `sklearn.metrics.confusion_matrix`

## Project Workflow 

![image](https://github.com/user-attachments/assets/f39dc825-7a4c-4530-a67d-fd997f0aa45d)

## Team Members 
[Ashish Bhusal](https://github.com/bhusalashish)
[Rajeev Ranjan Chaurasia](https://github.com/rajeev-chaurasia)
[Shrivaikunth Krishnakumar](https://github.com/shrivaikunthk)
