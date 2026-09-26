# Amazon Product Review Rating Prediction

This project explores whether **customer review text alone can be used to predict Amazon product ratings** using Natural Language Processing (NLP) and Machine Learning.

## Project Overview

The project started with the messy **Amazon Product Reviews dataset from Kaggle**, which contained **30 columns** with a mixture of product information, review information, identifiers, and other metadata.

We first performed **Exploratory Data Analysis (EDA)** to understand the dataset, examine missing values, analyze feature distributions, and investigate relationships between the available features and review ratings.

During the analysis, we found that many columns contained substantial amounts of missing data and several features were not directly relevant to predicting a review's rating. As a result, we focused the modeling task on the two most relevant components:

* **Feature:** Review text
* **Target:** Review rating (1–5 stars)

## Data Processing

The review data was cleaned by:

* Removing rows with missing review text or ratings
* Validating and standardizing rating values
* Removing duplicate reviews
* Removing empty reviews
* Cleaning the review text
* Converting text into numerical features using **TF-IDF**
* Using unigrams and bigrams to capture individual words as well as short phrases

The dataset was then divided into training and testing sets while maintaining the distribution of rating classes.

## Machine Learning Models

Two classification algorithms were trained and compared:

* **Logistic Regression**
* **Linear Support Vector Classifier (LinearSVC)**

The models were evaluated using accuracy and F1-based metrics, with particular attention to **macro F1** because of the significant class imbalance in the dataset.

## Results & Limitations

The models achieved approximately **66–69% accuracy**, which is substantially higher than random five-class prediction but only moderately above the majority-class baseline.

The main limitation was the **severe imbalance in rating classes**. The dataset contains far more 4★ and 5★ reviews than lower-rated reviews, while 1★ and 2★ reviews have very few examples. Consequently, the models struggle to learn reliable patterns for the less-represented rating classes.

The results therefore highlight an important lesson: **model performance depends not only on the choice of algorithm, but also on the quality, balance, and representativeness of the training data.**

## Data Leakage

During experimentation, an earlier version of the model included `reviews.doRecommend` as a feature and achieved higher accuracy. However, this feature is derived from the review/rating information and therefore introduces **target leakage**.

That version was discarded, and `reviews.doRecommend` was excluded from the final reported models to ensure that the evaluation reflects genuine prediction from review text rather than information derived from the target.

## What I Learned

This project expanded on my earlier machine learning work by introducing **NLP and text-based machine learning**. It gave me practical experience with:

* Working with a messy real-world dataset
* Exploratory Data Analysis
* Data cleaning and feature selection
* Text preprocessing
* TF-IDF feature extraction
* Multiclass classification
* Logistic Regression and LinearSVC
* Handling class imbalance
* Model comparison and evaluation
* Identifying and preventing data leakage

## Future Improvements

Potential next steps include:

* Reformulating the problem into three sentiment/rating classes
* Testing additional non-leaky features
* Experimenting with different text preprocessing strategies
* Exploring more advanced NLP models such as fine-tuned transformers
* Collecting or using a more balanced dataset with sufficient examples across all rating classes

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* Matplotlib
* Seaborn
* Google colab
