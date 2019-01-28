import config
import pandas as pd
import pickle

from preprocess_tweets import ProcessLanguage

from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

tweet_df = ProcessLanguage("training.csv")
tweet_df.import_dataset()

pd.set_option('display.max_colwidth', -1)

tweet_df.set_columns(config.SET_COLUMNS)
tweet_df.drop_columns(config.REMOVE_COLUMNS)

tweet_df.clean_tweet()

clean_df = tweet_df.dataframe

print("Finished processing tweets...")

train, test = train_test_split(clean_df, test_size=0.02, random_state=40, shuffle=True)

X_train = train['tweet'].values
X_test = test['tweet'].values
y_train = train['sentiment']
y_test = test['sentiment']

vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1,3))
vectorizer.fit(X_train)
X_train_dtm = vectorizer.transform(X_train)
X_test_dtm = vectorizer.transform(X_test)

"""
# SAVE TDIDF
save_tfidf = open("tfidf.pkl", "wb")
pickle.dump(vectorizer, save_tfidf)
save_tfidf.close()
"""

svm_classifier = LinearSVC()
svm_classifier.fit(X_train_dtm, y_train)

accuracies = cross_val_score(estimator=svm_classifier, X=X_train_dtm, y=y_train, cv=10)

print("Accuracy of 10-Fold Cross Validation", accuracies.mean())

y_pred_svm = svm_classifier.predict(X_test_dtm)

# Measure the accuracy of our model on the testing data
print("Accuracy on testing data: \n", metrics.accuracy_score(y_test, y_pred_svm))
print("Precision Score: ", metrics.precision_score(y_test, y_pred_svm, pos_label=4))
print("F1 Score: ", metrics.f1_score(y_test, y_pred_svm, pos_label=4))
print("AUC: ", metrics.roc_auc_score(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))


"""
svm_model = open("svm_model.pkl", "wb")
pickle.dump(svm_classifier, svm_model)
svm_model.close()
open_model = open("svm_model.pkl", "rb")
trained_model = pickle.load(open_model)
open_model.close()
score = trained_model.score(X_test_dtm, y_test)
print(score)
"""