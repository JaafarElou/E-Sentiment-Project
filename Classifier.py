import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib



file_path = 'Data\model training data.csv'
data = pd.read_csv(file_path)

data.dropna(subset=['clean_tweet'], inplace=True)
data.dropna(subset=['Label'], inplace=True)

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_tweet'])
y = data['Label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)


joblib.dump(nb_classifier, r'Saved Model\nb_classifier.pkl')
joblib.dump(vectorizer, r'Saved Model\vectorizer.pkl')