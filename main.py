import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

nltk.download('movie_reviews')

data = [
    (list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)
]

# with open('data.txt', 'w') as f:
#     f.write(str(data))
#     f.close

random.shuffle(data)
texts= [' '.join(words) for words, category in data]
category = [category for words, category in data]

split_idx = int(len(texts)*0.8)
x_train,x_test, y_train, y_test = texts[:split_idx], texts[split_idx:], category[:split_idx], category[split_idx:]

vec = TfidfVectorizer(max_features=10000)
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)

model = LogisticRegression()
model.fit(x_train_vec, y_train)

y_pred= model.predict(x_test_vec)
acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred)

print(f'accuracy: {acc:.4f}')
print(f'classification: \n{rep}')