import nltk
from nltk.corpus import movie_reviews
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

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

vec = TfidfVectorizer(max_features=5000)
x_train_vec = vec.fit_transform(x_train)
x_test_vec = vec.transform(x_test)

model = LogisticRegression(
    C=5,                  # weaker regularization (can improve performance)
    penalty="l2",         # default, but explicit
    solver="liblinear",   # good for sparse, small datasets
    class_weight=None,    # or "balanced" if imbalance
    max_iter=2000,        # ensure convergence
    random_state=42
)
model.fit(x_train_vec, y_train)

y_pred= model.predict(x_test_vec)
acc = accuracy_score(y_test, y_pred)
rep = classification_report(y_test, y_pred)

print(f'accuracy: {acc:.4f}')
print(f'classification: \n{rep}')

joblib.dump(model, 'are_bisi_model_bann_gaya.pkl')
joblib.dump(vec, 'are_bisi_tokens.pkl')