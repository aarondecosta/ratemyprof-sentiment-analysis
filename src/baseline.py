import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Load Data from JSON File
with open("reviews.json", "r") as f:
    data = json.load(f)

# Extract features (comments) and target (ratings) from data
comments = [sample["comment"] for sample in data]
ratings = [sample["rating"] for sample in data]

# Step 2: Preprocess Data (if needed)
# No preprocessing in this example, but you can apply tokenization, remove stopwords, etc.

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(
    comments, ratings, test_size=0.2, random_state=42
)

# Step 4: Define and Train Model
# Define TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

# Define Multinomial Naive Bayes classifier
clf = MultinomialNB()

# Train the classifier
clf.fit(X_train_tfidf, y_train)

# Step 5: Evaluate Model
# Transform the test data using the TF-IDF vectorizer
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Make predictions on the test data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
