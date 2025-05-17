import os
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
import pickle

# Step 1: Prepare the Dataset
def prepare_intent_dataset(data_dir="./data"):
    data = []
    json_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    for file in json_files:
        file_path = os.path.join(data_dir, file)
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                entries = json.load(f)
                for entry in entries:
                    if "question" in entry and "intent" in entry:
                        data.append({"question": entry["question"], "intent": entry["intent"]})
            except Exception as e:
                print(f"Error reading {file}: {e}")
    return pd.DataFrame(data)

# Step 2: Load and Preprocess the Dataset
df = prepare_intent_dataset()
print(f"Dataset loaded with {len(df)} entries.")
print("Class distribution before balancing:")
print(df["intent"].value_counts())

# Step 3: Handle Classes with Fewer Samples
# Remove classes with fewer than 2 samples
class_counts = df["intent"].value_counts()
valid_classes = class_counts[class_counts > 1].index
df = df[df["intent"].isin(valid_classes)]

# Oversample minority classes to balance the dataset
class_counts = df["intent"].value_counts()
max_class_count = class_counts.max()
balanced_data = []
for intent in class_counts.index:
    class_data = df[df["intent"] == intent]
    if len(class_data) < max_class_count:
        class_data = resample(class_data, replace=True, n_samples=max_class_count, random_state=42)
    balanced_data.append(class_data)
df = pd.concat(balanced_data)

print("Class distribution after balancing:")
print(df["intent"].value_counts())

# Step 4: Split the Data
X = df["question"]
y = df["intent"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Convert Text to Numerical Features Using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 6: Handle Class Imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=y.unique(), y=y)
class_weight_dict = {cls: weight for cls, weight in zip(y.unique(), class_weights)}

# Step 7: Train the Logistic Regression Model
model = LogisticRegression(class_weight=class_weight_dict, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 8: Evaluate the Model
y_pred = model.predict(X_test_tfidf)
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Step 9: Save the Model and Vectorizer
with open("intent_classifier.pkl", "wb") as f:
    pickle.dump(model, f)
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved.")

# Step 10: Load the Model and Vectorizer for Prediction
with open("intent_classifier.pkl", "rb") as f:
    intent_classifier = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def classify_intent(query):
    try:
        query_tfidf = vectorizer.transform([query])
        intent = intent_classifier.predict(query_tfidf)[0]
        return intent
    except Exception as e:
        print(f"Intent classification failed: {e}")
        return "unknown"

# Example Usage
query = "What is the fee structure for the MBA program?"
print(f"Intent: {classify_intent(query)}")

