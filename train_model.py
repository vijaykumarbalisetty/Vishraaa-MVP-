import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset.csv")

# Create encoders
le_occasion = LabelEncoder()
le_color = LabelEncoder()
le_style = LabelEncoder()
le_target = LabelEncoder()

# Encode input columns
df["Occasion"] = le_occasion.fit_transform(df["Occasion"])
df["Color"] = le_color.fit_transform(df["Color"])
df["Style"] = le_style.fit_transform(df["Style"])

# Features and target
X = df[["Occasion", "Color", "Style"]]
y = le_target.fit_transform(df["Recommendation"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = DecisionTreeClassifier()

model.fit(X_train, y_train)

# Test model
pred = model.predict(X_test)

print(
    "Accuracy:",
    accuracy_score(y_test, pred)
)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(le_occasion, "occasion.pkl")
joblib.dump(le_color, "color.pkl")
joblib.dump(le_style, "style.pkl")
joblib.dump(le_target, "target.pkl")

print("Model Trained Successfully!")