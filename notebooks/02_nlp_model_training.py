import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure models and visuals directories exist
os.makedirs("../models", exist_ok=True)
os.makedirs("../visuals", exist_ok=True)

# 1. Load Clean Processed Dataset
print("--> Loading Clean Multilingual Fraud Dataset for Model Training...")
df = pd.read_csv("../data/processed/afri_fraud_clean_dataset.csv")

# 2. Train / Test Split
X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"    Training Samples: {len(X_train)} | Testing Samples: {len(X_test)}")

# ==============================================================================
# 3. Build & Train the NLP Machine Learning Pipeline
# ==============================================================================
print("\n--> Building & Training Advanced N-Gram TF-IDF + Logistic Regression Pipeline...")
# We use ngram_range=(1, 2) to capture word pairs like 'account blocked', 'urgent 2k', 'bvn update'
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
    ('clf', LogisticRegression(C=10.0, class_weight='balanced', random_state=42))
])

pipeline.fit(X_train, y_train)

# ==============================================================================
# 4. Model Evaluation & Metrics
# ==============================================================================
print("\n--> Evaluating Model Performance on Out-of-Sample Test Data...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n--- Model Accuracy: {accuracy * 100:.2f}% ---")
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Genuine Banking (0)', 'Fraud / Threat (1)']))

# ==============================================================================
# 5. Hero Chart 3: Confusion Matrix
# ==============================================================================
print("\n--> Generating Hero Chart 3: Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16, 'weight': 'bold'},
            xticklabels=['Genuine (0)', 'Fraud (1)'], yticklabels=['Genuine (0)', 'Fraud (1)'])

plt.title(f"NLP Model Confusion Matrix (Test Accuracy: {accuracy*100:.1f}%)", pad=20, fontweight='bold')
plt.xlabel("Predicted Label", fontweight='bold')
plt.ylabel("True Label", fontweight='bold')
plt.tight_layout()

chart3_path = "../visuals/03_nlp_confusion_matrix.png"
plt.savefig(chart3_path, dpi=300)
print(f"    Saved Hero Chart 3 to: {chart3_path}")
plt.show()

# ==============================================================================
# 6. Hero Chart 4: Top Fraud vs. Genuine Keywords (Interpretability)
# ==============================================================================
print("\n--> Extracting Feature Importance (Top Words & N-Grams)...")
# Extract vectorizer and classifier from pipeline
vectorizer = pipeline.named_steps['tfidf']
classifier = pipeline.named_steps['clf']

feature_names = np.array(vectorizer.get_feature_names_out())
coefficients = classifier.coef_[0]

# Combine into a dataframe
coef_df = pd.DataFrame({'Word / N-Gram': feature_names, 'Coefficient': coefficients})

# Top 10 Fraud words (highest positive coefficients) and Top 10 Genuine words (lowest negative coefficients)
top_fraud = coef_df.sort_values(by='Coefficient', ascending=False).head(12)
top_genuine = coef_df.sort_values(by='Coefficient', ascending=True).head(12)
top_features = pd.concat([top_fraud, top_genuine]).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(14, 8))
# Using hue and legend=False to avoid seaborn deprecation warnings
sns.barplot(x='Coefficient', y='Word / N-Gram', hue='Word / N-Gram', data=top_features, 
            palette=['#d9534f' if c > 0 else '#4682b4' for c in top_features['Coefficient']], legend=False)

plt.title("NLP Model Explainability: Top Keywords Driving Fraud vs. Genuine Classifications", pad=20, fontweight='bold')
plt.xlabel("Model Weight (Positive = Strong Fraud Indicator | Negative = Strong Genuine Indicator)", fontweight='bold')
plt.ylabel("Word / Bigram Feature", fontweight='bold')
plt.tight_layout()

chart4_path = "../visuals/04_nlp_top_fraud_words.png"
plt.savefig(chart4_path, dpi=300)
print(f"    Saved Hero Chart 4 to: {chart4_path}")
plt.show()

# ==============================================================================
# 7. Save Production Model Weights (For FastAPI & Streamlit)
# ==============================================================================
print("\n--> Saving Trained Production Model Pipeline...")
model_path = "../models/afri_fraud_model.joblib"
joblib.dump(pipeline, model_path)
print(f"    Successfully saved model pipeline to: {model_path}")
print(f"    File Size: {os.path.getsize(model_path) / 1024:.1f} KB")

print("\n--> Day 2 NLP Model Training Complete! Ready for FastAPI microservice integration.")