import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# Ensure visuals directory exists
os.makedirs("../visuals", exist_ok=True)

# 1. Load Clean Processed Dataset
print("--> Loading Clean Multilingual Fraud Dataset for EDA...")
df = pd.read_csv("../data/processed/afri_fraud_clean_dataset.csv")

print("\n--- Dataset Overview ---")
print(df.info())

print("\n--- Class Distribution ---")
print(df['label'].value_counts(normalize=True) * 100)

print("\n--- Language Breakdown ---")
print(df['language'].value_counts())

# ==============================================================================
# 2. Hero Chart 1: Multilingual Data Representation & Balance
# ==============================================================================
print("\n--> Generating Hero Chart 1: Multilingual Data Representation...")
plt.figure(figsize=(14, 7))

# Plot count of records per language colored by fraud label (0 = Genuine, 1 = Fraud/Threat)
# Using hue and legend to avoid seaborn future warnings
ax = sns.countplot(x='language', hue='label', data=df, palette=['#4682b4', '#d9534f'])

plt.title("Multilingual Banking Representation: Fraud vs. Genuine Transactions across African Languages", pad=20, fontweight='bold')
plt.xlabel("Language (Code-Switched & Indigenous)")
plt.ylabel("Number of Text Messages")

# Customizing legend
legend_labels = ['Genuine Banking / Support (Label 0)', 'Fraud / Loan Shark Threat (Label 1)']
plt.legend(title="Text Intent", labels=legend_labels, frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
chart1_path = "../visuals/01_multilingual_data_distribution.png"
plt.savefig(chart1_path, dpi=300)
print(f"    Saved Hero Chart 1 to: {chart1_path}")
plt.show()

# ==============================================================================
# 3. Hero Chart 2: Fraud Intent Across Banking Categories
# ==============================================================================
print("\n--> Generating Hero Chart 2: Fraud Intent by Category...")
plt.figure(figsize=(14, 7))

# Plot category breakdown
ax2 = sns.countplot(y='category', hue='label', data=df, palette=['#4682b4', '#d9534f'])

plt.title("Scam Vectors: Analysis of Threat Categories in Nigerian Banking Ecosystem", pad=20, fontweight='bold')
plt.xlabel("Number of Text Messages")
plt.ylabel("Scam / Transaction Category")

plt.legend(title="Text Intent", labels=legend_labels, frameon=True, facecolor='white', framealpha=0.9)

plt.tight_layout()
chart2_path = "../visuals/02_fraud_intent_by_category.png"
plt.savefig(chart2_path, dpi=300)
print(f"    Saved Hero Chart 2 to: {chart2_path}")
plt.show()

print("\n--> Day 1 EDA Complete! All visuals exported successfully.")