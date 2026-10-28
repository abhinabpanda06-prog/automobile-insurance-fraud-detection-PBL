# ============================================================
# IMPORTS
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

sns.set_style('whitegrid')
pd.set_option('display.max_columns', None)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv('dataset.csv', compression='zip', encoding='utf-8-sig')

TARGET = 'FraudFound_P'
ID_COLS = ['PolicyNumber']
print("Shape:", df.shape)

# ============================================================
# DATASET DESCRIPTION
# ============================================================
print(f"\nRows: {df.shape[0]}, Columns: {df.shape[1]}")
df.info()

numeric_cols = [c for c in df.select_dtypes(include='number').columns if c not in [TARGET] + ID_COLS]
categorical_cols = list(df.select_dtypes(include='object').columns)
print(f"\nNumeric predictors ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical predictors ({len(categorical_cols)}): {categorical_cols}")

print("\nNumeric summary:\n", df.describe(include='number').T)
print("\nCategorical summary:\n", df.describe(include='object').T)

# ============================================================
# MISSING-VALUE ANALYSIS
# ============================================================
missing = df.isnull().sum()
print("\nMissing values:\n", missing[missing > 0] if missing.sum() else "No explicit NaNs.")

plt.figure(figsize=(10, 4))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
plt.title('Missing value map')
plt.show()

print("Age == 0:", (df['Age'] == 0).sum())
print("DayOfWeekClaimed == '0':", (df['DayOfWeekClaimed'] == '0').sum())
print("MonthClaimed == '0':", (df['MonthClaimed'] == '0').sum())

# ============================================================
# DUPLICATE ANALYSIS
# ============================================================
print("\nFully duplicated rows:", df.duplicated().sum())
print("Duplicated ignoring PolicyNumber:", df.drop(columns=ID_COLS).duplicated().sum())

# ============================================================
# CLASS DISTRIBUTION
# ============================================================
counts = df[TARGET].value_counts()
pct = df[TARGET].value_counts(normalize=True) * 100
print("\n", pd.DataFrame({'count': counts, 'pct': pct.round(2)}))

sns.countplot(x=TARGET, data=df)
plt.xticks([0, 1], ['Legitimate', 'Fraud'])
plt.title('Fraud vs Legitimate')
plt.show()

# ============================================================
# FEATURE DISTRIBUTIONS
# ============================================================
df[numeric_cols].hist(figsize=(14, 10), bins=20)
plt.tight_layout()
plt.show()

n = len(categorical_cols)
ncols = 4
nrows = -(-n // ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.5))
for ax, col in zip(axes.flatten(), categorical_cols):
    df[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(col, fontsize=9)
    ax.tick_params(axis='x', labelrotation=45)
for ax in axes.flatten()[n:]:
    ax.axis('off')
plt.tight_layout()
plt.show()

# ============================================================
# CORRELATION MATRIX
# ============================================================
num_corr = df[numeric_cols + [TARGET]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(num_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation - numeric features')
plt.show()

df_enc = df.drop(columns=ID_COLS).copy()
for col in categorical_cols:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

target_corr = df_enc.corr(numeric_only=True)[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print("\nTop 10 |correlation| with target (rough, label-encoded):\n", target_corr.head(10))

# ============================================================
# OUTLIER ANALYSIS
# ============================================================
q1, q3 = df['Age'].quantile([.25, .75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
age_outliers = df[(df['Age'] < lower) | (df['Age'] > upper)]
print(f"\nAge IQR outliers: {len(age_outliers)} (bounds {lower:.1f}-{upper:.1f})")

sns.boxplot(x=df['Age'])
plt.title('Age - boxplot')
plt.show()

z = np.abs(stats.zscore(df['Age']))
print("Age |z| > 3:", (z > 3).sum())

# ============================================================
# CATEGORICAL CARDINALITY
# ============================================================
cardinality = df[categorical_cols].nunique().sort_values(ascending=False)
print("\nCardinality per categorical column:\n", cardinality)
print("\nPolicyNumber unique vs rows:", df['PolicyNumber'].nunique(), "/", len(df))

# ============================================================
# BASIC FEATURE ANALYSIS
# ============================================================
print("\nNumeric feature means by target:\n", df.groupby(TARGET)[numeric_cols].mean().T)

for col in ['Sex', 'VehicleCategory', 'Fault', 'AccidentArea']:
    rate = (df.groupby(col)[TARGET].mean() * 100).sort_values(ascending=False)
    print(f"\nFraud rate (%) by {col}:\n{rate.round(2)}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================
X = df_enc.drop(columns=[TARGET])
y = df_enc[TARGET]

mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns).sort_values(ascending=False)
print("\nTop 10 by mutual information:\n", mi.head(10))

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 by Random Forest importance:\n", imp.head(10))
