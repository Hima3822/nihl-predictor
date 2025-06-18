import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

df = pd.read_csv(r"C:\Users\friend\Downloads\sd-1003-2019-0\sd-1003-2019-0\csv\paper3.csv")

df.replace([997, 998, 999], np.nan, inplace=True)
df.dropna(subset=['L3k', 'L4k', 'L6k', 'R3k', 'R4k', 'R6k'], inplace=True)

hearing_cols = ['L3k', 'L4k', 'L6k', 'R3k', 'R4k', 'R6k']
for col in hearing_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Left_avg'] = df[['L3k', 'L4k', 'L6k']].mean(axis=1)
df['Right_avg'] = df[['R3k', 'R4k', 'R6k']].mean(axis=1)
df['Avg_Hearing_Threshold'] = df[['Left_avg', 'Right_avg']].mean(axis=1)
df['Hearing_Loss'] = (df['Avg_Hearing_Threshold'] > 25).astype(int)

le = LabelEncoder()
label_enc_cols = ['gender', 'naics', 'age_group', 'region', 'NAICS_descr']
for col in label_enc_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

features = [
    'age_group', 'gender', 'region', 'naics',
    'L3k', 'L4k', 'L6k', 'R3k', 'R4k', 'R6k'
]
X = df[features]
y = df['Hearing_Loss']

X = X.dropna()
y = y.loc[X.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

y_pred = model.predict(X_test)

results_df = X_test.copy()
results_df['Actual'] = y_test.values
results_df['Predicted'] = y_pred

results_df['Actual'] = results_df['Actual'].map({0: 'No', 1: 'Yes'})
results_df['Predicted'] = results_df['Predicted'].map({0: 'No', 1: 'Yes'})

print(results_df.head(10))
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()


import joblib
joblib.dump(model, 'hearing_loss_model.pkl')
model = joblib.load("hearing_loss_model.pkl")  



