import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, classification_report
import pickle

# ============================================
# LOAD DATASET
# ============================================
df = pd.read_csv("online_course_engagement_data.csv")
df = df.copy()
df.drop_duplicates(inplace=True)

# Drop UserID (not used)
if "UserID" in df.columns:
    df.drop("UserID", axis=1, inplace=True)

# Feature/Target
X = df.drop("CourseCompletion", axis=1)
y = df["CourseCompletion"]

# ============================================
# TRAIN/TEST SPLIT
# ============================================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_cols = [
    'TimeSpentOnCourse',
    'NumberOfVideosWatched',
    'NumberOfQuizzesTaken',
    'QuizScores',
    'CompletionRate'
]

categorical_cols = ['CourseCategory']

# ============================================
# PREPROCESSING
# ============================================
preprocess = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# ============================================
# MODEL PIPELINE
# ============================================
model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('rf', RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=4,
        min_samples_split=2,
        random_state=42
    ))
])

# Train
model.fit(X_train_raw, y_train)

# ============================================
# EVALUATION
# ============================================
y_pred = model.predict(X_test_raw)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall kelas 0:", recall_score(y_test, y_pred, pos_label=0))
print(classification_report(y_test, y_pred))

# ============================================
# SAVE MODEL
# ============================================
with open("course_completion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan ke course_completion_model.pkl")