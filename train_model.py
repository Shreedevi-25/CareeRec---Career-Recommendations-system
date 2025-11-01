import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import warnings
import time
import re  # <-- IMPORT REGULAR EXPRESSIONS

# --- Configuration ---

DATA_FILE = 'C:/Users/91982/OneDrive/Desktop/COLLEGE/PROJECT/preprocessed_dataset.csv'
MODEL_FILE = 'career_model.pkl'

# --- Define Feature Lists ---

TARGET_COLUMN = 'Title'
NUMERIC_FEATURES = ['YearsOfExperience']
CATEGORICAL_FEATURES = ['ExperienceLevel']

# --- !!!!!!!!!!! NEW HELPER FUNCTION !!!!!!!!!!! ---
def clean_experience(value):
    """
    Cleans the 'YearsOfExperience' column.
    Converts strings like '0-1' or '5+' to a single float.
    """
    value_str = str(value).strip()
    
    # Find the first number (integer or float)
    match = re.search(r'(\d+\.?\d*)', value_str)
    
    if match:
        return float(match.group(1))
    else:
        # If no number is found (e.g., 'Fresher' or empty string), default to 0
        return 0.0
# --- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---


# --- Main Functions ---

def load_and_prepare_data(filepath):
    """
    Loads data, cleans it, and performs feature engineering.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"--- ERROR: Data file '{filepath}' not found. ---")
        return None, None, None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None, None, None

    # Drop JobID
    if 'JobID' in df.columns:
        df = df.drop('JobID', axis=1)

    # FIX #1: Drop rows with missing 'Title'
    initial_rows = len(df)
    df = df.dropna(subset=[TARGET_COLUMN])
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows due to missing 'Title'.")
    
    # FIX #2: Drop rows with rare 'Title' classes
    class_counts = df[TARGET_COLUMN].value_counts()
    MIN_SAMPLES = 5 
    titles_to_keep = class_counts[class_counts >= MIN_SAMPLES].index
    initial_rows = len(df)
    initial_unique_titles = len(class_counts)
    df = df[df[TARGET_COLUMN].isin(titles_to_keep)]
    rows_dropped = initial_rows - len(df)
    titles_dropped = initial_unique_titles - len(titles_to_keep)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows and {titles_dropped} unique titles due to rarity (less than {MIN_SAMPLES} samples).")

    # --- !!!!!!!!!!! NEW FIX IS HERE !!!!!!!!!!! ---
    #
    # FIX #3: Clean the 'YearsOfExperience' column
    #
    print(f"Cleaning '{NUMERIC_FEATURES[0]}' column...")
    df[NUMERIC_FEATURES[0]] = df[NUMERIC_FEATURES[0]].apply(clean_experience)
    # --- !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ---


    # --- Text Feature Engineering ---
    skill_cols = [col for col in df.columns if col.startswith('Skill_')]
    text_cols = ['Responsibilities', 'Keywords'] + skill_cols
    df[text_cols] = df[text_cols].fillna('')
    print("Combining all text features (Responsibilities, Keywords, Skills) into 'master_text'...")
    df['master_text'] = df[text_cols].apply(lambda x: ' '.join(x), axis=1)
    
    # --- Define X (features) and y (target) ---
    features_to_use = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ['master_text']
    
    for col in features_to_use:
        if col not in df.columns and col != 'master_text':
            print(f"--- ERROR: Missing expected column: {col} ---")
            return None, None, None, None
            
    if TARGET_COLUMN not in df.columns:
        print(f"--- ERROR: Missing target column: {TARGET_COLUMN} ---")
        return None, None, None, None

    X = df[features_to_use]
    y = df[TARGET_COLUMN]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Data loaded and split: {len(X_train)} training samples, {len(X_test)} test samples.")
    
    return X_train, X_test, y_train, y_test

def build_pipeline():
    """
    Defines and builds the full scikit-learn pipeline.
    """
    print("Building preprocessing and model pipeline...")
    
    # 1. Numeric
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # 2. Categorical
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 3. Text
    text_transformer = Pipeline(steps=[
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000))
    ])

    # Combine preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('text', text_transformer, 'master_text')
        ],
        remainder='drop'
    )

    # --- Model Building: StackingClassifier ---
    
    # Base Models (Level 0)
    base_models = [
        ('knn', KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=100, n_jobs=-1)),
        ('log_reg_base', LogisticRegression(solver='liblinear', random_state=42))
    ]

    # Meta-Model (Level 1)
    meta_model = LogisticRegression(solver='saga', max_iter=1000, random_state=42, n_jobs=-1)

    # Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    # --- Create the Final, Full Pipeline ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('stacker', stacking_model)
    ])
    
    print("Pipeline built successfully.")
    return pipeline

def main():
    """
    Main function to run the training and evaluation process.
    """
    warnings.filterwarnings('ignore')
    start_time = time.time()

    # 1. Load and prepare data
    X_train, X_test, y_train, y_test = load_and_prepare_data(DATA_FILE)
    if X_train is None:
        return # Stop execution if data loading failed

    # 2. Build the pipeline
    model_pipeline = build_pipeline()

    # 3. Train the model
    print("Training model... (This may take several minutes)...")
    model_pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 4. Evaluate the model
    print("\n--- Model Evaluation ---")
    y_pred = model_pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Weighted F1-Score: {f1:.4f}  <-- This is your key metric!")
    
    print("\nFull Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 5. Save the model
    print(f"\nSaving the trained pipeline to '{MODEL_FILE}'...")
    joblib.dump(model_pipeline, MODEL_FILE)
    
    end_time = time.time()
    print(f"\nModel saved successfully. Phase 1 is complete!")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()