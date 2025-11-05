
# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# 1. CREATE STUDENT DATASET
print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION USING DECISION TREE")
print("=" * 70)
print("\n1. Creating Student Dataset...")

# Create sample student data
np.random.seed(42)
n_students = 200

# Generate student data
study_hours = np.random.randint(1, 10, n_students)
attendance = np.random.randint(40, 100, n_students)
previous_marks = np.random.randint(30, 100, n_students)
sleep_hours = np.random.randint(4, 10, n_students)
extracurricular = np.random.randint(0, 2, n_students)  # 0=No, 1=Yes

# Create target variable (Pass/Fail) based on conditions
result = []
for i in range(n_students):
    score = 0
    # Study hours contribution
    if study_hours[i] >= 6:
        score += 3
    elif study_hours[i] >= 4:
        score += 2
    else:
        score += 1
    
    # Attendance contribution
    if attendance[i] >= 75:
        score += 3
    elif attendance[i] >= 60:
        score += 2
    else:
        score += 1
    
    # Previous marks contribution
    if previous_marks[i] >= 70:
        score += 3
    elif previous_marks[i] >= 50:
        score += 2
    else:
        score += 1
    
    # Sleep hours contribution
    if 6 <= sleep_hours[i] <= 8:
        score += 1
    
    # Extracurricular contribution
    if extracurricular[i] == 1:
        score += 1
    
    # Determine Pass/Fail (threshold: 7)
    if score >= 7:
        result.append('Pass')
    else:
        result.append('Fail')

# Create DataFrame
student_data = pd.DataFrame({
    'Study_Hours': study_hours,
    'Attendance_%': attendance,
    'Previous_Marks': previous_marks,
    'Sleep_Hours': sleep_hours,
    'Extracurricular': extracurricular,
    'Result': result
})

print(f"\nTotal Students: {len(student_data)}")
print(f"Number of Features: 5")
print("\nFeatures:")
print("  1. Study_Hours (hours per day)")
print("  2. Attendance_% (percentage)")
print("  3. Previous_Marks (out of 100)")
print("  4. Sleep_Hours (hours per day)")
print("  5. Extracurricular (0=No, 1=Yes)")

# Display first 10 students
print("\nFirst 10 Students Data:")
print(student_data.head(10))

# Check class distribution
print("\nStudent Performance Distribution:")
print(student_data['Result'].value_counts())
print(f"\nPass Rate: {(student_data['Result'] == 'Pass').sum() / len(student_data) * 100:.1f}%")
print(f"Fail Rate: {(student_data['Result'] == 'Fail').sum() / len(student_data) * 100:.1f}%")

# Statistics
print("\nDataset Statistics:")
print(student_data.describe())

# 2. PREPARE DATA FOR TRAINING
print("\n" + "=" * 70)
print("2. Preparing Data for Machine Learning...")
print("=" * 70)

# Separate features (X) and target (y)
X = student_data.drop('Result', axis=1)
y = student_data['Result']

# Split data: 75% training, 25% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"\nTraining Set: {len(X_train)} students")
print(f"Testing Set: {len(X_test)} students")
print(f"\nTraining Set - Pass: {(y_train == 'Pass').sum()}, Fail: {(y_train == 'Fail').sum()}")
print(f"Testing Set - Pass: {(y_test == 'Pass').sum()}, Fail: {(y_test == 'Fail').sum()}")

# 3. TRAIN DECISION TREE MODEL
print("\n" + "=" * 70)
print("3. Training Decision Tree Model...")
print("=" * 70)

# Create Decision Tree Classifier
dt_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    criterion='gini'
)

# Train the model
dt_model.fit(X_train, y_train)
print("\n✓ Model Training Completed!")

# Display tree information
print(f"\nDecision Tree Details:")
print(f"  Tree Depth: {dt_model.get_depth()}")
print(f"  Number of Leaves: {dt_model.get_n_leaves()}")
print(f"  Total Nodes: {dt_model.tree_.node_count}")

# 4. MAKE PREDICTIONS
print("\n" + "=" * 70)
print("4. Making Predictions on Test Data...")
print("=" * 70)

# Predict on test set
y_pred = dt_model.predict(X_test)

# Show sample predictions
print("\nSample Predictions (First 15 students):")
print(f"{'Study_Hrs':<10} {'Attend%':<10} {'Prev_Marks':<12} {'Sleep_Hrs':<11} {'Extra':<8} {'Actual':<8} {'Predicted':<10} {'Status':<8}")
print("-" * 90)

for i in range(min(15, len(X_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    status = "✓" if actual == predicted else "✗"
    print(f"{X_test.iloc[i]['Study_Hours']:<10} {X_test.iloc[i]['Attendance_%']:<10} "
          f"{X_test.iloc[i]['Previous_Marks']:<12} {X_test.iloc[i]['Sleep_Hours']:<11} "
          f"{X_test.iloc[i]['Extracurricular']:<8} {actual:<8} {predicted:<10} {status:<8}")

# 5. EVALUATE MODEL PERFORMANCE
print("\n" + "=" * 70)
print("5. Model Evaluation")
print("=" * 70)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Model Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=['Pass', 'Fail'])
print(f"\n                Predicted")
print(f"              Pass    Fail")
print(f"Actual Pass    {cm[0][0]:<7} {cm[0][1]:<7}")
print(f"       Fail    {cm[1][0]:<7} {cm[1][1]:<7}")

# Calculate metrics
true_pass = cm[0][0]
false_fail = cm[0][1]
false_pass = cm[1][0]
true_fail = cm[1][1]

print(f"\nCorrect Predictions: {true_pass + true_fail}")
print(f"Incorrect Predictions: {false_fail + false_pass}")

# Classification Report
print("\n📈 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# 6. VISUALIZE DECISION TREE
print("\n" + "=" * 70)
print("6. Creating Decision Tree Visualization...")
print("=" * 70)

plt.figure(figsize=(25, 12))
tree.plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=['Fail', 'Pass'],
    filled=True,
    rounded=True,
    fontsize=11,
    proportion=True
)
plt.title("Student Performance Decision Tree", fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('student_decision_tree.png', dpi=300, bbox_inches='tight')
print("✓ Decision tree saved as 'student_decision_tree.png'")
plt.show()

# 7. FEATURE IMPORTANCE
print("\n" + "=" * 70)
print("7. Feature Importance Analysis")
print("=" * 70)

# Get feature importance
feature_importance = dt_model.feature_importances_
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance Ranking:")
print(features_df.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features_df)))
plt.barh(features_df['Feature'], features_df['Importance'], color=colors)
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Which Factors Matter Most for Student Success?', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('student_feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Feature importance chart saved as 'student_feature_importance.png'")
plt.show()

# 8. PREDICT FOR NEW STUDENTS
print("\n" + "=" * 70)
print("8. Predicting Performance for New Students")
print("=" * 70)

# Example new students
new_students = pd.DataFrame({
    'Study_Hours': [8, 3, 6, 2],
    'Attendance_%': [90, 55, 75, 45],
    'Previous_Marks': [85, 45, 70, 40],
    'Sleep_Hours': [7, 5, 6, 9],
    'Extracurricular': [1, 0, 1, 0]
})

predictions = dt_model.predict(new_students)
probabilities = dt_model.predict_proba(new_students)

print("\n🎓 New Student Predictions:")
for i in range(len(new_students)):
    print(f"\n--- Student {i+1} ---")
    print(f"Study Hours: {new_students.iloc[i]['Study_Hours']} hrs/day")
    print(f"Attendance: {new_students.iloc[i]['Attendance_%']}%")
    print(f"Previous Marks: {new_students.iloc[i]['Previous_Marks']}/100")
    print(f"Sleep Hours: {new_students.iloc[i]['Sleep_Hours']} hrs/day")
    print(f"Extracurricular: {'Yes' if new_students.iloc[i]['Extracurricular'] == 1 else 'No'}")
    print(f"\n🎯 Prediction: {predictions[i]}")
    print(f"Confidence - Pass: {probabilities[i][1]*100:.1f}%, Fail: {probabilities[i][0]*100:.1f}%")

