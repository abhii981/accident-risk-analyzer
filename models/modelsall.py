import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix,
                           roc_auc_score)
import warnings
import joblib
import os
import json
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')

print("=" * 60)
print("🚀 INDIAN ROAD ACCIDENT RISK ANALYZER - MODEL TRAINING")
print("=" * 60)

# 1. Load and preprocess data
print("\n📊 Loading dataset...")
df = pd.read_csv('data/indian_roads_dataset.csv')
df['festival'] = df['festival'].fillna('none')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# 2. Feature engineering
print("🔧 Engineering features...")
cat_cols = ['city', 'state', 'weather', 'road_type', 'visibility', 
            'traffic_density', 'festival', 'cause', 'day_of_week']
le = LabelEncoder()
for col in cat_cols:
    df[col + '_enc'] = le.fit_transform(df[col])

severity_map = {'minor': 0, 'major': 1, 'fatal': 2}
df['severity_label'] = df['accident_severity'].map(severity_map)

# 3. Define features
FEATURES = [
    'hour', 'is_weekend', 'is_peak_hour', 'temperature', 'lanes', 'traffic_signal',
    'vehicles_involved', 'casualties', 'risk_score', 'month', 'weather_enc', 'road_type_enc',
    'visibility_enc', 'traffic_density_enc', 'festival_enc', 'cause_enc', 'day_of_week_enc',
    'city_enc', 'state_enc'
]

X = df[FEATURES]
y = df['severity_label']

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# 6. Dictionary to store all metrics
all_metrics = {}

# 7. Train models with comprehensive evaluation
def evaluate_model(model, X_test, y_test, model_name):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    
    # Calculate all metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_macro': f1_score(y_test, y_pred, average='macro'),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted'),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted'),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted')
    }
    
    # ROC-AUC for multi-class
    try:
        y_pred_proba = model.predict_proba(X_test)
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except:
        metrics['roc_auc'] = None
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics

# Train Random Forest
print("\n" + "=" * 45)
print("🌲 Random Forest Classifier")
print("=" * 45)
rf = RandomForestClassifier(
    n_estimators=200,
    class_weight={0: 1, 1: 3, 2: 2},
    random_state=42,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['minor', 'major', 'fatal']))
all_metrics['Random Forest'] = evaluate_model(rf, X_test, y_test, 'Random Forest')

# Train XGBoost
print("\n" + "=" * 45)
print("⚡ XGBoost Classifier")
print("=" * 45)
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb, target_names=['minor', 'major', 'fatal']))
all_metrics['XGBoost'] = evaluate_model(xgb, X_test, y_test, 'XGBoost')

# Train KNN
print("\n" + "=" * 45)
print("📍 K-Nearest Neighbors")
print("=" * 45)
knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
knn.fit(X_train_sc, y_train)
y_pred_knn = knn.predict(X_test_sc)
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_knn, target_names=['minor', 'major', 'fatal']))
all_metrics['KNN'] = evaluate_model(knn, X_test_sc, y_test, 'KNN')

# Train Logistic Regression
print("\n" + "=" * 45)
print("📈 Logistic Regression")
print("=" * 45)
lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=['minor', 'major', 'fatal']))
all_metrics['Logistic Regression'] = evaluate_model(lr, X_test_sc, y_test, 'Logistic Regression')

# Advanced Clustering (DBSCAN + KMeans)
print("\n" + "=" * 45)
print("🗺️ Advanced Clustering Analysis")
print("=" * 45)

# KMeans clustering
kmeans_features = ['latitude', 'longitude', 'risk_score', 'hour', 'traffic_density_enc']
X_km = df[kmeans_features]
scaler_km = StandardScaler()
X_km_sc = scaler_km.fit_transform(X_km)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
df['cluster_kmeans'] = kmeans.fit_predict(X_km_sc)

# DBSCAN clustering for anomaly detection
dbscan = DBSCAN(eps=0.3, min_samples=10)
df['cluster_dbscan'] = dbscan.fit_predict(X_km_sc)

print(f"KMeans Clusters: {len(df['cluster_kmeans'].unique())}")
print(f"DBSCAN Clusters: {len(df[df['cluster_dbscan'] != -1]['cluster_dbscan'].unique())}")
print(f"DBSCAN Noise Points: {(df['cluster_dbscan'] == -1).sum()}")

# Feature Importance Analysis
print("\n" + "=" * 45)
print("🎯 Feature Importance Analysis")
print("=" * 45)
feature_importance = pd.DataFrame({
    'feature': FEATURES,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Cross-validation scores
print("\n" + "=" * 45)
print("🔄 Cross-Validation Scores (5-fold)")
print("=" * 45)
cv_scores = cross_val_score(rf, X, y, cv=5, scoring='accuracy')
print(f"Random Forest CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Save all models and metrics
print("\n" + "=" * 45)
print("💾 Saving Models and Metrics")
print("=" * 45)

os.makedirs('models', exist_ok=True)

# Save models
joblib.dump(rf, 'models/rf_model.pkl')
joblib.dump(xgb, 'models/xgb_model.pkl')
joblib.dump(knn, 'models/knn_model.pkl')
joblib.dump(lr, 'models/lr_model.pkl')
joblib.dump(kmeans, 'models/kmeans_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(scaler_km, 'models/scaler_km.pkl')

# Save feature importance
feature_importance.to_csv('models/feature_importance.csv', index=False)

# Save metrics
with open('models/model_metrics.json', 'w') as f:
    json.dump(all_metrics, f, indent=2)

# Save clustered data
clustered_data = df[['latitude', 'longitude', 'cluster_kmeans', 'cluster_dbscan', 
                     'accident_severity', 'risk_score', 'city']]
joblib.dump(clustered_data, 'models/clustered_data.pkl')

# Save feature names for later use
joblib.dump(FEATURES, 'models/feature_names.pkl')

print("\n✅ All models and metrics saved successfully!")
print(f"📁 Models saved in: {os.path.abspath('models')}")
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)

# Summary Statistics
print("\n📊 MODEL PERFORMANCE SUMMARY")
print("-" * 60)
for model_name, metrics in all_metrics.items():
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision_weighted']:.4f}")
    print(f"  Recall:    {metrics['recall_weighted']:.4f}")
    print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
    if metrics['roc_auc']:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")