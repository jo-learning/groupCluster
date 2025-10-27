import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_json("dataset.json")

# --- ENCODING ---
mlb_services = MultiLabelBinarizer()
services_encoded = mlb_services.fit_transform(data["desiredServices"])

mlb_goals = MultiLabelBinarizer()
goals_encoded = mlb_goals.fit_transform(data["goals"])

mlb_lang = MultiLabelBinarizer()
langs_encoded = mlb_lang.fit_transform(data["languages"])

levels_map = {
    "beginner": 1,
    "recreational": 2,
    "high school player": 3,
    "college player": 4,
    "tournament player": 5,
    "professional": 6
}
data["level_num"] = data["level"].map(levels_map)

# Combine features
features = pd.DataFrame({
    "rank": data["rank"].fillna(0),
    "budget": data["maxBudgetPerSession"].fillna(0),
    "travel": data["travelDistance"].fillna(0),
    "level": data["level_num"].fillna(1)
})

# Combine all encoded features
X = np.concatenate(
    [features.values, services_encoded, goals_encoded, langs_encoded],
    axis=1
)

# --- SCALING ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- CLUSTERING ---
kmeans = KMeans(n_clusters=4, random_state=42)
data["cluster"] = kmeans.fit_predict(X_scaled)

# --- SAVE MODEL ---
joblib.dump(kmeans, "player_cluster_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump({
    "services": mlb_services,
    "goals": mlb_goals,
    "languages": mlb_lang
}, "encoders.pkl")

# --- SAVE RESULTS ---
data[["id", "cluster"]].to_json("player_clusters.json", orient="records")

print("✅ Model trained and saved as player_cluster_model.pkl")
print(data[["id", "cluster"]])
