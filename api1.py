from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = joblib.load("player_cluster_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
data = pd.read_json("dataset.json")  # original player dataset with cluster info

# Load cluster assignments if they exist
try:
    clusters = pd.read_json("player_clusters.json")
    data = pd.merge(data, clusters, on="id", how="left")
except:
    data["cluster"] = -1

levels_map = {
    "beginner": 1,
    "recreational": 2,
    "high school player": 3,
    "college player": 4,
    "tournament player": 5,
    "professional": 6
}


@app.route("/cluster-player", methods=["POST"])
def cluster_player():
    d = request.json
    try:
        # Encode categorical lists
        services_encoded = encoders["services"].transform([d["desiredServices"]])
        goals_encoded = encoders["goals"].transform([d["goals"]])
        lang_encoded = encoders["languages"].transform([d["languages"]])

        # Numeric fields
        level = levels_map.get(d["level"], 1)
        rank = d.get("rank", 0)
        budget = d.get("maxBudgetPerSession", 0)
        travel = d.get("travelDistance", 0)

        # Combine features
        X = np.concatenate(
            [[rank, budget, travel, level],
             services_encoded[0], goals_encoded[0], lang_encoded[0]]
        ).reshape(1, -1)

        # Scale and predict
        X_scaled = scaler.transform(X)
        cluster = int(model.predict(X_scaled)[0])

        # Find top 5 similar players in same cluster
        similar_players = (
            data[data["cluster"] == cluster][["id", "level", "rank", "maxBudgetPerSession"]]
            .head(5)
            .to_dict(orient="records")
        )

        return jsonify({
            "cluster": cluster,
            "recommendedPlayers": similar_players
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
