from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load("player_cluster_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

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
    data = request.json
    try:
        # Encode categorical lists
        services_encoded = encoders["services"].transform([data["desiredServices"]])
        goals_encoded = encoders["goals"].transform([data["goals"]])
        lang_encoded = encoders["languages"].transform([data["languages"]])

        # Numeric fields
        level = levels_map.get(data["level"], 1)
        rank = data.get("rank", 0)
        budget = data.get("maxBudgetPerSession", 0)
        travel = data.get("travelDistance", 0)

        # Combine all features
        X = np.concatenate(
            [[rank, budget, travel, level], services_encoded[0], goals_encoded[0], lang_encoded[0]]
        ).reshape(1, -1)

        # Scale and predict
        X_scaled = scaler.transform(X)
        cluster = int(model.predict(X_scaled)[0])

        return jsonify({"cluster": cluster})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
