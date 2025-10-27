from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd
import uuid
# --- New Imports for Swagger ---
from flasgger import Swagger, swag_from

app = Flask(__name__)
CORS(app)

# --- Swagger Configuration ---
app.config['SWAGGER'] = {
    'title': 'Player Clustering and Management API',
    'uiversion': 3,
    'description': 'API for clustering new players and managing a player database.',
    'version': '1.0.0'
}
swagger = Swagger(app)
# -----------------------------

# Load model and encoders
model = joblib.load("player_cluster_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
# NOTE: In a real app, ensure dataset.json, player_cluster_model.pkl, scaler.pkl, and encoders.pkl exist.
# For this example, assuming they are available.
try:
    data = pd.read_json("dataset.json")  # original player dataset with cluster info
except:
    data = pd.DataFrame(columns=["id", "level", "rank", "maxBudgetPerSession", "cluster"])
    print("Warning: 'dataset.json' not found or failed to load. Using empty DataFrame for recommendations.")


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

# Store players in memory (in production, use a database)
players_db = []

# --- Endpoint Documentation using docstrings ---

@app.route("/cluster-player", methods=["POST"])
@swag_from({
    'tags': ['Clustering'],
    'summary': 'Cluster a single new player and recommend similar existing players.',
    'description': 'Takes a player\'s profile data, determines their cluster, and returns the cluster ID along with the top 5 players from the existing dataset who belong to that same cluster.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'desiredServices': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of services desired (e.g., ["hitting partner", "coaching"]).'},
                    'goals': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of goals (e.g., ["improve serve", "win tournament"]).'},
                    'languages': {'type': 'array', 'items': {'type': 'string'}, 'description': 'List of languages spoken (e.g., ["English", "Spanish"]).'},
                    'level': {'type': 'string', 'enum': list(levels_map.keys()), 'description': 'Player skill level.'},
                    'rank': {'type': 'integer', 'description': 'Player rank (e.g., UTR, ITF).'},
                    'maxBudgetPerSession': {'type': 'number', 'format': 'float', 'description': 'Maximum budget per session.'},
                    'travelDistance': {'type': 'integer', 'description': 'Maximum travel distance in miles/km.'}
                },
                'required': ['desiredServices', 'goals', 'languages', 'level']
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Player successfully clustered and recommendations provided.',
            'schema': {
                'type': 'object',
                'properties': {
                    'cluster': {'type': 'integer', 'description': 'The predicted cluster ID.'},
                    'recommendedPlayers': {'type': 'array', 'items': {'type': 'object'}, 'description': 'List of similar players from the dataset.'}
                }
            }
        },
        '400': {
            'description': 'Invalid input data.',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
})
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

@app.route("/players", methods=["GET", "POST"])
@swag_from({
    'tags': ['Player Management'],
    'summary': 'Retrieve the list of stored players or add a new player.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': False,
            'schema': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string'},
                    'level': {'type': 'string'},
                    # Add other player fields as needed for POST request
                },
            },
            'description': 'Player data for a POST request.'
        }
    ],
    'responses': {
        '200': {'description': 'List of players retrieved (GET) or Player added successfully (POST).'},
    }
})
def manage_players():
    if request.method == "GET":
        return jsonify(players_db)
    
    if request.method == "POST":
        player = request.json
        player["id"] = str(uuid.uuid4())
        players_db.append(player)
        return jsonify({"id": player["id"], "message": "Player added successfully"})

@app.route("/players/<player_id>", methods=["DELETE"])
@swag_from({
    'tags': ['Player Management'],
    'summary': 'Delete a player by ID.',
    'parameters': [
        {
            'name': 'player_id',
            'in': 'path',
            'type': 'string',
            'required': True,
            'description': 'The unique ID of the player to delete.'
        }
    ],
    'responses': {
        '200': {'description': 'Player deleted successfully.'},
    }
})
def delete_player(player_id):
    global players_db
    players_db = [p for p in players_db if p["id"] != player_id]
    return jsonify({"message": "Player deleted successfully"})

@app.route("/cluster-all", methods=["POST"])
@swag_from({
    'tags': ['Clustering'],
    'summary': 'Cluster a list of players in a batch.',
    'description': 'Takes a list of player profiles and returns a list of their IDs and predicted cluster IDs.',
    'parameters': [
        {
            'name': 'body',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string', 'description': 'Player unique ID.'},
                        'desiredServices': {'type': 'array', 'items': {'type': 'string'}},
                        'goals': {'type': 'array', 'items': {'type': 'string'}},
                        'languages': {'type': 'array', 'items': {'type': 'string'}},
                        'level': {'type': 'string', 'enum': list(levels_map.keys())},
                        'rank': {'type': 'integer'},
                        'maxBudgetPerSession': {'type': 'number', 'format': 'float'},
                        'travelDistance': {'type': 'integer'}
                    },
                    'required': ['id', 'desiredServices', 'goals', 'languages', 'level']
                }
            }
        }
    ],
    'responses': {
        '200': {
            'description': 'Batch clustering successful.',
            'schema': {
                'type': 'array',
                'items': {
                    'type': 'object',
                    'properties': {
                        'id': {'type': 'string'},
                        'cluster': {'type': 'integer'}
                    }
                }
            }
        },
        '400': {
            'description': 'Error during batch clustering.',
            'schema': {'type': 'object', 'properties': {'error': {'type': 'string'}}}
        }
    }
})
def cluster_all_players():
    try:
        players = request.json
        results = []
        
        for player in players:
            # Encode categorical lists
            services_encoded = encoders["services"].transform([player["desiredServices"]])
            goals_encoded = encoders["goals"].transform([player["goals"]])
            lang_encoded = encoders["languages"].transform([player["languages"]])

            # Numeric fields
            level = levels_map.get(player["level"], 1)
            rank = player.get("rank", 0)
            budget = player.get("maxBudgetPerSession", 0)
            travel = player.get("travelDistance", 0)

            # Combine features
            X = np.concatenate(
                [[rank, budget, travel, level],
                 services_encoded[0], goals_encoded[0], lang_encoded[0]]
            ).reshape(1, -1)

            # Scale and predict
            X_scaled = scaler.transform(X)
            cluster = int(model.predict(X_scaled)[0])
            
            results.append({
                "id": player["id"],
                "cluster": cluster
            })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6005, debug=True)