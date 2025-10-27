from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
from flask_cors import CORS
import pandas as pd
import uuid

app = Flask(__name__)
CORS(app)

# Initialize Flask-RESTX API
api = Api(app, 
          version='1.0', 
          title='Tennis Player Clustering API',
          description='API for clustering tennis players based on their attributes',
          doc='/swagger/')

# Namespaces
players_ns = api.namespace('players', description='Player operations')
clustering_ns = api.namespace('clustering', description='Clustering operations')

# Load model and encoders
model = joblib.load("player_cluster_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
data = pd.read_json("dataset.json")

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

# Store players in memory
players_db = []

# Request/Response Models
player_model = api.model('Player', {
    'name': fields.String(required=True, description='Player name'),
    'level': fields.String(required=True, description='Skill level', 
                          enum=['beginner', 'recreational', 'high school player', 
                                'college player', 'tournament player', 'professional']),
    'rank': fields.Integer(required=True, description='Player rank (0-100)', min=0, max=100),
    'maxBudgetPerSession': fields.Integer(required=True, description='Max budget per session ($)'),
    'travelDistance': fields.Integer(required=True, description='Travel distance (miles)'),
    'desiredServices': fields.List(fields.String, required=True, description='Desired services'),
    'goals': fields.List(fields.String, required=True, description='Player goals'),
    'languages': fields.List(fields.String, required=True, description='Spoken languages')
})

cluster_request_model = api.model('ClusterRequest', {
    'level': fields.String(required=True, description='Skill level',
                          enum=['beginner', 'recreational', 'high school player', 
                                'college player', 'tournament player', 'professional']),
    'rank': fields.Integer(required=True, description='Player rank', min=0, max=100),
    'maxBudgetPerSession': fields.Integer(required=True, description='Max budget per session'),
    'travelDistance': fields.Integer(required=True, description='Travel distance'),
    'desiredServices': fields.List(fields.String, required=True, description='Desired services'),
    'goals': fields.List(fields.String, required=True, description='Player goals'),
    'languages': fields.List(fields.String, required=True, description='Spoken languages')
})

recommended_player_model = api.model('RecommendedPlayer', {
    'id': fields.String(description='Player ID'),
    'level': fields.String(description='Skill level'),
    'rank': fields.Integer(description='Player rank'),
    'maxBudgetPerSession': fields.Integer(description='Max budget per session')
})

cluster_response_model = api.model('ClusterResponse', {
    'cluster': fields.Integer(description='Assigned cluster number'),
    'recommendedPlayers': fields.List(fields.Nested(recommended_player_model))
})

cluster_result_model = api.model('ClusterResult', {
    'id': fields.String(description='Player ID'),
    'cluster': fields.Integer(description='Assigned cluster number')
})

player_response_model = api.model('PlayerResponse', {
    'id': fields.String(description='Player ID'),
    'message': fields.String(description='Response message')
})

message_model = api.model('Message', {
    'message': fields.String(description='Response message')
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message')
})

@players_ns.route('/')
class PlayerList(Resource):
    @players_ns.marshal_list_with(player_model)
    @players_ns.doc(description='Get all players')
    def get(self):
        """Get all players"""
        return players_db

    @players_ns.expect(player_model)
    @players_ns.marshal_with(player_response_model, code=201)
    @players_ns.response(400, 'Bad Request', error_model)
    def post(self):
        """Add a new player"""
        player = request.json
        player["id"] = str(uuid.uuid4())
        players_db.append(player)
        return {"id": player["id"], "message": "Player added successfully"}, 201

@players_ns.route('/<string:player_id>')
@players_ns.param('player_id', 'The player identifier')
class Player(Resource):
    @players_ns.marshal_with(message_model)
    @players_ns.response(404, 'Player not found', error_model)
    def delete(self, player_id):
        """Delete a player by ID"""
        global players_db
        initial_count = len(players_db)
        players_db = [p for p in players_db if p["id"] != player_id]
        
        if len(players_db) == initial_count:
            return {"error": "Player not found"}, 404
            
        return {"message": "Player deleted successfully"}

@clustering_ns.route('/cluster-player')
class ClusterPlayer(Resource):
    @clustering_ns.expect(cluster_request_model)
    @clustering_ns.marshal_with(cluster_response_model)
    @clustering_ns.response(400, 'Bad Request', error_model)
    def post(self):
        """Cluster a single player"""
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

            return {
                "cluster": cluster,
                "recommendedPlayers": similar_players
            }
        except Exception as e:
            return {"error": str(e)}, 400

@clustering_ns.route('/cluster-all')
class ClusterAllPlayers(Resource):
    @clustering_ns.expect([cluster_request_model])
    @clustering_ns.marshal_list_with(cluster_result_model)
    @clustering_ns.response(400, 'Bad Request', error_model)
    def post(self):
        """Cluster multiple players"""
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
            
            return results
        except Exception as e:
            return {"error": str(e)}, 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)