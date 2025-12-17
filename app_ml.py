from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model
import traceback

app = Flask(__name__)
CORS(app)

class PredictionService:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.sequence_length = 60
        self.load_models()
    
    def load_models(self):
        """Charge le modèle unifié"""
        try:
            # Charger depuis les fichiers locaux ou Google Drive
            if os.path.exists('unified_model.h5'):
                self.model = load_model('unified_model.h5')
                with open('scaler_unified.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✅ Modèle chargé avec succès!")
            else:
                print("❌ Fichiers modèle non trouvés")
                
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
    
    def prepare_features(self, df):
        """Prépare les features"""
        df = df.copy()
        df = df.sort_values('timestamp')
        
        df['returns'] = df['close'].pct_change()
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['volatility'] = df['returns'].rolling(window=20, min_periods=1).std()
        
        return df.fillna(method='bfill').fillna(method='ffill')
    
    def calculate_rsi(self, prices, period=14):
        """RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def predict(self, data):
        """Fait les prédictions"""
        try:
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            df_processed = self.prepare_features(df)
            feature_columns = ['close', 'open', 'high', 'low', 'volume', 
                              'returns', 'sma_5', 'sma_20', 'rsi', 'volatility']
            
            data_array = df_processed[feature_columns].values
            data_scaled = self.scaler.transform(data_array)
            
            if len(data_scaled) < self.sequence_length:
                return {
                    'error': f'Données insuffisantes. Min: {self.sequence_length}'
                }
            
            last_sequence = data_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            current_price = float(df_processed['close'].iloc[-1])
            
            # Prédiction
            predictions = self.model.predict(last_sequence, verbose=0)
            
            result = {
                'asset': data[0].get('asset', 'Unknown'),
                'current_price': current_price,
                'timestamp': df_processed['timestamp'].iloc[-1].isoformat(),
                'horizons': []
            }
            
            horizons_data = [
                (1, "1 jour", predictions[0][0][0]),
                (7, "1 semaine", predictions[1][0][0]),
                (30, "1 mois", predictions[2][0][0]),
                (365, "1 an", predictions[3][0][0])
            ]
            
            for days, name, prob in horizons_data:
                prob = float(prob)
                direction = "HAUSSE" if prob > 0.5 else "BAISSE"
                confidence = prob if prob > 0.5 else (1 - prob)
                
                result['horizons'].append({
                    'days': days,
                    'name': name,
                    'direction': direction,
                    'probability_up': round(prob * 100, 2),
                    'probability_down': round((1 - prob) * 100, 2),
                    'confidence': round(confidence * 100, 2)
                })
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'trace': traceback.format_exc()}

# Instance globale
prediction_service = PredictionService()

# Endpoints
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': prediction_service.model is not None,
        'server': 'ML Server (Render)'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Format invalide'}), 400
        
        data = request_data['data']
        
        if len(data) < 60:
            return jsonify({
                'error': f'Données insuffisantes. Min: 60'
            }), 400
        
        result = prediction_service.predict(data)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
