"""
Real-time anomaly detection for crisis events using statistical and ML methods.
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class AnomalyDetector:
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.event_buffer: List[Dict] = []
        self.buffer_size = 1000
        
    def _extract_features(self, event: Dict) -> np.ndarray:
        """Extract numerical features from an event."""
        features = [
            event.get('fatalities', 0),
            event.get('latitude', 0),
            event.get('longitude', 0),
            len(event.get('notes', '')),  # Text length as a simple feature
            1 if 'violence' in event.get('event_type', '').lower() else 0,
            1 if 'protest' in event.get('event_type', '').lower() else 0
        ]
        return np.array(features).reshape(1, -1)
        
    def _update_model(self):
        """Update the anomaly detection model with current buffer."""
        if len(self.event_buffer) > 50:  # Minimum events needed
            features = np.vstack([
                self._extract_features(event)[0]
                for event in self.event_buffer
            ])
            self.scaler.fit(features)
            normalized_features = self.scaler.transform(features)
            self.isolation_forest.fit(normalized_features)
            
    async def process_event(self, event: Dict) -> Tuple[Dict, bool]:
        """Process a new event and detect if it's anomalous."""
        # Add to buffer
        self.event_buffer.append(event)
        if len(self.event_buffer) > self.buffer_size:
            self.event_buffer.pop(0)
            
        # Update model periodically
        if len(self.event_buffer) % 100 == 0:
            self._update_model()
            
        # Detect anomaly
        if len(self.event_buffer) > 50:
            features = self._extract_features(event)
            normalized = self.scaler.transform(features)
            is_anomaly = self.isolation_forest.predict(normalized)[0] == -1
            
            if is_anomaly:
                event['anomaly_detected'] = True
                event['anomaly_score'] = float(
                    self.isolation_forest.score_samples(normalized)[0]
                )
            return event, is_anomaly
            
        return event, False
        
    def get_recent_anomalies(self, hours: int = 24) -> List[Dict]:
        """Get anomalous events from the last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.event_buffer
            if event.get('anomaly_detected', False)
            and datetime.strptime(event['event_date'], '%Y-%m-%d') > cutoff_time
        ]

# Example usage
async def main():
    detector = AnomalyDetector()
    
    # Example event
    event = {
        'event_type': 'Violence against civilians',
        'fatalities': 5,
        'latitude': 31.5,
        'longitude': 34.75,
        'notes': 'Significant escalation in civilian area',
        'event_date': datetime.now().strftime('%Y-%m-%d')
    }
    
    processed_event, is_anomaly = await detector.process_event(event)
    if is_anomaly:
        print(f"Anomaly detected! Score: {processed_event['anomaly_score']}")
