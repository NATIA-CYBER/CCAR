"""
Real-time geospatial analysis for crisis events.
Implements spatial clustering, hotspot detection, and risk propagation modeling.
"""
import numpy as np
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
import geopandas as gpd
from shapely.geometry import Point, Polygon

class SpatialAnalyzer:
    def __init__(self):
        self.events_buffer: List[Dict] = []
        self.hotspots: List[Dict] = []
        self.risk_zones: List[Polygon] = []
        self.eps_km = 50  # Clustering radius in kilometers
        self.min_samples = 3  # Minimum events for a cluster
        
    def _haversine_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
        
    def _create_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> np.ndarray:
        """Create a distance matrix for all points using Haversine distance."""
        n = len(coordinates)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                lat1, lon1 = coordinates[i]
                lat2, lon2 = coordinates[j]
                distance = self._haversine_distance(lat1, lon1, lat2, lon2)
                distance_matrix[i,j] = distance_matrix[j,i] = distance
                
        return distance_matrix
        
    async def process_event(self, event: Dict) -> Dict:
        """Process a new event and update spatial analysis."""
        self.events_buffer.append(event)
        
        # Keep buffer size manageable
        if len(self.events_buffer) > 1000:
            self.events_buffer = self.events_buffer[-1000:]
            
        # Update analysis if we have enough events
        if len(self.events_buffer) >= self.min_samples:
            await self._update_spatial_analysis()
            
        # Add spatial context to event
        event = await self._add_spatial_context(event)
        return event
        
    async def _update_spatial_analysis(self):
        """Update spatial clustering and hotspot detection."""
        # Extract coordinates
        coordinates = [
            (event['latitude'], event['longitude'])
            for event in self.events_buffer
            if 'latitude' in event and 'longitude' in event
        ]
        
        if not coordinates:
            return
            
        # Create distance matrix
        distance_matrix = self._create_distance_matrix(coordinates)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(
            eps=self.eps_km,
            min_samples=self.min_samples,
            metric='precomputed'
        ).fit(distance_matrix)
        
        # Update hotspots
        self.hotspots = []
        labels = clustering.labels_
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label != -1:  # Not noise
                cluster_points = np.array(coordinates)[labels == label]
                center = cluster_points.mean(axis=0)
                radius = np.max([
                    self._haversine_distance(
                        center[0], center[1],
                        point[0], point[1]
                    )
                    for point in cluster_points
                ])
                
                self.hotspots.append({
                    'center': center,
                    'radius': radius,
                    'num_events': len(cluster_points),
                    'severity': self._calculate_cluster_severity(
                        [e for e, l in zip(self.events_buffer, labels) if l == label]
                    )
                })
                
    def _calculate_cluster_severity(self, cluster_events: List[Dict]) -> float:
        """Calculate severity score for a cluster based on event properties."""
        if not cluster_events:
            return 0.0
            
        # Factors to consider
        fatalities = sum(event.get('fatalities', 0) for event in cluster_events)
        violence_events = sum(
            1 for event in cluster_events
            if 'violence' in event.get('event_type', '').lower()
        )
        is_anomalous = any(
            event.get('anomaly_detected', False) for event in cluster_events
        )
        
        # Simple severity score
        severity = (fatalities * 0.4 + 
                   violence_events * 0.3 + 
                   (1 if is_anomalous else 0) * 0.3)
        return severity
        
    async def _add_spatial_context(self, event: Dict) -> Dict:
        """Add spatial context to an event based on current analysis."""
        if not ('latitude' in event and 'longitude' in event):
            return event
            
        # Check if event is in any hotspot
        event_location = (event['latitude'], event['longitude'])
        for hotspot in self.hotspots:
            distance = self._haversine_distance(
                event_location[0], event_location[1],
                hotspot['center'][0], hotspot['center'][1]
            )
            if distance <= hotspot['radius']:
                event['in_hotspot'] = True
                event['hotspot_severity'] = hotspot['severity']
                break
                
        return event
        
    def get_active_hotspots(self) -> List[Dict]:
        """Get current active hotspots."""
        return sorted(
            self.hotspots,
            key=lambda x: x['severity'],
            reverse=True
        )

# Example usage
async def main():
    analyzer = SpatialAnalyzer()
    
    # Example event
    event = {
        'latitude': 31.5,
        'longitude': 34.75,
        'event_type': 'Violence against civilians',
        'fatalities': 5
    }
    
    processed_event = await analyzer.process_event(event)
    if processed_event.get('in_hotspot'):
        print(f"Event is in a hotspot with severity: {processed_event['hotspot_severity']}")
        
    hotspots = analyzer.get_active_hotspots()
    print(f"Number of active hotspots: {len(hotspots)}")
