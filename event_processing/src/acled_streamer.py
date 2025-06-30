"""
ACLED (Armed Conflict Location & Event Data Project) real-time data streamer.
"""
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class ACLEDStreamer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.acleddata.com/acled/read"
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def get_recent_events(self, hours: int = 24) -> List[Dict]:
        """Get events from the last N hours."""
        if not self.session:
            raise RuntimeError("Streamer must be used as async context manager")
            
        start_date = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d")
        params = {
            "key": self.api_key,
            "start_date": start_date,
            "limit": 500,
            "terms": "accept"
        }
        
        async with self.session.get(self.base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("data", [])
            else:
                raise Exception(f"ACLED API error: {response.status}")
                
    async def stream_events(self, callback, interval_seconds: int = 300):
        """Stream events continuously, checking every interval_seconds."""
        async with self:
            last_event_time = datetime.now() - timedelta(hours=24)
            while True:
                try:
                    events = await self.get_recent_events()
                    new_events = [
                        event for event in events
                        if datetime.strptime(event["event_date"], "%Y-%m-%d") > last_event_time
                    ]
                    
                    if new_events:
                        last_event_time = max(
                            datetime.strptime(event["event_date"], "%Y-%m-%d")
                            for event in new_events
                        )
                        for event in new_events:
                            await callback(event)
                            
                    await asyncio.sleep(interval_seconds)
                except Exception as e:
                    print(f"Error streaming events: {e}")
                    await asyncio.sleep(interval_seconds)

# Example usage
async def process_event(event: Dict):
    """Process each new ACLED event."""
    print(f"New event: {event['event_type']} in {event['location']}")
    
async def main():
    streamer = ACLEDStreamer("your-api-key")
    await streamer.stream_events(process_event)
