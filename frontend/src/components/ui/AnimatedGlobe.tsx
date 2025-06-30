
import React, { useEffect, useRef } from 'react';

const AnimatedGlobe: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Set canvas dimensions
    const setCanvasDimensions = () => {
      const devicePixelRatio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      
      canvas.width = rect.width * devicePixelRatio;
      canvas.height = rect.height * devicePixelRatio;
      
      ctx.scale(devicePixelRatio, devicePixelRatio);
    };
    
    window.addEventListener('resize', setCanvasDimensions);
    setCanvasDimensions();
    
    // Globe parameters
    const width = canvas.width;
    const height = canvas.height;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.min(width, height) * 0.35;
    
    // Points to draw on the globe
    const points: { lat: number; lng: number; intensity: number }[] = [];
    
    // Generate random points for the demo
    for (let i = 0; i < 200; i++) {
      points.push({
        lat: Math.random() * 180 - 90, // -90 to 90
        lng: Math.random() * 360 - 180, // -180 to 180
        intensity: Math.random(),
      });
    }
    
    // Add some "hotspot" clusters to represent crisis regions
    const hotspots = [
      { lat: 5, lng: 20, radius: 15 }, // Africa
      { lat: 30, lng: 70, radius: 15 }, // Middle East
      { lat: 15, lng: -90, radius: 15 }, // Central America
      { lat: -10, lng: 130, radius: 10 }, // Southeast Asia
    ];
    
    for (const hotspot of hotspots) {
      for (let i = 0; i < 30; i++) {
        const angle = Math.random() * Math.PI * 2;
        const distance = Math.random() * hotspot.radius;
        const lat = hotspot.lat + Math.sin(angle) * distance;
        const lng = hotspot.lng + Math.cos(angle) * distance;
        
        points.push({
          lat,
          lng,
          intensity: 0.5 + Math.random() * 0.5, // Higher intensity for hotspots
        });
      }
    }
    
    // Animation variables
    let rotation = 0;
    const rotationSpeed = 0.0005;
    
    // Convert lat/lng to 3D coordinates
    const latLngTo3d = (lat: number, lng: number, radius: number) => {
      const phi = (90 - lat) * (Math.PI / 180);
      const theta = (lng + 180) * (Math.PI / 180);
      
      const x = -radius * Math.sin(phi) * Math.cos(theta + rotation);
      const z = radius * Math.sin(phi) * Math.sin(theta + rotation);
      const y = radius * Math.cos(phi);
      
      return { x, y, z };
    };
    
    // Render function
    const render = () => {
      ctx.clearRect(0, 0, width, height);
      
      // Draw the globe outline
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(180, 180, 200, 0.2)';
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Create an array to sort points by z-index
      const visiblePoints = points.map(point => {
        const { x, y, z } = latLngTo3d(point.lat, point.lng, radius);
        return { x: x + centerX, y: y + centerY, z, intensity: point.intensity };
      }).filter(point => point.z > -radius * 0.2); // Only render points on the visible side
      
      // Sort by z-index for proper rendering
      visiblePoints.sort((a, b) => a.z - b.z);
      
      // Draw connections between nearby points
      ctx.strokeStyle = 'rgba(65, 145, 255, 0.15)';
      ctx.lineWidth = 0.5;
      
      for (let i = 0; i < visiblePoints.length; i++) {
        const pointA = visiblePoints[i];
        
        for (let j = i + 1; j < visiblePoints.length; j++) {
          const pointB = visiblePoints[j];
          
          // Calculate distance
          const dx = pointA.x - pointB.x;
          const dy = pointA.y - pointB.y;
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // Draw connection if points are close enough
          if (distance < radius * 0.4) {
            const alpha = (1 - distance / (radius * 0.4)) * 0.4;
            ctx.strokeStyle = `rgba(65, 145, 255, ${alpha})`;
            
            ctx.beginPath();
            ctx.moveTo(pointA.x, pointA.y);
            ctx.lineTo(pointB.x, pointB.y);
            ctx.stroke();
          }
        }
      }
      
      // Draw points
      for (const point of visiblePoints) {
        const size = 1.5 + point.intensity * 2;
        const alpha = 0.4 + point.intensity * 0.6;
        const normZ = (point.z + radius) / (radius * 2); // Normalize z from -radius to radius to 0 to 1
        
        // Create gradient for point glow
        const gradient = ctx.createRadialGradient(
          point.x, point.y, 0,
          point.x, point.y, size * 4
        );
        
        const color = point.intensity > 0.8 ? [255, 90, 90] : [65, 145, 255];
        
        gradient.addColorStop(0, `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha})`);
        gradient.addColorStop(1, `rgba(${color[0]}, ${color[1]}, ${color[2]}, 0)`);
        
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(point.x, point.y, size * 2, 0, Math.PI * 2);
        ctx.fill();
        
        // Add brighter center
        ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${alpha * 1.5})`;
        ctx.beginPath();
        ctx.arc(point.x, point.y, size * 0.7, 0, Math.PI * 2);
        ctx.fill();
      }
      
      // Rotate the globe
      rotation += rotationSpeed;
      
      requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      window.removeEventListener('resize', setCanvasDimensions);
    };
  }, []);
  
  return (
    <canvas 
      ref={canvasRef} 
      className="w-full h-full"
    />
  );
};

export default AnimatedGlobe;
