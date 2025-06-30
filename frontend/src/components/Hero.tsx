
import React from 'react';
import { Button } from "@/components/ui/button";
import AnimatedGlobe from './ui/AnimatedGlobe';

const Hero = () => {
  return (
    <section className="relative min-h-screen flex items-center pt-16 overflow-hidden">
      {/* Background elements */}
      <div className="absolute inset-0 bg-gradient-radial from-white to-gray-50 -z-10"></div>
      <div className="absolute w-full h-full top-0 left-0 bg-[url('https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?auto=format&fit=crop&q=80&w=2274&ixlib=rb-4.0.3')] bg-cover bg-center opacity-[0.03] -z-10"></div>
      
      <div className="section-container relative z-10">
        <div className="grid md:grid-cols-2 items-center gap-12">
          <div className="max-w-2xl md:order-1 order-2">
            <div className="space-y-6 animate-slideUp">
              <span className="inline-block px-3 py-1 text-xs font-medium bg-primary/5 text-primary rounded-full">
                Humanitarian Crisis Forecasting
              </span>
              
              <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold tracking-tight leading-tight">
                Anticipate Crises. <br/>
                <span className="text-primary">Save More Lives.</span>
              </h1>
              
              <p className="text-lg text-muted-foreground max-w-lg">
                Empower your humanitarian operations with data-driven forecasting to anticipate crises before they escalate.
              </p>
              
              <div className="flex flex-col sm:flex-row gap-4 pt-4">
                <Button className="bg-primary text-white hover:bg-primary/90 rounded-md px-6 py-6 h-auto">
                  Request Demo
                </Button>
                <Button variant="outline" className="bg-white/80 backdrop-blur-sm rounded-md px-6 py-6 h-auto">
                  Learn More
                </Button>
              </div>
              
              <div className="pt-8 flex items-center gap-6">
                <div className="flex -space-x-2">
                  {[1, 2, 3, 4].map((i) => (
                    <div key={i} className={`w-10 h-10 rounded-full border-2 border-white bg-gray-${100 + i*100} flex items-center justify-center text-xs font-medium text-white`}>
                      {String.fromCharCode(64 + i)}
                    </div>
                  ))}
                </div>
                <p className="text-sm text-muted-foreground">
                  Trusted by <span className="font-medium text-foreground">leading humanitarian organizations</span> worldwide
                </p>
              </div>
            </div>
          </div>
          
          <div className="md:order-2 order-1 relative w-full aspect-square max-w-lg mx-auto animate-fadeIn">
            <div className="absolute inset-0 rounded-full bg-primary/5 animate-pulse"></div>
            <div className="absolute inset-0">
              <AnimatedGlobe />
            </div>
          </div>
        </div>
      </div>
      
      {/* Bottom curve */}
      <div className="absolute bottom-0 left-0 right-0 h-16 bg-white rounded-t-[50%] transform translate-y-1/2"></div>
    </section>
  );
};

export default Hero;
