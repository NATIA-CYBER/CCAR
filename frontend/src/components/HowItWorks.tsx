
import React from 'react';
import { cn } from '@/lib/utils';
import { Database, LineChart, BarChart, PieChart } from 'lucide-react';

const HowItWorks = () => {
  const steps = [
    {
      number: '01',
      title: 'Data Collection & Integration',
      description: 'Our system aggregates data from diverse sources including conflict reports, climate data, economic indicators, population movements, and social media signals.',
      highlight: 'Comprehensive data pipeline',
      icon: <Database className="h-10 w-10 text-primary" />
    },
    {
      number: '02',
      title: 'Predictive Analysis',
      description: 'Advanced machine learning algorithms process the data to identify patterns and predict regions with the highest risk of humanitarian crises.',
      highlight: 'AI-powered forecasting',
      icon: <LineChart className="h-10 w-10 text-primary" />
    },
    {
      number: '03',
      title: 'Data Visualization',
      description: 'Interactive dashboarding services showcase the forecast model through intuitive maps, charts, and actionable insights for decision-makers.',
      highlight: 'Interactive dashboards',
      icon: <BarChart className="h-10 w-10 text-primary" />
    },
    {
      number: '04',
      title: 'Resource Planning',
      description: 'Based on the forecasts, make informed decisions about resource allocation, pre-positioning supplies, and staff deployment.',
      highlight: 'Optimized response',
      icon: <PieChart className="h-10 w-10 text-primary" />
    }
  ];

  return (
    <section id="how-it-works" className="py-20 bg-gradient-to-b from-secondary/50 to-background relative">
      <div className="absolute inset-0 bg-blue-tech bg-cover bg-fixed opacity-[0.03] -z-10"></div>
      
      <div className="section-container">
        <div className="text-center max-w-3xl mx-auto mb-16 animate-slideUp">
          <span className="inline-block px-3 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full mb-4">
            The Process
          </span>
          <h2 className="section-title">How C-CAR Works</h2>
          <p className="text-muted-foreground text-lg">
            Our systematic approach transforms complex data into actionable insights for humanitarian response.
          </p>
        </div>
        
        <div className="relative">
          {/* Vertical line connecting steps */}
          <div className="absolute left-[27px] md:left-1/2 top-0 bottom-0 w-[1px] bg-primary/20 -ml-px"></div>
          
          <div className="space-y-12 md:space-y-24 relative">
            {steps.map((step, index) => (
              <div key={index} className={cn(
                "relative grid md:grid-cols-2 gap-8 items-center animate-slideUp",
                index % 2 === 1 ? "md:rtl" : ""
              )}
              style={{ animationDelay: `${index * 100}ms` }}>
                {/* Number marker */}
                <div className="absolute left-0 md:left-1/2 w-14 h-14 rounded-full bg-white shadow-md flex items-center justify-center -ml-7 md:rtl:ml-7 md:rtl:-mr-7 border border-primary/20 z-10">
                  <span className="text-primary font-bold">{step.number}</span>
                </div>
                
                {/* Content */}
                <div className={cn(
                  "pl-20 md:pl-0 md:text-right md:rtl:text-left",
                  index % 2 === 1 ? "md:order-2" : ""
                )}>
                  <h3 className="text-2xl font-bold mb-2">{step.title}</h3>
                  <p className="text-muted-foreground mb-4">{step.description}</p>
                  <span className="inline-block px-3 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full">
                    {step.highlight}
                  </span>
                </div>
                
                {/* Illustration */}
                <div className={cn(
                  "glass-card bg-white/5 backdrop-blur-md border border-primary/10",
                  "aspect-video flex items-center justify-center p-6",
                  index % 2 === 1 ? "md:order-1" : ""
                )}>
                  <div className="w-full h-full flex flex-col items-center justify-center">
                    {step.icon}
                    <span className="text-primary font-semibold mt-4">Step {step.number}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
