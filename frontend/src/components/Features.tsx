
import React from 'react';
import { 
  Database, 
  Map, 
  BarChart3, 
  Globe, 
  LineChart,
  Building
} from 'lucide-react';

const Features = () => {
  const features = [
    {
      icon: <Database className="h-6 w-6 text-primary" />,
      title: "Historical Data Analysis",
      description: "Leverage validated secondary data from trusted sources including ACLED for conflict data and EM-DAT for meteorological events."
    },
    {
      icon: <Map className="h-6 w-6 text-primary" />,
      title: "Crisis Forecasting",
      description: "Advanced predictive modeling to identify regions with the highest likelihood of humanitarian crises based on historical patterns."
    },
    {
      icon: <Globe className="h-6 w-6 text-primary" />,
      title: "HDI Integration",
      description: "Incorporation of Human Development Index data to prioritize regions with the most humanitarian needs."
    },
    {
      icon: <BarChart3 className="h-6 w-6 text-primary" />,
      title: "Comprehensive Analytics",
      description: "Detailed analysis and visualization of historical crisis patterns and socio-economic indicators."
    },
    {
      icon: <LineChart className="h-6 w-6 text-primary" />,
      title: "Trend Analysis",
      description: "Track and analyze long-term crisis patterns using validated historical data to improve forecast accuracy."
    },
    {
      icon: <Building className="h-6 w-6 text-primary" />,
      title: "Institutional Access",
      description: "Flexible subscription plans tailored for organizations of all sizes, from NGOs to major international institutions."
    }
  ];

  return (
    <section id="features" className="py-20 bg-white relative">
      <div className="section-container">
        <div className="text-center max-w-3xl mx-auto mb-16 animate-slideUp">
          <span className="inline-block px-3 py-1 text-xs font-medium bg-primary/5 text-primary rounded-full mb-4">
            Core Features
          </span>
          <h2 className="section-title">Data-Driven Decision Making</h2>
          <p className="text-muted-foreground text-lg">
            Our comprehensive platform offers the analytics you need to anticipate where crises will impact the most vulnerable.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <div 
              key={index}
              className="glass-card p-6 sm:p-8 flex flex-col animate-slideUp"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <div className="rounded-full bg-primary/10 p-3 w-fit mb-6">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-muted-foreground flex-grow">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
