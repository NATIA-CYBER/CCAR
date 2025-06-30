
import React from 'react';
import { Check } from 'lucide-react';
import { Button } from "@/components/ui/button";

const PricingTier = ({ 
  name, 
  price, 
  features, 
  isPopular 
}: { 
  name: string;
  price: number;
  features: string[];
  isPopular?: boolean;
}) => (
  <div className={`relative flex flex-col p-6 bg-white rounded-2xl shadow-lg ${isPopular ? 'border-2 border-primary' : 'border border-gray-200'}`}>
    {isPopular && (
      <div className="absolute -top-4 left-1/2 -translate-x-1/2 px-4 py-1 bg-primary text-white text-sm font-medium rounded-full">
        Most Popular
      </div>
    )}
    
    <div className="mb-6">
      <h3 className="text-2xl font-bold">{name}</h3>
      <div className="mt-4 flex items-baseline">
        <span className="text-4xl font-bold">€{price}</span>
        <span className="ml-2 text-gray-500">/month</span>
      </div>
    </div>
    
    <ul className="space-y-4 flex-grow mb-6">
      {features.map((feature, index) => (
        <li key={index} className="flex items-start gap-3">
          <Check className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          <span className="text-gray-600 text-left">{feature}</span>
        </li>
      ))}
    </ul>
    
    <Button className="w-full" variant={isPopular ? "default" : "outline"}>
      Get Started
    </Button>
  </div>
);

const Pricing = () => {
  const plans = [
    {
      name: "Basic",
      price: 500,
      features: [
        "Core risk assessment",
        "Limited country coverage",
        "Standard support",
        "Basic analytics dashboard",
        "Monthly reports"
      ]
    },
    {
      name: "Standard",
      price: 1000,
      features: [
        "Full risk assessment tools",
        "All-country coverage",
        "Advanced analytics",
        "Priority support",
        "Weekly reports",
        "Custom dashboards"
      ],
      isPopular: true
    },
    {
      name: "Premium",
      price: 1500,
      features: [
        "Custom risk reports",
        "Dedicated account manager",
        "API integrations",
        "24/7 priority support",
        "Daily reports",
        "Custom analysis"
      ]
    }
  ];

  return (
    <section id="pricing" className="py-20 bg-gray-50">
      <div className="section-container">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <span className="inline-block px-3 py-1 text-xs font-medium bg-primary/5 text-primary rounded-full mb-4">
            Pricing Plans
          </span>
          <h2 className="text-3xl md:text-4xl font-bold mb-6">
            Choose the Right Plan for Your Organization
          </h2>
          <p className="text-muted-foreground text-lg">
            Flexible pricing options designed to meet the needs of humanitarian organizations of all sizes.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {plans.map((plan) => (
            <PricingTier key={plan.name} {...plan} />
          ))}
        </div>
      </div>
    </section>
  );
};

export default Pricing;
