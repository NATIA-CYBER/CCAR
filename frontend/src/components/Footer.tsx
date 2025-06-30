
import React from 'react';

const Footer = () => {
  const currentYear = new Date().getFullYear();
  
  const footerLinks = {
    "Product": ["Features", "Use Cases", "Pricing", "API", "Integrations"],
    "Resources": ["Documentation", "Guides", "Webinars", "Blog", "Case Studies"],
    "Company": ["About Us", "Team", "Careers", "Contact Us", "Partners"],
    "Legal": ["Privacy Policy", "Terms of Service", "Security", "Compliance"]
  };
  
  return (
    <footer className="bg-gray-50 pt-16 pb-8 border-t border-gray-100">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-8 pb-12">
          <div className="col-span-2 md:col-span-1 lg:col-span-2">
            <div className="flex items-center space-x-2 mb-6">
              <span className="h-8 w-8 rounded-full bg-primary/90 flex items-center justify-center">
                <span className="text-white font-bold text-sm">C</span>
              </span>
              <span className="font-semibold text-lg">C-CAR</span>
            </div>
            
            <p className="text-muted-foreground max-w-xs mb-6">
              Advanced crisis forecasting platform empowering humanitarian organizations to anticipate and respond to emergencies more effectively.
            </p>
            
            <div className="flex space-x-4">
              {["twitter", "linkedin", "facebook", "github"].map((social) => (
                <a 
                  key={social}
                  href="#" 
                  className="h-10 w-10 rounded-full bg-white shadow-sm border border-gray-100 flex items-center justify-center text-muted-foreground hover:text-primary transition-colors"
                >
                  <span className="sr-only">{social}</span>
                  <div className="h-5 w-5"></div>
                </a>
              ))}
            </div>
          </div>
          
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h3 className="font-semibold text-sm text-foreground mb-4">{category}</h3>
              <ul className="space-y-3">
                {links.map((link) => (
                  <li key={link}>
                    <a 
                      href="#" 
                      className="text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      {link}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        
        <div className="pt-8 border-t border-gray-200 flex flex-col md:flex-row justify-between items-center">
          <p className="text-sm text-muted-foreground">
            &copy; {currentYear} C-CAR. All rights reserved.
          </p>
          
          <div className="mt-4 md:mt-0 flex items-center space-x-4">
            <a href="#" className="text-xs text-muted-foreground hover:text-primary">
              Privacy Policy
            </a>
            <span className="text-muted-foreground">•</span>
            <a href="#" className="text-xs text-muted-foreground hover:text-primary">
              Terms of Service
            </a>
            <span className="text-muted-foreground">•</span>
            <a href="#" className="text-xs text-muted-foreground hover:text-primary">
              Cookies
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
