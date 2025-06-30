
import React, { useState, useEffect } from 'react';
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Link } from 'react-router-dom';

const Nav = () => {
  const [scrolled, setScrolled] = useState(false);
  
  useEffect(() => {
    const handleScroll = () => {
      const isScrolled = window.scrollY > 10;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, [scrolled]);

  return (
    <header 
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-300 ease-in-out",
        scrolled ? "bg-white/80 backdrop-blur-md shadow-sm py-3" : "bg-transparent py-5"
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Link to="/" className="flex items-center space-x-2">
              <span className="h-8 w-8 rounded-full bg-primary/90 flex items-center justify-center">
                <span className="text-white font-bold text-sm">C</span>
              </span>
              <span className={cn(
                "font-semibold text-lg transition-colors duration-300",
                scrolled ? "text-foreground" : "text-foreground"
              )}>
                C-CAR
              </span>
            </Link>
          </div>
          
          <nav className="hidden md:flex items-center space-x-8">
            {["Features", "How it Works", "About Us", "Contact"].map((item) => (
              <a 
                key={item}
                href={`#${item.toLowerCase().replace(/\s+/g, '-')}`}
                className={cn(
                  "text-sm font-medium transition-all duration-300 hover:text-primary",
                  scrolled ? "text-foreground" : "text-foreground"
                )}
              >
                {item}
              </a>
            ))}
          </nav>
          
          <div className="flex items-center space-x-4">
            <Button 
              variant="outline" 
              size="sm" 
              className={cn(
                "transition-all duration-300 hidden sm:inline-flex",
                scrolled 
                  ? "bg-white text-foreground hover:bg-gray-100" 
                  : "bg-white/80 backdrop-blur-sm text-foreground hover:bg-white"
              )}
            >
              Log In
            </Button>
            <Link to="/trial-dashboard">
              <Button 
                size="sm" 
                className={cn(
                  "transition-all duration-300",
                  "bg-primary text-white hover:bg-primary/90"
                )}
              >
                Start Free Trial
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Nav;
