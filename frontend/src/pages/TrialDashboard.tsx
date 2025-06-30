import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { ChartContainer, ChartLegend, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart";
import { Button } from "@/components/ui/button";
import { 
  LineChart, Line, BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, Area, AreaChart 
} from 'recharts';
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { 
  Map, ChevronLeft, ChevronRight, Download, Info, Save, Bell, BarChart3, 
  LineChart as LineChartIcon, Filter, RefreshCw, Database
} from 'lucide-react';
import Nav from '../components/Nav';
import Footer from '../components/Footer';
import { useToast } from '@/hooks/use-toast';
import { useQuery } from '@tanstack/react-query';

// Database interface types based on the database schema
interface CrisisData {
  year: number;
  week_number: number;
  country: string;
  num_deaths: number;
  num_conflicts: number;
  num_disasters: number;
  num_injured: number;
  num_affected: number;
  hdi: number;
  most_needs: number;
}

// Data for the crisis map and charts
const forecastData = [
  { country: 'Ukraine', forecast: 5123, average: 4616, change: 11 },
  { country: 'Russia', forecast: 1668, average: 1565, change: 7 },
  { country: 'Myanmar', forecast: 902, average: 1003, change: -10 },
  { country: 'Syria', forecast: 748, average: 967, change: -23 },
  { country: 'Palestine', forecast: 759, average: 1029, change: -26 },
  { country: 'Mexico', forecast: 641, average: 636, change: 1 },
  { country: 'Brazil', forecast: 552, average: 736, change: -25 },
  { country: 'Iraq', forecast: 443, average: 530, change: -16 },
  { country: 'Sudan', forecast: 418, average: 453, change: -8 },
  { country: 'Somalia', forecast: 359, average: 269, change: 33 },
  { country: 'Yemen', forecast: 301, average: 352, change: -14 },
  { country: 'Nigeria', forecast: 253, average: 329, change: -23 },
  { country: 'Dem. Rep. Congo', forecast: 234, average: 208, change: 12 },
  { country: 'Pakistan', forecast: 217, average: 179, change: 21 },
  { country: 'Ethiopia', forecast: 216, average: 280, change: -23 },
  { country: 'Colombia', forecast: 192, average: 186, change: 3 },
  { country: 'Lebanon', forecast: 190, average: 286, change: -33 },
  { country: 'Burkina Faso', forecast: 142, average: 154, change: -8 },
  { country: 'Mali', forecast: 127, average: 136, change: -7 },
  { country: 'South Sudan', forecast: 127, average: 91, change: 40 }
];

const violenceTypeData = [
  { type: 'All Event Types', events: 12853, change: 2 },
  { type: 'Explosions & Remote Violence', events: 6271, change: -5 },
  { type: 'Battles', events: 4108, change: 4 },
  { type: 'Violence Against Civilians', events: 2474, change: 7 }
];

const disasterTypeData = [
  { type: 'Floods', events: 387, change: 12 },
  { type: 'Droughts', events: 142, change: 8 },
  { type: 'Earthquakes', events: 89, change: -3 },
  { type: 'Wildfires', events: 76, change: 15 }
];

const historyData = [
  { month: 'Jan', historical: 14000, forecasted: null },
  { month: 'Feb', historical: 14500, forecasted: null },
  { month: 'Mar', historical: 14200, forecasted: null },
  { month: 'Apr', historical: 15500, forecasted: null },
  { month: 'May', historical: 16200, forecasted: null },
  { month: 'Jun', historical: 17500, forecasted: null },
  { month: 'Jul', historical: 19000, forecasted: null },
  { month: 'Aug', historical: 18000, forecasted: null },
  { month: 'Sep', historical: 14800, forecasted: null },
  { month: 'Oct', historical: 13500, forecasted: null },
  { month: 'Nov', historical: 12900, forecasted: null },
  { month: 'Dec', historical: 12500, forecasted: null },
  { month: 'Jan', historical: null, forecasted: 12800 },
  { month: 'Feb', historical: null, forecasted: 12800 },
  { month: 'Mar', historical: null, forecasted: 12800 },
  { month: 'Apr', historical: null, forecasted: 12800 },
];

const vulnerabilityData = [
  { country: 'Ethiopia', hdi: 0.498, vulnerabilityScore: 89 },
  { country: 'Sudan', hdi: 0.510, vulnerabilityScore: 87 },
  { country: 'Haiti', hdi: 0.535, vulnerabilityScore: 85 },
  { country: 'Yemen', hdi: 0.455, vulnerabilityScore: 92 },
  { country: 'Mali', hdi: 0.428, vulnerabilityScore: 91 },
  { country: 'Burkina Faso', hdi: 0.449, vulnerabilityScore: 86 },
  { country: 'Dem. Rep. Congo', hdi: 0.480, vulnerabilityScore: 88 },
  { country: 'Somalia', hdi: 0.361, vulnerabilityScore: 95 },
  { country: 'Syria', hdi: 0.577, vulnerabilityScore: 84 },
  { country: 'Afghanistan', hdi: 0.478, vulnerabilityScore: 90 }
];

// Data for new chart types
const conflictClusterData = [
  { country: 'Afghanistan', conflicts: 200000, cluster: 1 },
  { country: 'Ukraine', conflicts: 180000, cluster: 1 },
  { country: 'Yemen', conflicts: 170000, cluster: 1 },
  { country: 'Syria', conflicts: 130000, cluster: 2 },
  { country: 'Iraq', conflicts: 100000, cluster: 2 },
  { country: 'Sudan', conflicts: 95000, cluster: 2 },
  { country: 'Myanmar', conflicts: 85000, cluster: 3 },
  { country: 'Somalia', conflicts: 80000, cluster: 3 },
  { country: 'Nigeria', conflicts: 70000, cluster: 3 },
  { country: 'Palestine', conflicts: 65000, cluster: 3 },
  { country: 'Ethiopia', conflicts: 45000, cluster: 3 },
  { country: 'Pakistan', conflicts: 40000, cluster: 3 },
  { country: 'Democratic Republic of Congo', conflicts: 35000, cluster: 3 },
  { country: 'South Sudan', conflicts: 30000, cluster: 3 },
  { country: 'Brazil', conflicts: 25000, cluster: 3 },
  { country: 'Burkina Faso', conflicts: 20000, cluster: 3 },
  { country: 'Libya', conflicts: 15000, cluster: 3 },
  { country: 'Mali', conflicts: 10000, cluster: 3 },
  { country: 'Mexico', conflicts: 5000, cluster: 3 }
];

const disasterData = [
  { country: 'China', injured: 1671 },
  { country: 'India', injured: 1133 },
  { country: 'United States', injured: 798 },
  { country: 'Indonesia', injured: 682 },
  { country: 'Philippines', injured: 679 },
  { country: 'Pakistan', injured: 651 },
  { country: 'Iran', injured: 636 },
  { country: 'Japan', injured: 634 },
  { country: 'Bangladesh', injured: 495 },
  { country: 'Nigeria', injured: 481 }
];

const hdiDisasterData = [
  { country: 'Uganda', hdi: 590.8, conflictEvents: 20, numDisaster: 95 },
  { country: 'Sri Lanka', hdi: 616.9, conflictEvents: 150, numDisaster: 105 },
  { country: 'Sudan', hdi: 629.0, conflictEvents: 5000, numDisaster: 200 },
  { country: 'Nigeria', hdi: 646.2, conflictEvents: 8000, numDisaster: 300 },
  { country: 'Thailand', hdi: 679.7, conflictEvents: 1000, numDisaster: 400 },
  { country: 'Egypt', hdi: 685.4, conflictEvents: 1200, numDisaster: 350 },
  { country: 'Kenya', hdi: 691.1, conflictEvents: 1500, numDisaster: 250 },
  { country: 'China', hdi: 725.1, conflictEvents: 2000, numDisaster: 1355 },
  { country: 'South Africa', hdi: 850.7, conflictEvents: 10000, numDisaster: 500 },
  { country: 'Algeria', hdi: 891.1, conflictEvents: 13574, numDisaster: 600 }
];

const savedViews = [
  { id: 1, name: "Global Overview", countries: ["All"], eventTypes: ["All Event Types"], timeframe: "12months" },
  { id: 2, name: "Africa Focus", countries: ["Ethiopia", "Sudan", "Somalia"], eventTypes: ["All Event Types"], timeframe: "6months" },
  { id: 3, name: "Middle East Crisis", countries: ["Syria", "Yemen", "Iraq"], eventTypes: ["Battles", "Violence Against Civilians"], timeframe: "3months" }
];

const getChangeColor = (change: number) => {
  if (change > 15) return 'text-red-600';
  if (change > 5) return 'text-orange-500';
  if (change > -5) return 'text-gray-700';
  if (change > -15) return 'text-yellow-500';
  return 'text-green-600';
};

const getChangeBackground = (change: number) => {
  if (change > 15) return 'bg-red-100';
  if (change > 5) return 'bg-orange-50';
  if (change > -5) return 'bg-gray-50';
  if (change > -15) return 'bg-yellow-50';
  return 'bg-green-50';
};

const getVulnerabilityColor = (score: number) => {
  if (score > 90) return 'text-red-600';
  if (score > 80) return 'text-orange-500';
  if (score > 70) return 'text-yellow-500';
  return 'text-blue-500';
};

const getClusterColor = (cluster: number) => {
  switch (cluster) {
    case 1: return '#3B82F6'; // blue
    case 2: return '#F97316'; // orange
    case 3: return '#EF4444'; // red
    case 4: return '#10B981'; // green
    default: return '#6B7280'; // gray
  }
};

const TrialDashboard = () => {
  const { toast } = useToast();
  const [activeTab, setActiveTab] = useState('needsIndex');
  const [selectedView, setSelectedView] = useState(0);
  const [selectedCountries, setSelectedCountries] = useState<string[]>(['All']);
  const [selectedEventTypes, setSelectedEventTypes] = useState<string[]>(['All Event Types']);
  const [timeframe, setTimeframe] = useState('12months');
  const [chartType, setChartType] = useState('line');
  const [showSubscribeModal, setShowSubscribeModal] = useState(false);
  const [dashboardName, setDashboardName] = useState('');
  const [currentSlide, setCurrentSlide] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedYear, setSelectedYear] = useState(new Date().getFullYear());
  const [selectedWeek, setSelectedWeek] = useState(1);
  
  const slides = [
    { id: 'needsIndex', title: 'Most Needs Index Map', subtitle: 'Bar charts of most affected countries' },
    { id: 'hdiForecasts', title: 'HDI aggregation per conflict + Forecast', subtitle: 'HDI aggregation per country' }
  ];
  
  const currentMonth = new Date().toLocaleString('default', { month: 'long' });
  const currentYear = new Date().getFullYear();

  // Query to fetch crisis data from database
  const { data: crisisData, error, isError } = useQuery({
    queryKey: ['crisisData', selectedYear, selectedWeek],
    queryFn: async (): Promise<CrisisData[]> => {
      setIsLoading(true);
      try {
        const response = await fetch(`/api/crisis-data?year=${selectedYear}&week=${selectedWeek}`);
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        const data = await response.json();
        return data;
      } catch (error) {
        console.error('Error fetching crisis data:', error);
        throw error;
      } finally {
        setIsLoading(false);
      }
    }
  });

  // Transform database data to chart format 
  const transformDataForCharts = () => {
    if (!crisisData || crisisData.length === 0) {
      return {
        countryForecasts: forecastData, // Use mock data as fallback
        conflictClusterData: conflictClusterData,
        disasterData: disasterData,
        hdiDisasterData: hdiDisasterData
      };
    }

    // Transform for country forecasts
    const countryForecasts = crisisData.map(item => {
      // Calculate a pseudo "change" value based on week_number variation
      const change = ((item.num_conflicts / (item.week_number || 1)) - 10);
      
      return {
        country: item.country,
        forecast: item.num_conflicts,
        average: item.num_conflicts - (item.num_conflicts * (change / 100)),
        change: Math.round(change)
      };
    });

    // Transform for conflict cluster
    const transformedConflictClusterData = crisisData.map(item => {
      // Define cluster based on needs and HDI
      const cluster = item.hdi < 0.5 ? 1 : item.hdi < 0.7 ? 2 : 3;
      
      return {
        country: item.country,
        conflicts: item.num_conflicts,
        cluster: cluster
      };
    });

    // Transform for disaster data
    const transformedDisasterData = crisisData.map(item => ({
      country: item.country,
      injured: item.num_injured
    }));

    // Transform for HDI and disaster correlation
    const transformedHdiData = crisisData.map(item => ({
      country: item.country,
      hdi: item.hdi * 1000, // Scale HDI for better visualization
      conflictEvents: item.num_conflicts,
      numDisaster: item.num_disasters
    }));

    return {
      countryForecasts: countryForecasts.slice(0, 20), // Limit to 20 countries
      conflictClusterData: transformedConflictClusterData.slice(0, 19),
      disasterData: transformedDisasterData.slice(0, 10),
      hdiDisasterData: transformedHdiData.slice(0, 10)
    };
  };

  const { 
    countryForecasts, 
    conflictClusterData: transformedConflictData, 
    disasterData: transformedDisasterData,
    hdiDisasterData: transformedHdiData
  } = transformDataForCharts();

  useEffect(() => {
    if (isError) {
      toast({
        title: "Error loading data",
        description: "Could not fetch crisis data from the database. Using fallback data.",
        variant: "destructive"
      });
    }
  }, [isError, toast]);

  const filteredForecastData = forecastData.filter(item => {
    if (selectedCountries.includes('All')) return true;
    return selectedCountries.includes(item.country);
  });

  const handleSaveView = () => {
    if (dashboardName.trim() !== '') {
      toast({
        title: "Configuration saved",
        description: `"${dashboardName}" has been saved to your dashboard views.`,
      });
      setDashboardName('');
      setShowSubscribeModal(true);
    } else {
      toast({
        title: "Name required",
        description: "Please provide a name for your dashboard configuration.",
        variant: "destructive",
      });
    }
  };

  const handleSetupAlerts = () => {
    toast({
      title: "Upgrade required",
      description: "Alert configuration is available in the premium subscription.",
    });
    setShowSubscribeModal(true);
  };

  const handleGenerateReport = () => {
    toast({
      title: "Generating report",
      description: "Your insights report is being prepared.",
    });
    
    setTimeout(() => {
      setShowSubscribeModal(true);
    }, 2000);
  };

  const handleNextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % slides.length);
    setActiveTab(slides[(currentSlide + 1) % slides.length].id);
  };

  const handlePrevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + slides.length) % slides.length);
    setActiveTab(slides[(currentSlide - 1 + slides.length) % slides.length].id);
  };

  const handleYearChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedYear(parseInt(e.target.value));
  };

  const handleWeekChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedWeek(parseInt(e.target.value));
  };
  
  return (
    <div className="min-h-screen flex flex-col">
      <Nav />
      
      <main className="flex-grow pt-24 pb-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center mb-8">
            <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">Complex Crisis Anticipated Response</h1>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-muted-foreground">Trial Version</span>
              <Button size="sm" variant="outline">
                <Download className="h-4 w-4 mr-2" /> Export
              </Button>
            </div>
          </div>
          
          <div className="relative mb-8">
            <div className="flex justify-center">
              <div className="flex items-center space-x-2">
                <Button variant="outline" size="icon" onClick={handlePrevSlide}>
                  <ChevronLeft className="h-4 w-4" />
                </Button>
                
                <div className="grid grid-cols-2 gap-4 w-[580px]">
                  {slides.map((slide, index) => (
                    <div 
                      key={slide.id}
                      className={`p-4 cursor-pointer border rounded-md ${
                        currentSlide === index 
                          ? "bg-gray-100 border-gray-300" 
                          : "bg-white border-gray-200"
                      }`}
                      onClick={() => {
                        setCurrentSlide(index);
                        setActiveTab(slide.id);
                      }}
                    >
                      <div className="font-medium">{slide.title}</div>
                      <div className="text-sm text-gray-500">{slide.subtitle}</div>
                    </div>
                  ))}
                </div>
                
                <Button variant="outline" size="icon" onClick={handleNextSlide}>
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
          
          <div className="space-y-6">
            {activeTab === 'needsIndex' && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <Card className="col-span-3">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-xl">Most Needs Index</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="relative">
                        <div className="aspect-[16/9] bg-gray-100 flex items-center justify-center">
                          <img 
                            src="/lovable-uploads/96a24a9c-df61-447f-840d-2ca71c02cb38.png" 
                            alt="Crisis Forecast Map" 
                            className="w-full h-full object-cover"
                          />
                        </div>
                        
                        <div className="absolute bottom-4 left-4 right-4 flex justify-between">
                          <div className="bg-white/80 backdrop-blur-sm p-3 rounded-md shadow-sm">
                            <div className="text-sm font-medium">Most Needs</div>
                            <div className="flex items-center space-x-4 mt-2">
                              <div className="flex items-center">
                                <div className="w-4 h-4 rounded-full bg-blue-500 mr-2"></div>
                                <span className="text-xs">Cluster 1</span>
                              </div>
                              <div className="flex items-center">
                                <div className="w-4 h-4 rounded-full bg-orange-500 mr-2"></div>
                                <span className="text-xs">Cluster 2</span>
                              </div>
                              <div className="flex items-center">
                                <div className="w-4 h-4 rounded-full bg-red-500 mr-2"></div>
                                <span className="text-xs">Cluster 3</span>
                              </div>
                            </div>
                          </div>
                          
                          <div className="flex flex-col space-y-2">
                            <div className="bg-white/80 backdrop-blur-sm p-2 rounded-md shadow-sm">
                              <select className="text-xs w-full bg-transparent border-none focus:ring-0">
                                <option>All</option>
                                {countryForecasts.map(item => (
                                  <option key={item.country} value={item.country}>{item.country}</option>
                                ))}
                              </select>
                            </div>
                            <div className="bg-white/80 backdrop-blur-sm p-2 rounded-md shadow-sm">
                              <select className="text-xs w-full bg-transparent border-none focus:ring-0">
                                <option>All Event Types</option>
                                {violenceTypeData.map(item => (
                                  <option key={item.type} value={item.type}>{item.type}</option>
                                ))}
                              </select>
                            </div>
                            <div className="bg-white/80 backdrop-blur-sm p-2 rounded-md shadow-sm">
                              <select className="text-xs w-full bg-transparent border-none focus:ring-0">
                                <option>Last 12 months</option>
                                <option>Last 6 months</option>
                                <option>Last 3 months</option>
                              </select>
                            </div>
                            <div className="bg-white/80 backdrop-blur-sm p-2 rounded-md shadow-sm">
                              <select className="text-xs w-full bg-transparent border-none focus:ring-0">
                                <option>April 2025</option>
                                <option>May 2025</option>
                                <option>June 2025</option>
                              </select>
                            </div>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Conflict Clusters</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-96">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={transformedConflictData || conflictClusterData}
                            layout="vertical"
                            margin={{ top: 5, right: 30, left: 60, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} />
                            <XAxis type="number" />
                            <YAxis 
                              type="category" 
                              dataKey="country" 
                              tick={{ fontSize: 12 }} 
                              width={100}
                            />
                            <Tooltip formatter={(value) => [`${value} events`, 'Conflicts']} />
                            <Bar dataKey="conflicts" name="Conflict Events">
                              {(transformedConflictData || conflictClusterData).map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={getClusterColor(entry.cluster)} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                      <div className="mt-4">
                        <div className="text-sm font-medium mb-2">Cluster Legends</div>
                        <div className="space-y-1 text-sm">
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-sm bg-blue-500 mr-2"></div>
                            <span>Cluster 1 = extreme severity of needs (most conflict events + lowest HDI)</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-sm bg-orange-500 mr-2"></div>
                            <span>Cluster 2 = high severity of needs (high conflict events + medium HDI)</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-sm bg-red-500 mr-2"></div>
                            <span>Cluster 3 = medium severity of needs (mid conflict events + high HDI)</span>
                          </div>
                          <div className="flex items-center">
                            <div className="w-3 h-3 rounded-sm bg-green-500 mr-2"></div>
                            <span>Cluster 4 = low severity of needs (low conflict events + highest HDI)</span>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Top 10 Disaster-Affected Countries</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-96">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={transformedDisasterData || disasterData}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="country" />
                            <YAxis />
                            <Tooltip formatter={(value) => [`${value} people`, 'Injured or Affected']} />
                            <Bar dataKey="injured" name="Number of Injured/Affected" fill="#38bdf8" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </>
            )}
            
            {activeTab === 'hdiForecasts' && (
              <>
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  <Card className="lg:col-span-1">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">Country Search Engine</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        <table className="w-full text-sm">
                          <thead>
                            <tr className="border-b">
                              <th className="text-left font-medium py-2 px-2">Country</th>
                              <th className="text-right font-medium py-2 px-2">Forecast</th>
                              <th className="text-right font-medium py-2 px-2">Average</th>
                              <th className="text-right font-medium py-2 px-2">Change</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(countryForecasts).slice(0, 10).map((row) => (
                              <tr key={row.country} className="border-b hover:bg-gray-50">
                                <td className="py-1 px-2">{row.country}</td>
                                <td className="text-right py-1 px-2">{row.forecast}</td>
                                <td className="text-right py-1 px-2">{row.average}</td>
                                <td className="text-right py-1 px-2">
                                  <span className={`px-1 py-0.5 rounded text-xs ${getChangeColor(row.change)} ${getChangeBackground(row.change)}`}>
                                    {row.change > 0 ? '+' : ''}{row.change}%
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        
                        <div className="pt-3 border-t">
                          <div className="text-sm font-medium mb-2">Search engine with filters</div>
                          <div className="flex flex-col space-y-3">
                            <input 
                              type="text" 
                              placeholder="Search countries..." 
                              className="border rounded-md p-2 text-sm"
                            />
                            <select className="border rounded-md p-2 text-sm">
                              <option>All Event Types</option>
                              {violenceTypeData.map(item => (
                                <option key={item.type}>{item.type}</option>
                              ))}
                            </select>
                            <Button size="sm">
                              <Filter className="h-4 w-4 mr-2" />
                              Apply Filters
                            </Button>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  
                  <Card className="lg:col-span-2">
                    <CardHeader className="pb-2">
                      <CardTitle className="text-lg">HDI and Disaster Correlation</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="h-96">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={transformedHdiData || hdiDisasterData}
                            margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="country" />
                            <YAxis />
                            <Tooltip 
                              formatter={(value, name) => [
                                `${value}`, 
                                name === 'hdi' ? 'HDI Score' : 
                                name === 'numDisaster' ? 'Disaster Events' : 'Conflict Events'
                              ]} 
                            />
                            <Bar dataKey="hdi" name="HDI Score" fill="#f97316" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </CardContent>
                  </Card>
                </div>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg">HDI and Conflict Correlation</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="h-96">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={transformedHdiData || hdiDisasterData}
                          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" vertical={false} />
                          <XAxis dataKey="country" />
                          <YAxis />
                          <Tooltip 
                            formatter={(value, name) => [
                              `${value}`, 
                              name === 'hdi' ? 'HDI Score' : 
                              name === 'conflictEvents' ? 'Conflict Events' : 'Disaster Events'
                            ]} 
                          />
                          <Bar dataKey="hdi" name="HDI Score" fill="#f97316" />
                          <Bar dataKey="conflictEvents" name="Conflict Events" fill="#ef4444" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="mt-4 text-sm text-gray-500">
                      <p>Search engine with filters of layers of map (HDI + conflict events + disaster events).</p>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
            
            {showSubscribeModal && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
                <Card className="w-full max-w-md">
                  <CardHeader>
                    <CardTitle>Upgrade to Premium</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <p>To unlock premium features including:</p>
                    <ul className="list-disc pl-5 space-y-2">
                      <li>Save custom dashboard configurations</li>
                      <li>Set up email alerts for forecast changes</li>
                      <li>Generate detailed insights reports</li>
                      <li>Access historical data going back 5 years</li>
                      <li>API access for integration with your systems</li>
                    </ul>
                    <div className="flex justify-end space-x-3 mt-6">
                      <Button 
                        variant="outline" 
                        onClick={() => setShowSubscribeModal(false)}
                      >
                        Continue with Trial
                      </Button>
                      <Button onClick={() => {
                        toast({
                          title: "Thank you for your interest!",
                          description: "This is a demo application. In a real app, you would be directed to subscription options.",
                        });
                        setShowSubscribeModal(false);
                      }}>
                        View Plans
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default TrialDashboard;
