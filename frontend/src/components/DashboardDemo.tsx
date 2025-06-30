
import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { 
  ChartContainer, 
  ChartTooltip, 
  ChartTooltipContent, 
  ChartLegend, 
  ChartLegendContent 
} from "@/components/ui/chart";
import { 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  ResponsiveContainer,
  LineChart,
  Line
} from "recharts";
import { ArrowUpRight, ArrowDownRight, Users, MapPin } from 'lucide-react';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

const DashboardDemo = () => {
  // Sample data for the dashboard charts
  const displacementData = {
    current: 5.9,
    previous: 5.4,
    change: 0.5,
    changePercent: 9.3,
    isIncrease: true
  };

  const refugeeData = {
    current: 7.1,
    previous: 7.8,
    change: 0.7,
    changePercent: 8.9,
    isIncrease: false
  };

  const topCountriesData = [
    { name: 'Syria', value: 2000000 },
    { name: 'Yemen', value: 1700000 },
    { name: 'Ethiopia', value: 1200000 },
    { name: 'Sudan', value: 800000 },
    { name: 'Ukraine', value: 600000 }
  ];

  const conflictVsDisasterData = [
    { name: 'Conflict', value: 38 },
    { name: 'Disaster', value: 62 }
  ];

  const ageDistributionData = [
    { name: '<18', count: 1200000 },
    { name: '18-59', count: 1650000 },
    { name: '60+', count: 2100000 }
  ];

  const genderData = [
    { name: 'Female', value: 53 },
    { name: 'Male', value: 47 }
  ];

  // Format large numbers to shorter forms
  const formatNumber = (num: number) => {
    if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
    if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
    return num.toString();
  };

  return (
    <section className="py-16 bg-gradient-to-br from-secondary/50 to-background">
      <div className="section-container">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <span className="inline-block px-3 py-1 text-xs font-medium bg-primary/10 text-primary rounded-full mb-4">
            Analytics Preview
          </span>
          <h2 className="section-title mb-4">Insights Dashboard</h2>
          <p className="text-muted-foreground text-lg mb-8">
            Make data-driven decisions using our comprehensive analytics platform
          </p>
        </div>

        <div className="dashboard-container p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
            {/* Total Internal Displacements */}
            <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Total Internal Displacements
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-3xl font-bold">{displacementData.current}M</div>
                    <div className="flex items-center mt-1">
                      {displacementData.isIncrease ? (
                        <ArrowUpRight className="h-4 w-4 text-destructive mr-1" />
                      ) : (
                        <ArrowDownRight className="h-4 w-4 text-[#00C49F] mr-1" />
                      )}
                      <span className={`text-xs font-medium ${displacementData.isIncrease ? 'text-destructive' : 'text-[#00C49F]'}`}>
                        {displacementData.changePercent}%
                      </span>
                    </div>
                  </div>
                  <div className="w-20 h-10 bg-primary/10 rounded-md"></div>
                </div>
              </CardContent>
            </Card>

            {/* Refugee Numbers */}
            <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Refugee Numbers
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-3xl font-bold">{refugeeData.current}M</div>
                    <div className="flex items-center mt-1">
                      {refugeeData.isIncrease ? (
                        <ArrowUpRight className="h-4 w-4 text-destructive mr-1" />
                      ) : (
                        <ArrowDownRight className="h-4 w-4 text-[#00C49F] mr-1" />
                      )}
                      <span className={`text-xs font-medium ${refugeeData.isIncrease ? 'text-destructive' : 'text-[#00C49F]'}`}>
                        {refugeeData.changePercent}%
                      </span>
                    </div>
                  </div>
                  <div className="w-20 h-10 bg-primary/10 rounded-md"></div>
                </div>
              </CardContent>
            </Card>

            {/* Active Regions */}
            <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Active Crisis Regions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-3xl font-bold">27</div>
                    <div className="flex items-center mt-1">
                      <MapPin className="h-4 w-4 text-primary mr-1" />
                      <span className="text-xs font-medium text-muted-foreground">
                        5 critical
                      </span>
                    </div>
                  </div>
                  <div className="w-20 h-10 bg-primary/10 rounded-md"></div>
                </div>
              </CardContent>
            </Card>

            {/* Affected Population */}
            <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  Affected Population
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-end justify-between">
                  <div>
                    <div className="text-3xl font-bold">37.8M</div>
                    <div className="flex items-center mt-1">
                      <Users className="h-4 w-4 text-primary mr-1" />
                      <span className="text-xs font-medium text-muted-foreground">
                        16.2M children
                      </span>
                    </div>
                  </div>
                  <div className="w-20 h-10 bg-primary/10 rounded-md"></div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            {/* Top 10 Countries */}
            <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
              <CardHeader>
                <CardTitle>Top Countries</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      layout="vertical"
                      data={topCountriesData}
                      margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
                    >
                      <XAxis type="number" tickFormatter={formatNumber} />
                      <YAxis dataKey="name" type="category" width={60} />
                      <Tooltip formatter={(value) => formatNumber(value as number)} />
                      <Bar dataKey="value" fill="#0088FE" radius={[0, 4, 4, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Age Distribution */}
            <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
              <CardHeader>
                <CardTitle>Age Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[250px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart
                      data={ageDistributionData}
                      margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                    >
                      <XAxis dataKey="name" />
                      <YAxis tickFormatter={formatNumber} />
                      <Tooltip formatter={(value) => formatNumber(value as number)} />
                      <Bar dataKey="count" fill="#00C49F" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>

            {/* Conflict vs Disaster + Gender Distribution */}
            <div className="grid grid-cols-1 gap-4">
              <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
                <CardHeader>
                  <CardTitle>Conflict vs Disaster</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[120px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={conflictVsDisasterData}
                          cx="50%"
                          cy="50%"
                          innerRadius={30}
                          outerRadius={50}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {conflictVsDisasterData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-card/50 backdrop-blur-sm border-primary/20">
                <CardHeader>
                  <CardTitle>Gender Breakdown</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[120px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={genderData}
                          cx="50%"
                          cy="50%"
                          innerRadius={30}
                          outerRadius={50}
                          paddingAngle={2}
                          dataKey="value"
                          label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                        >
                          {genderData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DashboardDemo;
