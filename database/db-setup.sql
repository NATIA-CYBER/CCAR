
-- Create the crisis_data table matching the schema from the image
CREATE TABLE IF NOT EXISTS crisis_data (
    id SERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    week_number INTEGER NOT NULL,
    country VARCHAR(100) NOT NULL,
    num_deaths INTEGER NOT NULL,
    num_conflicts INTEGER NOT NULL,
    num_disasters INTEGER NOT NULL, 
    num_injured INTEGER NOT NULL,
    num_affected INTEGER NOT NULL,
    hdi DECIMAL(5,3) NOT NULL,
    most_needs INTEGER NOT NULL
);

-- Clear existing data if needed
TRUNCATE TABLE crisis_data RESTART IDENTITY;

-- Insert mockup data for multiple countries and weeks
-- Using the mock data from the TrialDashboard component

-- Week 1 Data for 2025
INSERT INTO crisis_data (year, week_number, country, num_deaths, num_conflicts, num_disasters, num_injured, num_affected, hdi, most_needs)
VALUES 
-- Cluster 1 countries (high severity)
(2025, 1, 'Ukraine', 851, 5123, 12, 2410, 12500, 0.773, 1),
(2025, 1, 'Russia', 320, 1668, 5, 912, 3800, 0.822, 1),
(2025, 1, 'Myanmar', 215, 902, 8, 630, 4200, 0.583, 1),
(2025, 1, 'Syria', 189, 748, 3, 520, 8700, 0.577, 1),
(2025, 1, 'Palestine', 201, 759, 2, 1105, 12400, 0.708, 1),

-- Cluster 2 countries (medium severity)
(2025, 1, 'Mexico', 158, 641, 4, 329, 1800, 0.758, 2),
(2025, 1, 'Brazil', 126, 552, 11, 418, 3600, 0.754, 2),
(2025, 1, 'Iraq', 98, 443, 3, 276, 1900, 0.686, 2),
(2025, 1, 'Sudan', 120, 418, 7, 682, 9300, 0.510, 2),
(2025, 1, 'Somalia', 89, 359, 5, 495, 8200, 0.361, 2),

-- Cluster 3 countries (lower severity)
(2025, 1, 'Yemen', 72, 301, 4, 210, 5700, 0.455, 3),
(2025, 1, 'Nigeria', 63, 253, 9, 481, 7300, 0.539, 3),
(2025, 1, 'Dem. Rep. Congo', 58, 234, 6, 317, 6200, 0.480, 3),
(2025, 1, 'Pakistan', 49, 217, 12, 651, 8900, 0.544, 3),
(2025, 1, 'Ethiopia', 55, 216, 7, 412, 7100, 0.498, 3),
(2025, 1, 'Colombia', 41, 192, 3, 186, 2300, 0.752, 3),
(2025, 1, 'Lebanon', 39, 190, 1, 124, 1700, 0.730, 3),
(2025, 1, 'Burkina Faso', 32, 142, 5, 228, 3400, 0.449, 3),
(2025, 1, 'Mali', 29, 127, 4, 169, 2800, 0.428, 3),
(2025, 1, 'South Sudan', 28, 127, 3, 215, 3200, 0.385, 3);

-- Week 2 Data for 2025 (slight variations to show week-to-week changes)
INSERT INTO crisis_data (year, week_number, country, num_deaths, num_conflicts, num_disasters, num_injured, num_affected, hdi, most_needs)
VALUES 
-- Cluster 1 countries (high severity)
(2025, 2, 'Ukraine', 878, 5231, 10, 2510, 12800, 0.773, 1),
(2025, 2, 'Russia', 332, 1712, 6, 928, 3900, 0.822, 1),
(2025, 2, 'Myanmar', 201, 892, 9, 612, 4150, 0.583, 1),
(2025, 2, 'Syria', 196, 762, 4, 530, 8800, 0.577, 1),
(2025, 2, 'Palestine', 212, 782, 3, 1150, 12600, 0.708, 1),

-- Cluster 2 countries (medium severity)
(2025, 2, 'Mexico', 151, 629, 5, 340, 1850, 0.758, 2),
(2025, 2, 'Brazil', 132, 567, 12, 430, 3700, 0.754, 2),
(2025, 2, 'Iraq', 102, 456, 2, 282, 1950, 0.686, 2),
(2025, 2, 'Sudan', 115, 410, 8, 670, 9250, 0.510, 2),
(2025, 2, 'Somalia', 95, 372, 6, 510, 8300, 0.361, 2),

-- Cluster 3 countries (lower severity)
(2025, 2, 'Yemen', 68, 294, 5, 205, 5650, 0.455, 3),
(2025, 2, 'Nigeria', 68, 261, 8, 490, 7350, 0.539, 3),
(2025, 2, 'Dem. Rep. Congo', 62, 242, 5, 325, 6300, 0.480, 3),
(2025, 2, 'Pakistan', 53, 224, 11, 643, 8850, 0.544, 3),
(2025, 2, 'Ethiopia', 51, 210, 8, 401, 7050, 0.498, 3),
(2025, 2, 'Colombia', 45, 198, 2, 192, 2350, 0.752, 3),
(2025, 2, 'Lebanon', 42, 195, 1, 130, 1750, 0.730, 3),
(2025, 2, 'Burkina Faso', 30, 138, 6, 232, 3450, 0.449, 3),
(2025, 2, 'Mali', 32, 132, 3, 173, 2850, 0.428, 3),
(2025, 2, 'South Sudan', 31, 132, 4, 221, 3250, 0.385, 3);

-- Add data for other major countries mentioned in disaster data
INSERT INTO crisis_data (year, week_number, country, num_deaths, num_conflicts, num_disasters, num_injured, num_affected, hdi, most_needs)
VALUES
(2025, 1, 'China', 42, 98, 18, 1671, 25000, 0.768, 4),
(2025, 1, 'India', 65, 143, 22, 1133, 31000, 0.633, 3),
(2025, 1, 'United States', 21, 87, 14, 798, 12000, 0.921, 4),
(2025, 1, 'Indonesia', 35, 105, 16, 682, 14500, 0.705, 3),
(2025, 1, 'Philippines', 29, 92, 15, 679, 18300, 0.699, 3),
(2025, 1, 'Iran', 31, 124, 7, 636, 7800, 0.774, 3),
(2025, 1, 'Japan', 12, 23, 9, 634, 5200, 0.919, 4),
(2025, 1, 'Bangladesh', 24, 56, 13, 495, 24700, 0.661, 3),
(2025, 1, 'Egypt', 33, 167, 6, 342, 4800, 0.731, 3),
(2025, 1, 'Uganda', 19, 78, 8, 286, 5300, 0.544, 3),
(2025, 1, 'Sri Lanka', 14, 89, 11, 317, 6100, 0.782, 3),
(2025, 1, 'Kenya', 27, 142, 9, 413, 7800, 0.601, 3),
(2025, 1, 'South Africa', 23, 128, 7, 362, 4200, 0.709, 3),
(2025, 1, 'Algeria', 18, 103, 5, 278, 3100, 0.748, 3),
(2025, 1, 'Thailand', 16, 67, 12, 321, 5400, 0.800, 3);

-- Week 2 data for these additional countries
INSERT INTO crisis_data (year, week_number, country, num_deaths, num_conflicts, num_disasters, num_injured, num_affected, hdi, most_needs)
VALUES
(2025, 2, 'China', 45, 102, 17, 1685, 25300, 0.768, 4),
(2025, 2, 'India', 68, 149, 21, 1150, 31200, 0.633, 3),
(2025, 2, 'United States', 19, 85, 15, 810, 12100, 0.921, 4),
(2025, 2, 'Indonesia', 37, 109, 15, 690, 14600, 0.705, 3),
(2025, 2, 'Philippines', 32, 96, 14, 685, 18400, 0.699, 3),
(2025, 2, 'Iran', 29, 120, 8, 642, 7850, 0.774, 3),
(2025, 2, 'Japan', 10, 21, 10, 640, 5250, 0.919, 4),
(2025, 2, 'Bangladesh', 27, 59, 12, 503, 24800, 0.661, 3),
(2025, 2, 'Egypt', 35, 172, 5, 348, 4850, 0.731, 3),
(2025, 2, 'Uganda', 21, 82, 7, 292, 5350, 0.544, 3),
(2025, 2, 'Sri Lanka', 16, 93, 10, 325, 6150, 0.782, 3),
(2025, 2, 'Kenya', 29, 148, 8, 421, 7900, 0.601, 3),
(2025, 2, 'South Africa', 25, 133, 6, 368, 4300, 0.709, 3),
(2025, 2, 'Algeria', 20, 107, 4, 285, 3150, 0.748, 3),
(2025, 2, 'Thailand', 18, 70, 11, 328, 5450, 0.800, 3);

-- Data for historical comparison - 2024
INSERT INTO crisis_data (year, week_number, country, num_deaths, num_conflicts, num_disasters, num_injured, num_affected, hdi, most_needs)
VALUES
(2024, 1, 'Ukraine', 795, 4616, 8, 2310, 11800, 0.773, 1),
(2024, 1, 'Russia', 289, 1565, 4, 867, 3600, 0.822, 1),
(2024, 1, 'Myanmar', 232, 1003, 7, 645, 4300, 0.583, 1),
(2024, 1, 'Syria', 215, 967, 2, 542, 8900, 0.577, 1),
(2024, 1, 'Palestine', 273, 1029, 1, 1250, 13100, 0.708, 1),
(2024, 1, 'Mexico', 163, 636, 3, 318, 1750, 0.758, 2),
(2024, 1, 'Brazil', 168, 736, 9, 432, 3800, 0.754, 2),
(2024, 1, 'Iraq', 118, 530, 2, 291, 2050, 0.686, 2),
(2024, 1, 'Sudan', 109, 453, 6, 653, 9100, 0.510, 2),
(2024, 1, 'Somalia', 71, 269, 4, 462, 7900, 0.361, 2);

-- Create indexes to improve query performance
CREATE INDEX IF NOT EXISTS crisis_data_year_week_idx ON crisis_data(year, week_number);
CREATE INDEX IF NOT EXISTS crisis_data_country_idx ON crisis_data(country);
