
# Crisis Tracker Database Setup

This document provides instructions for setting up the PostgreSQL database for the Crisis Tracker application, inserting sample data, and connecting it to the React application.

## Prerequisites

- PostgreSQL installed and running
- Node.js installed
- A PostgreSQL database created for the application

## Setup Instructions

1. **Create the PostgreSQL database**

   ```sql
   CREATE DATABASE crisis_tracker;
   ```

2. **Update connection details**

   Open `db-setup.js` and update the PostgreSQL connection details:

   ```javascript
   const pool = new Pool({
     user: 'postgres',       // Your PostgreSQL username
     host: 'localhost',      // Your PostgreSQL host
     database: 'crisis_tracker',  // Your database name
     password: 'your_password',   // Your PostgreSQL password
     port: 5432,             // Your PostgreSQL port
   });
   ```

   Also update the same connection details in `src/server.js`.

3. **Run the database setup script**

   ```bash
   node db-setup.js
   ```

   This will create the `crisis_data` table and insert all the sample data.

4. **Start the API server**

   First, install the required dependencies:

   ```bash
   npm install express pg cors
   ```

   Then start the server:

   ```bash
   node src/server.js
   ```

   The API server will start on port 3001 (or the port specified in your environment variables).

5. **Configure the React app to use the API**

   The TrialDashboard component is already configured to fetch data from `/api/crisis-data`. 
   You might need to update the API URL in the fetch call if your API server is running on a different port.

## Testing

To test the end-to-end functionality:

1. Start the PostgreSQL server
2. Run the database setup script (if not already done)
3. Start the API server
4. Start your React application
5. Navigate to the Trial Dashboard page

The dashboard should now display data from your PostgreSQL database instead of using the hardcoded mock data.

## Troubleshooting

- Check the console logs in both the server and client for error messages
- Verify PostgreSQL connection details are correct
- Confirm the API server is running and accessible
- Check that the React app is making requests to the correct API endpoint
