
// Simple Express server to serve crisis data API
const express = require('express');
const { Pool } = require('pg');
const cors = require('cors');

const app = express();
const port = process.env.PORT || 3001;

// Configure PostgreSQL connection
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'crisis_tracker',  // Make sure this matches your database name
  password: 'your_password',   // Replace with your PostgreSQL password
  port: 5432,
});

app.use(cors());
app.use(express.json());

// API endpoint to get crisis data by year and week
app.get('/api/crisis-data', async (req, res) => {
  try {
    const { year, week } = req.query;
    
    let query = 'SELECT * FROM crisis_data';
    const queryParams = [];
    
    // Add filters if provided
    if (year && week) {
      query += ' WHERE year = $1 AND week_number = $2';
      queryParams.push(year, week);
    } else if (year) {
      query += ' WHERE year = $1';
      queryParams.push(year);
    }
    
    // Order results for consistency
    query += ' ORDER BY country';
    
    const result = await pool.query(query, queryParams);
    res.json(result.rows);
  } catch (err) {
    console.error('Database query error:', err);
    res.status(500).json({ error: 'Internal server error' });
  }
});

app.listen(port, () => {
  console.log(`API server running on port ${port}`);
});
