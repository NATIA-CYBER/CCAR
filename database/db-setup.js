
const { Pool } = require('pg');
const fs = require('fs');
const path = require('path');

// Update these connection details as needed
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'crisis_tracker',  // Make sure this database exists
  password: 'your_password',   // Replace with your PostgreSQL password
  port: 5432,
});

async function setupDatabase() {
  try {
    console.log('Reading SQL file...');
    const sqlFilePath = path.join(__dirname, 'db-setup.sql');
    const sqlScript = fs.readFileSync(sqlFilePath, 'utf8');

    console.log('Connecting to database...');
    const client = await pool.connect();

    console.log('Executing SQL script...');
    await client.query(sqlScript);
    
    console.log('Database setup completed successfully.');

    // Query to verify data was inserted
    const res = await client.query('SELECT COUNT(*) FROM crisis_data');
    console.log(`Total records inserted: ${res.rows[0].count}`);

    client.release();
  } catch (err) {
    console.error('Error setting up database:', err);
  } finally {
    // Close pool
    await pool.end();
  }
}

setupDatabase();
