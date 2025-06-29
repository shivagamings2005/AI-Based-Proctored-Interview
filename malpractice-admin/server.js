const express = require('express');
const { MongoClient } = require('mongodb');
const cors = require('cors');
require('dotenv').config();

const app = express();
const port = 5001;

app.use(cors());
app.use(express.json());

const uri = process.env.MONGODB_URI;
const client = new MongoClient(uri);

async function connectDB() {
  try {
    await client.connect();
    console.log("Connected to MongoDB");
  } catch (error) {
    console.error("MongoDB connection error:", error);
  }
}

connectDB();

app.post('/api/proctoring', async (req, res) => {
  try {
    const { emails, privileges, examName, sessionMode } = req.body;
    const db = client.db('proctoring');
    const collection = db.collection('proctoring');

    // Insert data into a single collection
    await collection.insertOne({
      emails,
      privileges,
      examName,
      sessionMode
    });

    res.status(200).json({ message: 'Data saved successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to save data' });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});