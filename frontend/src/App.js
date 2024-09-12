// frontend/src/App.js

import React, { useState } from 'react';
import InputForm from './components/InputForm';
import ResultDisplay from './components/ResultDisplay';

function App() {
  const [results, setResults] = useState([]);

  const handleSubmit = async (data) => {
    try {
      const response = await fetch('http://127.0.0.1:5000/api/operate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
  
      if (!response.ok) {
        // Extract error message from response
        const errorData = await response.json();
        setResults({ error: errorData.error || 'An error occurred' });
      } else {
        const results = await response.json();
        setResults(results);
      }
    } catch (error) {
      console.error('Error:', error);
      setResults({ error: 'An unexpected error occurred' });
    }
  };

  return (
    <div className="App">
      <h1>Vector Search</h1>
      <InputForm onSubmit={handleSubmit} />
      <ResultDisplay results={results} />
    </div>
  );
}

export default App;