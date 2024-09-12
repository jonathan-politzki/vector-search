import React, { useState } from 'react';
import InputForm from './components/InputForm';
import ResultDisplay from './components/ResultDisplay';

function App() {
  const [results, setResults] = useState([]);

  const handleSubmit = async (data) => {
    try {
      const response = await fetch('http://localhost:5000/api/operate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(data),
      });
      const results = await response.json();
      setResults(results);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <h1>Embedding Space Navigator</h1>
      <InputForm onSubmit={handleSubmit} />
      <ResultDisplay results={results} />
    </div>
  );
}

export default App;