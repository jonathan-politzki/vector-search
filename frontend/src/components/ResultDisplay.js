// frontend/src/components/ResultDisplay.js

import React from 'react';

const ResultDisplay = ({ results }) => {
  if (!results || !results.results || results.results.length === 0) {
    return <p>No results to display.</p>;
  }

  return (
    <div>
      <h2>Results:</h2>
      {results.results.map((item, index) => (
        <div key={index}>
          <h3>{item.word}</h3>
          <p>Similarity: {(item.similarity * 100).toFixed(2)}%</p>
          <p>Input word: {item.is_input ? 'Yes' : 'No'}</p>
        </div>
      ))}
      {results.message && <p><strong>Note:</strong> {results.message}</p>}
    </div>
  );
};

export default ResultDisplay;