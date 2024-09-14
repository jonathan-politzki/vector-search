// frontend/src/components/ResultDisplay.js

import React from 'react';

const ResultDisplay = ({ results }) => {
  if (!results || !results.results) {
    return <p>No results to display.</p>;
  }

  return (
    <div>
      <h2>Results:</h2>
      {results.results.map((item, index) => (
        <div key={index}>
          <h3>{item.word}</h3>
          <p>Distance: {item.distance}</p>
          {item.vector && (
            <details>
              <summary>Vector (click to expand)</summary>
              <pre>{JSON.stringify(item.vector, null, 2)}</pre>
            </details>
          )}
        </div>
      ))}
      {results.message && <p><strong>Note:</strong> {results.message}</p>}
    </div>
  );
};

export default ResultDisplay;