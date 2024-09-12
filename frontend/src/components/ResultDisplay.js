import React from 'react';

const ResultDisplay = ({ results }) => {
  return (
    <div>
      <h2>Results:</h2>
      <ul>
        {results.map((result, index) => (
          <li key={index}>{result[0]}: {result[1].toFixed(4)}</li>
        ))}
      </ul>
    </div>
  );
};

export default ResultDisplay;