import React from 'react';

const ResultDisplay = ({ results }) => {
  return (
    <div>
      <h2>Results:</h2>
      {results.error ? (
        <p>Error: {results.error}</p>
      ) : (
        <ul>
          {results.map((result, index) => (
            <li key={index}>
              {result.word}: {result.distance.toFixed(4)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};


export default ResultDisplay;