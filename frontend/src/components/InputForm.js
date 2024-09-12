// frontend/src/components/InputForm.js

import React, { useState } from 'react';

const InputForm = ({ onSubmit }) => {
  const [positive, setPositive] = useState('');
  const [negative, setNegative] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      positive: positive.split(',').map(word => word.trim()),
      negative: negative.split(',').map(word => word.trim()),
    });
  };

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={positive}
        onChange={(e) => setPositive(e.target.value)}
        placeholder="Positive words (comma-separated)"
      />
      <input
        type="text"
        value={negative}
        onChange={(e) => setNegative(e.target.value)}
        placeholder="Negative words (comma-separated)"
      />
      <button type="submit">Calculate</button>
    </form>
  );
};

export default InputForm;