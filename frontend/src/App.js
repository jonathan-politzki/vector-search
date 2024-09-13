// frontend/src/App.js

import React, { useState } from 'react';
import { Container, Typography, Box, Snackbar, Alert } from '@mui/material';
import InputForm from './components/InputForm';
import ResultDisplay from './components/ResultDisplay';

function App() {
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');

  const handleSubmit = async (data) => {
    try {
      const response = await fetch('http://localhost:5000/api/operate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errorData = await response.json();
        setError(errorData.error || 'An error occurred');
        setResults([]);
      } else {
        const results = await response.json();
        setResults(results);
        setError('');
      }
    } catch (error) {
      console.error('Error:', error);
      setError('An unexpected error occurred');
      setResults([]);
    }
  };

  const handleCloseSnackbar = () => {
    setError('');
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Embedding Space Navigator
        </Typography>
        <InputForm onSubmit={handleSubmit} />
        <ResultDisplay results={results} />
      </Box>
      <Snackbar
        open={!!error}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleCloseSnackbar} severity="error" sx={{ width: '100%' }}>
          {error}
        </Alert>
      </Snackbar>
    </Container>
  );
}

export default App;
