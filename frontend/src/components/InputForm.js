// frontend/src/components/InputForm.js

import React, { useState } from 'react';
import { TextField, Button, Grid, Paper, Box } from '@mui/material';

const InputForm = ({ onSubmit }) => {
  const [positive, setPositive] = useState('');
  const [negative, setNegative] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!positive && !negative) {
      alert('Please enter at least one positive or negative word.');
      return;
    }
    onSubmit({
      positive: positive.split(',').map((word) => word.trim()).filter(Boolean),
      negative: negative.split(',').map((word) => word.trim()).filter(Boolean),
    });
  };

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
      <form onSubmit={handleSubmit}>
        <Grid container spacing={2} alignItems="center">
          <Grid item xs={12} sm={6}>
            <TextField
              label="Positive Words"
              variant="outlined"
              fullWidth
              value={positive}
              onChange={(e) => setPositive(e.target.value)}
              placeholder="e.g., king, woman"
              helperText="Comma-separated words to add"
            />
          </Grid>
          <Grid item xs={12} sm={6}>
            <TextField
              label="Negative Words"
              variant="outlined"
              fullWidth
              value={negative}
              onChange={(e) => setNegative(e.target.value)}
              placeholder="e.g., man"
              helperText="Comma-separated words to subtract"
            />
          </Grid>
          <Grid item xs={12} sx={{ textAlign: 'center', mt: 2 }}>
            <Button variant="contained" color="primary" type="submit" size="large">
              Calculate
            </Button>
          </Grid>
        </Grid>
      </form>
    </Paper>
  );
};

export default InputForm;
