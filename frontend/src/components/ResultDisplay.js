// frontend/src/components/ResultDisplay.js

import React from 'react';
import { Typography, List, ListItem, ListItemText, Paper, Box } from '@mui/material';

const ResultDisplay = ({ results }) => {
  if (!results || results.length === 0) {
    return null;
  }

  return (
    <Paper elevation={3} sx={{ p: 3, mt: 4 }}>
      <Typography variant="h5" gutterBottom>
        Results:
      </Typography>
      <List>
        {results.map((result, index) => (
          <ListItem key={index}>
            <ListItemText
              primary={`${index + 1}. ${result.word}`}
              secondary={`Distance: ${result.distance.toFixed(4)}`}
            />
          </ListItem>
        ))}
      </List>
    </Paper>
  );
};

export default ResultDisplay;
