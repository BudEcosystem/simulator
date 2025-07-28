import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import AIMemoryCalculator from './AIMemoryCalculator';

const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
root.render(
  <React.StrictMode>
    <AIMemoryCalculator />
  </React.StrictMode>
);