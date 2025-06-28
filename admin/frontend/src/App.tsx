import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { AuthProvider } from './contexts/AuthContext';
import LoginPage from './pages/LoginPage';
import DashboardPage from './pages/DashboardPage';
import ModelsPage from './pages/ModelsPage';
import HardwarePage from './pages/HardwarePage';
import UsecasesPage from './pages/UsecasesPage';
import FeedbackPage from './pages/FeedbackPage';
import AuditLogsPage from './pages/AuditLogsPage';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  return (
    <AuthProvider>
      <Router>
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<ProtectedRoute><Layout /></ProtectedRoute>}>
            <Route index element={<Navigate to="/dashboard" replace />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="models" element={<ModelsPage />} />
            <Route path="hardware" element={<HardwarePage />} />
            <Route path="usecases" element={<UsecasesPage />} />
            <Route path="feedback" element={<FeedbackPage />} />
            <Route path="audit-logs" element={<AuditLogsPage />} />
          </Route>
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;