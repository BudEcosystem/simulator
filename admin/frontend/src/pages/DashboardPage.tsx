import React, { useState, useEffect } from 'react';
import { 
  Users, 
  Globe, 
  TrendingUp, 
  Activity,
  Package,
  Cpu,
  FileText,
  MessageSquare
} from 'lucide-react';
import { analyticsService, feedbackService } from '../services';
import { AnalyticsSummary, SystemHealth } from '../types/analytics';
import { 
  LineChart, 
  Line, 
  BarChart, 
  Bar, 
  PieChart, 
  Pie, 
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer 
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8'];

const DashboardPage: React.FC = () => {
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [feedbackStats, setFeedbackStats] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      const [analyticsData, healthData, feedbackData] = await Promise.all([
        analyticsService.getSummary(30),
        analyticsService.getSystemHealth(),
        feedbackService.getFeedbackStats()
      ]);
      
      setAnalytics(analyticsData);
      setSystemHealth(healthData);
      setFeedbackStats(feedbackData);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch dashboard data');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 p-4 rounded-md">
        <p className="text-red-800">Error: {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-500">
          Overview of system analytics and performance
        </p>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Users className="h-6 w-6 text-gray-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Total Users
                  </dt>
                  <dd className="text-lg font-semibold text-gray-900">
                    {analytics?.total_users || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Activity className="h-6 w-6 text-green-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Active Users (24h)
                  </dt>
                  <dd className="text-lg font-semibold text-gray-900">
                    {analytics?.active_users || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <MessageSquare className="h-6 w-6 text-blue-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Pending Feedback
                  </dt>
                  <dd className="text-lg font-semibold text-gray-900">
                    {feedbackStats?.status_breakdown?.pending || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <TrendingUp className="h-6 w-6 text-purple-400" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Custom SLOs Used
                  </dt>
                  <dd className="text-lg font-semibold text-gray-900">
                    {analytics?.custom_slos_usage || 0}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* API Usage Timeline */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">API Usage (Last 30 Days)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={analytics?.api_usage_timeline || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Device Breakdown */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Device Breakdown</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={analytics?.device_breakdown || []}
                dataKey="count"
                nameKey="device"
                cx="50%"
                cy="50%"
                outerRadius={80}
                label
              >
                {analytics?.device_breakdown.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Popular Models */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Popular Models</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={analytics?.popular_models.slice(0, 5) || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="id" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Countries */}
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Users by Country</h3>
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {analytics?.users_by_country.map((country, index) => (
              <div key={index} className="flex items-center justify-between py-2 border-b">
                <div className="flex items-center">
                  <Globe className="h-4 w-4 text-gray-400 mr-2" />
                  <span className="text-sm text-gray-900">{country.country}</span>
                </div>
                <span className="text-sm font-medium text-gray-900">{country.count}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* System Health */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-medium text-gray-900 mb-4">System Health</h3>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="border rounded-lg p-4">
            <p className="text-sm text-gray-500">Status</p>
            <p className={`text-lg font-semibold ${
              systemHealth?.status === 'healthy' ? 'text-green-600' : 'text-red-600'
            }`}>
              {systemHealth?.status || 'Unknown'}
            </p>
          </div>
          <div className="border rounded-lg p-4">
            <p className="text-sm text-gray-500">Avg Response Time</p>
            <p className="text-lg font-semibold text-gray-900">
              {systemHealth?.avg_response_time.toFixed(2) || 0} ms
            </p>
          </div>
          <div className="border rounded-lg p-4">
            <p className="text-sm text-gray-500">Errors (Last Hour)</p>
            <p className="text-lg font-semibold text-gray-900">
              {systemHealth?.error_count || 0}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;