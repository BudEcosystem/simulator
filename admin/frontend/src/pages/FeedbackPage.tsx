import React, { useState, useEffect } from 'react';
import { MessageSquare, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { feedbackService } from '../services';
import toast from 'react-hot-toast';
import { format } from 'date-fns';

const FeedbackPage: React.FC = () => {
  const [feedback, setFeedback] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedFeedback, setSelectedFeedback] = useState<any>(null);
  const [response, setResponse] = useState('');
  const [isInternal, setIsInternal] = useState(false);

  useEffect(() => {
    fetchFeedback();
  }, []);

  const fetchFeedback = async () => {
    try {
      setLoading(true);
      const data = await feedbackService.getFeedback({ limit: 100 });
      setFeedback(data);
    } catch (error) {
      toast.error('Failed to fetch feedback');
    } finally {
      setLoading(false);
    }
  };

  const handleStatusChange = async (id: number, status: string) => {
    try {
      await feedbackService.updateFeedback(id, { status });
      toast.success('Status updated');
      fetchFeedback();
    } catch (error) {
      toast.error('Failed to update status');
    }
  };

  const handleRespond = async () => {
    if (!selectedFeedback || !response.trim()) return;

    try {
      await feedbackService.respondToFeedback(selectedFeedback.id, {
        message: response,
        is_internal: isInternal
      });
      toast.success('Response sent');
      setResponse('');
      setSelectedFeedback(null);
      fetchFeedback();
    } catch (error) {
      toast.error('Failed to send response');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'pending':
        return <Clock className="h-5 w-5 text-yellow-500" />;
      case 'in_progress':
        return <AlertCircle className="h-5 w-5 text-blue-500" />;
      case 'resolved':
        return <CheckCircle className="h-5 w-5 text-green-500" />;
      default:
        return <MessageSquare className="h-5 w-5 text-gray-500" />;
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-gray-900">Feedback</h1>
        <p className="mt-1 text-sm text-gray-500">
          Manage and respond to user feedback
        </p>
      </div>

      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        {loading ? (
          <div className="p-6 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          </div>
        ) : feedback.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No feedback found
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {feedback.map((item) => (
              <li key={item.id} className="p-4 hover:bg-gray-50">
                <div className="flex items-start space-x-3">
                  {getStatusIcon(item.status)}
                  <div className="flex-1">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-medium text-gray-900">{item.title}</h3>
                      <span className="text-xs text-gray-500">
                        {format(new Date(item.created_at), 'MMM d, yyyy')}
                      </span>
                    </div>
                    <p className="mt-1 text-sm text-gray-600">{item.message}</p>
                    <div className="mt-2 flex items-center space-x-4 text-xs">
                      <span className="text-gray-500">Type: {item.feedback_type}</span>
                      <span className="text-gray-500">Category: {item.category}</span>
                      {item.rating && <span className="text-gray-500">Rating: {item.rating}/5</span>}
                    </div>
                    <div className="mt-3 flex items-center space-x-2">
                      <select
                        value={item.status}
                        onChange={(e) => handleStatusChange(item.id, e.target.value)}
                        className="text-sm border-gray-300 rounded-md focus:ring-primary-500 focus:border-primary-500"
                      >
                        <option value="pending">Pending</option>
                        <option value="in_progress">In Progress</option>
                        <option value="resolved">Resolved</option>
                        <option value="closed">Closed</option>
                      </select>
                      <button
                        onClick={() => setSelectedFeedback(item)}
                        className="text-sm text-primary-600 hover:text-primary-700"
                      >
                        Respond
                      </button>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Response Modal */}
      {selectedFeedback && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex min-h-screen items-center justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={() => setSelectedFeedback(null)}></div>
            <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
              <h3 className="text-lg font-medium text-gray-900 mb-4">Respond to Feedback</h3>
              <textarea
                value={response}
                onChange={(e) => setResponse(e.target.value)}
                rows={4}
                className="w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                placeholder="Type your response..."
              />
              <div className="mt-3">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={isInternal}
                    onChange={(e) => setIsInternal(e.target.checked)}
                    className="rounded text-primary-600 focus:ring-primary-500"
                  />
                  <span className="ml-2 text-sm text-gray-600">Internal note (not visible to user)</span>
                </label>
              </div>
              <div className="mt-4 flex justify-end space-x-3">
                <button
                  onClick={() => setSelectedFeedback(null)}
                  className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={handleRespond}
                  className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700"
                >
                  Send Response
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FeedbackPage;