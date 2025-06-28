import React, { useState, useEffect } from 'react';
import { Plus, Edit2, Trash2, Search } from 'lucide-react';
import { crudService } from '../services';
import toast from 'react-hot-toast';
import UsecaseForm from '../components/UsecaseForm';
import ConfirmDialog from '../components/ConfirmDialog';

const UsecasesPage: React.FC = () => {
  const [usecases, setUsecases] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingUsecase, setEditingUsecase] = useState<any>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    fetchUsecases();
  }, [search]);

  const fetchUsecases = async () => {
    try {
      setLoading(true);
      const response = await crudService.getUsecases({ search, limit: 100 });
      // Handle response structure - it may be an array or object with usecases array
      setUsecases(Array.isArray(response) ? response : response.usecases || []);
    } catch (error) {
      toast.error('Failed to fetch use cases');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => {
    setEditingUsecase(null);
    setShowForm(true);
  };

  const handleEdit = (usecase: any) => {
    setEditingUsecase(usecase);
    setShowForm(true);
  };

  const handleDelete = async (uniqueId: string) => {
    try {
      await crudService.deleteUsecase(uniqueId);
      toast.success('Use case deleted successfully');
      fetchUsecases();
    } catch (error) {
      toast.error('Failed to delete use case');
    }
    setDeleteConfirm(null);
  };

  const handleSave = async (data: any) => {
    try {
      if (editingUsecase) {
        await crudService.updateUsecase(editingUsecase.unique_id, data);
        toast.success('Use case updated successfully');
      } else {
        await crudService.createUsecase(data);
        toast.success('Use case created successfully');
      }
      setShowForm(false);
      fetchUsecases();
    } catch (error) {
      toast.error('Failed to save use case');
    }
  };

  return (
    <div className="space-y-6">
      <div className="sm:flex sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Use Cases</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage industry-specific use case configurations
          </p>
        </div>
        <button
          onClick={handleCreate}
          className="mt-4 sm:mt-0 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Use Case
        </button>
      </div>

      {/* Search */}
      <div className="relative">
        <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
          <Search className="h-5 w-5 text-gray-400" />
        </div>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md leading-5 bg-white placeholder-gray-500 focus:outline-none focus:placeholder-gray-400 focus:ring-1 focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
          placeholder="Search use cases..."
        />
      </div>

      {/* Use Cases Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        {loading ? (
          <div className="p-6 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          </div>
        ) : usecases.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No use cases found
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {usecases.map((usecase) => (
              <li key={usecase.unique_id}>
                <div className="px-4 py-4 sm:px-6">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <h3 className="text-lg font-medium text-gray-900">
                        {usecase.name}
                      </h3>
                      <p className="text-sm text-gray-500">{usecase.unique_id} • {usecase.industry}</p>
                      <div className="mt-2 text-sm text-gray-500">
                        <span className="mr-4">Input: {usecase.input_tokens_min}-{usecase.input_tokens_max} tokens</span>
                        <span>Output: {usecase.output_tokens_min}-{usecase.output_tokens_max} tokens</span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleEdit(usecase)}
                        className="p-2 text-gray-600 hover:text-gray-900"
                      >
                        <Edit2 className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => setDeleteConfirm(usecase.unique_id)}
                        className="p-2 text-red-600 hover:text-red-900"
                      >
                        <Trash2 className="h-5 w-5" />
                      </button>
                    </div>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* Usecase Form Modal */}
      {showForm && (
        <UsecaseForm
          usecase={editingUsecase}
          onSave={handleSave}
          onCancel={() => setShowForm(false)}
        />
      )}

      {/* Delete Confirmation */}
      {deleteConfirm && (
        <ConfirmDialog
          title="Delete Use Case"
          message="Are you sure you want to delete this use case? This action cannot be undone."
          onConfirm={() => handleDelete(deleteConfirm)}
          onCancel={() => setDeleteConfirm(null)}
        />
      )}
    </div>
  );
};

export default UsecasesPage;