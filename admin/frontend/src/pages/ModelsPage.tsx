import React, { useState, useEffect } from 'react';
import { Plus, Edit2, Trash2, Search } from 'lucide-react';
import { crudService } from '../services';
import toast from 'react-hot-toast';
import ModelForm from '../components/ModelForm';
import ConfirmDialog from '../components/ConfirmDialog';

const ModelsPage: React.FC = () => {
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingModel, setEditingModel] = useState<any>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    fetchModels();
  }, [search]);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await crudService.getModels({ search, limit: 100 });
      // Extract models array from the response
      setModels(response.models || []);
    } catch (error) {
      toast.error('Failed to fetch models');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => {
    setEditingModel(null);
    setShowForm(true);
  };

  const handleEdit = (model: any) => {
    setEditingModel(model);
    setShowForm(true);
  };

  const handleDelete = async (modelId: string) => {
    try {
      await crudService.deleteModel(modelId);
      toast.success('Model deleted successfully');
      fetchModels();
    } catch (error) {
      toast.error('Failed to delete model');
    }
    setDeleteConfirm(null);
  };

  const handleSave = async (data: any) => {
    try {
      if (editingModel) {
        await crudService.updateModel(editingModel.model_id, data);
        toast.success('Model updated successfully');
      } else {
        await crudService.createModel(data);
        toast.success('Model created successfully');
      }
      setShowForm(false);
      fetchModels();
    } catch (error) {
      toast.error('Failed to save model');
    }
  };

  return (
    <div className="space-y-6">
      <div className="sm:flex sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Models</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage AI models and their configurations
          </p>
        </div>
        <button
          onClick={handleCreate}
          className="mt-4 sm:mt-0 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Model
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
          placeholder="Search models..."
        />
      </div>

      {/* Models Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        {loading ? (
          <div className="p-6 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          </div>
        ) : models.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No models found
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {models.map((model) => (
              <li key={model.model_id}>
                <div className="px-4 py-4 sm:px-6">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <h3 className="text-lg font-medium text-gray-900">
                        {model.model_name}
                      </h3>
                      <p className="text-sm text-gray-500">{model.model_id}</p>
                      <div className="mt-2 flex items-center text-sm text-gray-500">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 mr-2">
                          {model.model_type}
                        </span>
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 mr-2">
                          {model.attention_type}
                        </span>
                        <span className="text-xs">
                          {(model.parameter_count / 1e9).toFixed(1)}B params
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleEdit(model)}
                        className="p-2 text-gray-600 hover:text-gray-900"
                      >
                        <Edit2 className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => setDeleteConfirm(model.model_id)}
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

      {/* Model Form Modal */}
      {showForm && (
        <ModelForm
          model={editingModel}
          onSave={handleSave}
          onCancel={() => setShowForm(false)}
        />
      )}

      {/* Delete Confirmation */}
      {deleteConfirm && (
        <ConfirmDialog
          title="Delete Model"
          message="Are you sure you want to delete this model? This action cannot be undone."
          onConfirm={() => handleDelete(deleteConfirm)}
          onCancel={() => setDeleteConfirm(null)}
        />
      )}
    </div>
  );
};

export default ModelsPage;