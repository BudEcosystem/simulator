import React, { useState } from 'react';
import { X } from 'lucide-react';

interface UsecaseFormProps {
  usecase: any;
  onSave: (data: any) => void;
  onCancel: () => void;
}

const UsecaseForm: React.FC<UsecaseFormProps> = ({ usecase, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    unique_id: usecase?.unique_id || '',
    name: usecase?.name || '',
    industry: usecase?.industry || '',
    description: usecase?.description || '',
    batch_size: usecase?.batch_size || 1,
    beam_size: usecase?.beam_size || 1,
    input_tokens_min: usecase?.input_tokens_min || 0,
    input_tokens_max: usecase?.input_tokens_max || 1000,
    output_tokens_min: usecase?.output_tokens_min || 0,
    output_tokens_max: usecase?.output_tokens_max || 500,
    ttft_min: usecase?.ttft_min || null,
    ttft_max: usecase?.ttft_max || null,
    e2e_min: usecase?.e2e_min || null,
    e2e_max: usecase?.e2e_max || null,
    inter_token_min: usecase?.inter_token_min || null,
    inter_token_max: usecase?.inter_token_max || null,
    tags: usecase?.tags || []
  });

  const [tagInput, setTagInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  const addTag = () => {
    if (tagInput.trim() && !formData.tags.includes(tagInput.trim())) {
      setFormData({
        ...formData,
        tags: [...formData.tags, tagInput.trim()]
      });
      setTagInput('');
    }
  };

  const removeTag = (tag: string) => {
    setFormData({
      ...formData,
      tags: formData.tags.filter((t: string) => t !== tag)
    });
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={onCancel}></div>

        <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-3xl sm:w-full sm:p-6">
          <div className="absolute top-0 right-0 pt-4 pr-4">
            <button onClick={onCancel} className="text-gray-400 hover:text-gray-500">
              <X className="h-6 w-6" />
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              {usecase ? 'Edit Use Case' : 'Create Use Case'}
            </h3>

            <div className="space-y-4">
              {/* Basic Information */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Unique ID</label>
                  <input
                    type="text"
                    required
                    disabled={!!usecase}
                    value={formData.unique_id}
                    onChange={(e) => setFormData({ ...formData, unique_id: e.target.value })}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm disabled:bg-gray-100"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Name</label>
                  <input
                    type="text"
                    required
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Industry</label>
                <input
                  type="text"
                  required
                  value={formData.industry}
                  onChange={(e) => setFormData({ ...formData, industry: e.target.value })}
                  className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Description</label>
                <textarea
                  rows={3}
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                />
              </div>

              {/* Configuration */}
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Configuration</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Batch Size</label>
                    <input
                      type="number"
                      min="1"
                      value={formData.batch_size}
                      onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Beam Size</label>
                    <input
                      type="number"
                      min="1"
                      value={formData.beam_size}
                      onChange={(e) => setFormData({ ...formData, beam_size: parseInt(e.target.value) })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>
                </div>
              </div>

              {/* Token Ranges */}
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Token Ranges</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Input Tokens Min</label>
                    <input
                      type="number"
                      min="0"
                      required
                      value={formData.input_tokens_min}
                      onChange={(e) => setFormData({ ...formData, input_tokens_min: parseInt(e.target.value) })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Input Tokens Max</label>
                    <input
                      type="number"
                      min="0"
                      required
                      value={formData.input_tokens_max}
                      onChange={(e) => setFormData({ ...formData, input_tokens_max: parseInt(e.target.value) })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Output Tokens Min</label>
                    <input
                      type="number"
                      min="0"
                      required
                      value={formData.output_tokens_min}
                      onChange={(e) => setFormData({ ...formData, output_tokens_min: parseInt(e.target.value) })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">Output Tokens Max</label>
                    <input
                      type="number"
                      min="0"
                      required
                      value={formData.output_tokens_max}
                      onChange={(e) => setFormData({ ...formData, output_tokens_max: parseInt(e.target.value) })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>
                </div>
              </div>

              {/* SLOs */}
              <div className="border-t pt-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Service Level Objectives (Optional)</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">TTFT Min (seconds)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.ttft_min || ''}
                      onChange={(e) => setFormData({ ...formData, ttft_min: e.target.value ? parseFloat(e.target.value) : null })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">TTFT Max (seconds)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.ttft_max || ''}
                      onChange={(e) => setFormData({ ...formData, ttft_max: e.target.value ? parseFloat(e.target.value) : null })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">E2E Min (seconds)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.e2e_min || ''}
                      onChange={(e) => setFormData({ ...formData, e2e_min: e.target.value ? parseFloat(e.target.value) : null })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700">E2E Max (seconds)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.e2e_max || ''}
                      onChange={(e) => setFormData({ ...formData, e2e_max: e.target.value ? parseFloat(e.target.value) : null })}
                      className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                    />
                  </div>
                </div>
              </div>

              {/* Tags */}
              <div className="border-t pt-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">Tags</label>
                <div className="flex items-center gap-2 mb-2">
                  <input
                    type="text"
                    value={tagInput}
                    onChange={(e) => setTagInput(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && (e.preventDefault(), addTag())}
                    placeholder="Add a tag..."
                    className="flex-1 border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                  />
                  <button
                    type="button"
                    onClick={addTag}
                    className="px-3 py-2 border border-primary-600 text-primary-600 rounded-md hover:bg-primary-50"
                  >
                    Add
                  </button>
                </div>
                <div className="flex flex-wrap gap-2">
                  {formData.tags.map((tag: string) => (
                    <span
                      key={tag}
                      className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-primary-100 text-primary-800"
                    >
                      {tag}
                      <button
                        type="button"
                        onClick={() => removeTag(tag)}
                        className="ml-1.5 text-primary-600 hover:text-primary-800"
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                onClick={onCancel}
                className="px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="px-4 py-2 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
              >
                Save
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default UsecaseForm;