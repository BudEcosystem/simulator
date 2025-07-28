import React, { useState } from 'react';
import { X, FileJson } from 'lucide-react';

interface ModelFormProps {
  model: any;
  onSave: (data: any) => void;
  onCancel: () => void;
}

const ModelForm: React.FC<ModelFormProps> = ({ model, onSave, onCancel }) => {
  const [formData, setFormData] = useState({
    model_id: model?.model_id || '',
    model_name: model?.model_name || '',
    model_type: model?.model_type || 'transformer',
    attention_type: model?.attention_type || 'MHA',
    parameter_count: model?.parameter_count || 0,
    config_json: model?.config_json || {},
    logo: model?.logo || '',
    model_analysis: model?.model_analysis || {
      advantages: [],
      disadvantages: [],
      use_cases: []
    }
  });
  
  const [showConfigEditor, setShowConfigEditor] = useState(false);
  const [configJsonText, setConfigJsonText] = useState(
    JSON.stringify(formData.config_json, null, 2)
  );
  const [configError, setConfigError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate config JSON if it was edited
    if (showConfigEditor) {
      try {
        const parsedConfig = JSON.parse(configJsonText);
        formData.config_json = parsedConfig;
      } catch (error) {
        setConfigError('Invalid JSON format');
        return;
      }
    }
    
    onSave(formData);
  };

  const handleConfigJsonChange = (value: string) => {
    setConfigJsonText(value);
    setConfigError('');
    
    // Try to parse to validate
    try {
      JSON.parse(value);
    } catch (error) {
      setConfigError('Invalid JSON format');
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex min-h-screen items-center justify-center px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75" onClick={onCancel}></div>

        <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-2xl sm:w-full sm:p-6">
          <div className="absolute top-0 right-0 pt-4 pr-4">
            <button onClick={onCancel} className="text-gray-400 hover:text-gray-500">
              <X className="h-6 w-6" />
            </button>
          </div>

          <form onSubmit={handleSubmit}>
            <h3 className="text-lg leading-6 font-medium text-gray-900 mb-4">
              {model ? 'Edit Model' : 'Create Model'}
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700">Model ID</label>
                <input
                  type="text"
                  required
                  disabled={!!model}
                  value={formData.model_id}
                  onChange={(e) => setFormData({ ...formData, model_id: e.target.value })}
                  className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm disabled:bg-gray-100"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Model Name</label>
                <input
                  type="text"
                  required
                  value={formData.model_name}
                  onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                  className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Model Type</label>
                  <select
                    value={formData.model_type}
                    onChange={(e) => setFormData({ ...formData, model_type: e.target.value })}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                  >
                    <option value="transformer">Transformer</option>
                    <option value="mamba">Mamba</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Attention Type</label>
                  <select
                    value={formData.attention_type}
                    onChange={(e) => setFormData({ ...formData, attention_type: e.target.value })}
                    className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                  >
                    <option value="MHA">MHA</option>
                    <option value="MQA">MQA</option>
                    <option value="GQA">GQA</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Parameter Count</label>
                <input
                  type="number"
                  required
                  value={formData.parameter_count}
                  onChange={(e) => setFormData({ ...formData, parameter_count: parseInt(e.target.value) })}
                  className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700">Logo URL</label>
                <input
                  type="text"
                  value={formData.logo}
                  onChange={(e) => setFormData({ ...formData, logo: e.target.value })}
                  className="mt-1 block w-full border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500 sm:text-sm"
                />
              </div>

              {/* Config JSON Editor */}
              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="block text-sm font-medium text-gray-700">Config JSON</label>
                  <button
                    type="button"
                    onClick={() => setShowConfigEditor(!showConfigEditor)}
                    className="inline-flex items-center text-sm text-primary-600 hover:text-primary-700"
                  >
                    <FileJson className="h-4 w-4 mr-1" />
                    {showConfigEditor ? 'Hide Editor' : 'Edit Config'}
                  </button>
                </div>
                
                {showConfigEditor && (
                  <div>
                    <textarea
                      value={configJsonText}
                      onChange={(e) => handleConfigJsonChange(e.target.value)}
                      className="mt-1 block w-full h-64 font-mono text-xs border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
                      placeholder="Enter valid JSON configuration..."
                    />
                    {configError && (
                      <p className="mt-1 text-sm text-red-600">{configError}</p>
                    )}
                  </div>
                )}
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

export default ModelForm;