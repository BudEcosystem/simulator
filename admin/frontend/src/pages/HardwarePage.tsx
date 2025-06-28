import React, { useState, useEffect } from 'react';
import { Plus, Edit2, Trash2, Search } from 'lucide-react';
import { crudService } from '../services';
import toast from 'react-hot-toast';
import HardwareForm from '../components/HardwareForm';
import ConfirmDialog from '../components/ConfirmDialog';

const HardwarePage: React.FC = () => {
  const [hardware, setHardware] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingHardware, setEditingHardware] = useState<any>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    fetchHardware();
  }, [search]);

  const fetchHardware = async () => {
    try {
      setLoading(true);
      const response = await crudService.getHardware({ search, limit: 100 });
      // Handle response structure - it may be an array or object with hardware array
      setHardware(Array.isArray(response) ? response : response.hardware || []);
    } catch (error) {
      toast.error('Failed to fetch hardware');
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = () => {
    setEditingHardware(null);
    setShowForm(true);
  };

  const handleEdit = (item: any) => {
    setEditingHardware(item);
    setShowForm(true);
  };

  const handleDelete = async (name: string) => {
    try {
      await crudService.deleteHardware(name);
      toast.success('Hardware deleted successfully');
      fetchHardware();
    } catch (error) {
      toast.error('Failed to delete hardware');
    }
    setDeleteConfirm(null);
  };

  const handleSave = async (data: any) => {
    try {
      if (editingHardware) {
        await crudService.updateHardware(editingHardware.name, data);
        toast.success('Hardware updated successfully');
      } else {
        await crudService.createHardware(data);
        toast.success('Hardware created successfully');
      }
      setShowForm(false);
      fetchHardware();
    } catch (error) {
      toast.error('Failed to save hardware');
    }
  };

  return (
    <div className="space-y-6">
      <div className="sm:flex sm:items-center sm:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Hardware</h1>
          <p className="mt-1 text-sm text-gray-500">
            Manage hardware accelerators and specifications
          </p>
        </div>
        <button
          onClick={handleCreate}
          className="mt-4 sm:mt-0 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700"
        >
          <Plus className="h-4 w-4 mr-2" />
          Add Hardware
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
          placeholder="Search hardware..."
        />
      </div>

      {/* Hardware Table */}
      <div className="bg-white shadow overflow-hidden sm:rounded-md">
        {loading ? (
          <div className="p-6 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto"></div>
          </div>
        ) : hardware.length === 0 ? (
          <div className="p-6 text-center text-gray-500">
            No hardware found
          </div>
        ) : (
          <ul className="divide-y divide-gray-200">
            {hardware.map((item) => (
              <li key={item.name}>
                <div className="px-4 py-4 sm:px-6">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <h3 className="text-lg font-medium text-gray-900">
                        {item.name}
                      </h3>
                      <p className="text-sm text-gray-500">{item.manufacturer}</p>
                      <div className="mt-2 flex items-center text-sm text-gray-500">
                        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800 mr-2">
                          {item.type}
                        </span>
                        <span className="text-xs">
                          {item.flops} TFLOPS • {item.memory_size}GB • {item.power}W
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => handleEdit(item)}
                        className="p-2 text-gray-600 hover:text-gray-900"
                      >
                        <Edit2 className="h-5 w-5" />
                      </button>
                      <button
                        onClick={() => setDeleteConfirm(item.name)}
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

      {/* Hardware Form Modal */}
      {showForm && (
        <HardwareForm
          hardware={editingHardware}
          onSave={handleSave}
          onCancel={() => setShowForm(false)}
        />
      )}

      {/* Delete Confirmation */}
      {deleteConfirm && (
        <ConfirmDialog
          title="Delete Hardware"
          message="Are you sure you want to delete this hardware? This action cannot be undone."
          onConfirm={() => handleDelete(deleteConfirm)}
          onCancel={() => setDeleteConfirm(null)}
        />
      )}
    </div>
  );
};

export default HardwarePage;