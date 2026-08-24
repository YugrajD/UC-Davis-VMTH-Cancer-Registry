import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { deleteAccount } from '../../api/client';

interface DeleteAccountModalProps {
  onClose: () => void;
}

export function DeleteAccountModal({ onClose }: DeleteAccountModalProps) {
  const { user, getAccessToken, signOut } = useAuth();
  const [confirmEmail, setConfirmEmail] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const email = user?.email ?? '';
  const canDelete = confirmEmail.trim().toLowerCase() === email.toLowerCase() && !loading;

  const handleDelete = async () => {
    setError(null);
    setLoading(true);
    try {
      const token = await getAccessToken();
      if (!token) {
        setError('Your session has expired. Please sign in again.');
        return;
      }
      await deleteAccount(token);
      await signOut();
      window.location.reload();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete account');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/50" onClick={loading ? undefined : onClose} />
      <div className="relative bg-white rounded-xl shadow-2xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-red-700">Delete Account</h2>
        </div>

        <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4 space-y-2">
          <p className="text-sm font-medium text-red-800">This action is permanent and cannot be undone.</p>
          <ul className="text-sm text-red-700 list-disc pl-4 space-y-1">
            <li>This device will be signed out immediately. Any other devices you&apos;re signed into will lose access once their session expires.</li>
            <li>Your account and role assignment ({email}) will be permanently deleted.</li>
            <li>Registry data you previously uploaded or reviewed is not deleted — it stays in the system.</li>
          </ul>
        </div>

        <label className="block text-sm font-medium text-gray-700 mb-1">
          Type your email address to confirm: <span className="font-semibold">{email}</span>
        </label>
        <input
          type="email"
          value={confirmEmail}
          onChange={(e) => setConfirmEmail(e.target.value)}
          disabled={loading}
          autoFocus
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-red-400 focus:border-transparent disabled:opacity-50"
          placeholder={email}
          autoComplete="off"
        />

        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-3 mt-3">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        <div className="flex gap-3 mt-6">
          <button
            type="button"
            onClick={onClose}
            disabled={loading}
            className="flex-1 py-2.5 bg-gray-100 text-gray-700 text-sm font-semibold rounded-md hover:bg-gray-200 disabled:opacity-50 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleDelete}
            disabled={!canDelete}
            className="flex-1 py-2.5 bg-red-600 text-white text-sm font-semibold rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? 'Deleting...' : 'Permanently Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}
