import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { DeleteAccountModal } from './DeleteAccountModal';

function RoleBadge({ active, label }: { active: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
        active
          ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
          : 'bg-gray-50 text-gray-500 border-gray-200'
      }`}
    >
      {label}
    </span>
  );
}

export function Settings() {
  const { user, isAdmin, isUploader, isReviewer } = useAuth();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  return (
    <div className="max-w-2xl space-y-6">
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">Account</h2>
        <div className="space-y-3">
          <div>
            <span className="text-sm text-[var(--color-text-secondary)]">Email</span>
            <p className="text-sm font-medium text-[var(--color-text-primary)]">{user?.email}</p>
          </div>
          <div>
            <span className="text-sm text-[var(--color-text-secondary)] block mb-1.5">Roles</span>
            <div className="flex flex-wrap gap-1.5">
              <RoleBadge active={isAdmin} label="Admin" />
              <RoleBadge active={isUploader} label="Uploader" />
              <RoleBadge active={isReviewer} label="Reviewer" />
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg border border-red-200 p-6">
        <h2 className="text-lg font-semibold text-red-700 mb-1">Danger Zone</h2>
        <p className="text-sm text-[var(--color-text-secondary)] mb-4">
          Permanently delete your account. This cannot be undone.
        </p>
        <button
          type="button"
          onClick={() => setShowDeleteModal(true)}
          className="px-4 py-2 bg-white border border-red-300 text-red-700 text-sm font-semibold rounded-md hover:bg-red-50 transition-colors"
        >
          Delete Account
        </button>
      </div>

      {showDeleteModal && <DeleteAccountModal onClose={() => setShowDeleteModal(false)} />}
    </div>
  );
}
