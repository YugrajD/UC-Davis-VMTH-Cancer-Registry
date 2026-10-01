import { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useLocalStorageState } from '../../hooks/useLocalStorageState';
import { HIDE_PREDICTIONS_BY_DEFAULT_KEY } from '../../lib/auditWorklistPrefs';
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
  const [hidePredictions, setHidePredictions] = useLocalStorageState(HIDE_PREDICTIONS_BY_DEFAULT_KEY, false);

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

      {isAdmin && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-[var(--color-text-primary)] mb-1">Audit Worklist</h2>
          <p className="text-sm text-[var(--color-text-secondary)] mb-4">
            Preferences for reviewing cases on the Audit Worklist.
          </p>
          <label className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={hidePredictions}
              onChange={(e) => setHidePredictions(e.target.checked)}
              className="mt-0.5"
            />
            <span>
              <span className="block text-sm font-medium text-[var(--color-text-primary)]">
                Hide model predictions by default
              </span>
              <span className="block text-sm text-[var(--color-text-secondary)]">
                When opening a case with no prior review, the model's predicted codes stay hidden
                until you choose to reveal them — so your own read of the case isn't influenced by
                what the model guessed. A case you've already reviewed still shows your own answer.
              </span>
            </span>
          </label>
        </div>
      )}

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
