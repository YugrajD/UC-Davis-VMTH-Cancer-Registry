import { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import {
  fetchUserRoles,
  fetchPendingRoleRequests,
  resolveRoleRequest,
  fetchPendingExportRequests,
  resolveExportRequest,
  isValidEmail,
  updateUserRoles,
  type UserRoles,
  type RoleRequest,
  type ExportRequest,
} from '../../api/client';

function StatusBadge({ value, label }: { value: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium border ${
        value
          ? 'bg-emerald-100 text-emerald-800 border-emerald-200'
          : 'bg-gray-50 text-gray-500 border-gray-200'
      }`}
    >
      {label}: {value ? 'yes' : 'no'}
    </span>
  );
}

export function UserManagement() {
  const { user, getAccessToken } = useAuth();
  const [emailInput, setEmailInput] = useState('');
  const [loadedEmail, setLoadedEmail] = useState<string | null>(null);
  const [roles, setRoles] = useState<UserRoles | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [savedAt, setSavedAt] = useState<string | null>(null);

  // Pending role requests
  const [pendingRequests, setPendingRequests] = useState<RoleRequest[]>([]);
  const [pendingLoading, setPendingLoading] = useState(false);
  const [resolvingId, setResolvingId] = useState<number | null>(null);
  const [resolveError, setResolveError] = useState<string | null>(null);

  // Pending export requests
  const [pendingExportRequests, setPendingExportRequests] = useState<ExportRequest[]>([]);
  const [pendingExportLoading, setPendingExportLoading] = useState(false);
  const [resolvingExportId, setResolvingExportId] = useState<number | null>(null);

  // Form state for the toggles. Synced from `roles` whenever a new
  // record is loaded.
  const [formAdmin, setFormAdmin] = useState(false);
  const [formUploader, setFormUploader] = useState(false);
  const [formReviewer, setFormReviewer] = useState(false);

  const loadPendingRequests = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    setPendingLoading(true);
    try {
      const reqs = await fetchPendingRoleRequests(token);
      setPendingRequests(reqs);
    } catch {
      // silently fail
    } finally {
      setPendingLoading(false);
    }
  }, [getAccessToken]);

  const loadPendingExportRequests = useCallback(async () => {
    const token = await getAccessToken();
    if (!token) return;
    setPendingExportLoading(true);
    try {
      const reqs = await fetchPendingExportRequests(token);
      setPendingExportRequests(reqs);
    } catch {
      // silently fail
    } finally {
      setPendingExportLoading(false);
    }
  }, [getAccessToken]);

  useEffect(() => {
    loadPendingRequests();
    loadPendingExportRequests();
  }, [loadPendingRequests, loadPendingExportRequests]);

  const handleResolve = useCallback(async (requestId: number, action: 'approve' | 'deny') => {
    const token = await getAccessToken();
    if (!token) return;
    setResolveError(null);
    setResolvingId(requestId);
    try {
      await resolveRoleRequest(token, requestId, action);
      await loadPendingRequests();
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : 'Failed to resolve request');
    } finally {
      setResolvingId(null);
    }
  }, [getAccessToken, loadPendingRequests]);

  const handleResolveExport = useCallback(async (requestId: number, action: 'approve' | 'deny') => {
    const token = await getAccessToken();
    if (!token) return;
    setResolveError(null);
    setResolvingExportId(requestId);
    try {
      await resolveExportRequest(token, requestId, action);
      await loadPendingExportRequests();
    } catch (err) {
      setResolveError(err instanceof Error ? err.message : 'Failed to resolve request');
    } finally {
      setResolvingExportId(null);
    }
  }, [getAccessToken, loadPendingExportRequests]);

  const isSelf =
    user?.email && roles?.email && roles.email.toLowerCase() === user.email.toLowerCase();

  const lookup = useCallback(async () => {
    setError(null);
    setSavedAt(null);
    if (!isValidEmail(emailInput)) {
      setError('Enter a valid email address.');
      return;
    }
    const token = await getAccessToken();
    if (!token) {
      setError('Not signed in.');
      return;
    }
    setLoading(true);
    try {
      const data = await fetchUserRoles(token, emailInput);
      setRoles(data);
      setLoadedEmail(data.email);
      setFormAdmin(data.is_admin);
      // Admin implies the lower roles, so the form mirrors that.
      setFormUploader(data.is_admin || data.is_uploader);
      setFormReviewer(data.is_admin || data.is_reviewer);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Lookup failed');
      setRoles(null);
      setLoadedEmail(null);
    } finally {
      setLoading(false);
    }
  }, [emailInput, getAccessToken]);

  const save = useCallback(async () => {
    if (!loadedEmail) return;
    setError(null);
    setSavedAt(null);
    const token = await getAccessToken();
    if (!token) {
      setError('Not signed in.');
      return;
    }
    setSaving(true);
    try {
      const data = await updateUserRoles(token, loadedEmail, {
        is_admin: formAdmin,
        is_uploader: formAdmin || formUploader,
        is_reviewer: formAdmin || formReviewer,
      });
      setRoles(data);
      setFormAdmin(data.is_admin);
      setFormUploader(data.is_uploader);
      setFormReviewer(data.is_reviewer);
      setSavedAt(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Save failed');
    } finally {
      setSaving(false);
    }
  }, [getAccessToken, loadedEmail, formAdmin, formUploader, formReviewer]);

  const dirty =
    roles !== null &&
    (formAdmin !== roles.is_admin ||
      formUploader !== (roles.is_admin || roles.is_uploader) ||
      formReviewer !== (roles.is_admin || roles.is_reviewer));

  return (
    <div className="space-y-4">
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
          User Management
        </h2>
        <p className="text-sm text-[var(--color-text-secondary)] mt-1">
          Look up a user by email and edit their roles. Admin implies
          uploader and reviewer; un-checking those while admin is on has
          no effect. You cannot remove your own admin role.
        </p>
      </div>

      {/* Pending Role Requests */}
      {pendingRequests.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
              Pending Role Requests
            </h3>
            <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 text-xs font-semibold rounded-full bg-amber-500 text-white">
              {pendingRequests.length}
            </span>
          </div>
          <div className="divide-y divide-gray-100">
            {pendingRequests.map((req) => (
              <div key={req.id} className="px-4 py-3 flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-[var(--color-text-primary)] truncate">{req.email}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      {req.requested_role}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(req.created_at).toLocaleString()}
                    </span>
                  </div>
                  {req.reason && (
                    <p className="text-xs text-gray-600 mt-1">{req.reason}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => handleResolve(req.id, 'approve')}
                    disabled={resolvingId === req.id}
                    className="px-3 py-1.5 text-xs font-medium bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => handleResolve(req.id, 'deny')}
                    disabled={resolvingId === req.id}
                    className="px-3 py-1.5 text-xs font-medium bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 transition-colors"
                  >
                    Deny
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {pendingLoading && pendingRequests.length === 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 flex justify-center">
          <div className="w-5 h-5 border-2 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
        </div>
      )}

      {/* Pending Export Requests */}
      {pendingExportRequests.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)]">
              Pending Export Requests
            </h3>
            <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 text-xs font-semibold rounded-full bg-amber-500 text-white">
              {pendingExportRequests.length}
            </span>
          </div>
          <div className="divide-y divide-gray-100">
            {pendingExportRequests.map((req) => (
              <div key={req.id} className="px-4 py-3 flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-[var(--color-text-primary)] truncate">{req.email}</p>
                  <div className="flex items-center gap-2 mt-1">
                    <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                      data export
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(req.created_at).toLocaleString()}
                    </span>
                  </div>
                  {req.reason && (
                    <p className="text-xs text-gray-600 mt-1">{req.reason}</p>
                  )}
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => handleResolveExport(req.id, 'approve')}
                    disabled={resolvingExportId === req.id}
                    className="px-3 py-1.5 text-xs font-medium bg-emerald-600 text-white rounded hover:bg-emerald-700 disabled:opacity-50 transition-colors"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => handleResolveExport(req.id, 'deny')}
                    disabled={resolvingExportId === req.id}
                    className="px-3 py-1.5 text-xs font-medium bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 transition-colors"
                  >
                    Deny
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {pendingExportLoading && pendingExportRequests.length === 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 flex justify-center">
          <div className="w-5 h-5 border-2 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
        </div>
      )}

      {resolveError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{resolveError}</p>
        </div>
      )}

      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <label htmlFor="email-input" className="block text-xs font-semibold uppercase tracking-wider text-[var(--color-text-primary)] mb-2">
          Email
        </label>
        <div className="flex gap-2">
          <input
            id="email-input"
            type="email"
            value={emailInput}
            onChange={(e) => setEmailInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') lookup();
            }}
            placeholder="user@ucdavis.edu"
            className="flex-1 px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]"
          />
          <button
            onClick={lookup}
            disabled={loading || !emailInput.trim()}
            className="px-4 py-2 text-sm font-medium bg-[var(--color-teal)] text-white rounded hover:bg-[var(--color-teal-dark)] disabled:opacity-50"
          >
            {loading ? 'Looking up…' : 'Look up'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {roles && (
        <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-[var(--color-text-primary)]">{roles.email}</p>
              {!roles.persisted && (
                <p className="text-xs text-amber-700 mt-0.5">
                  No DB record yet — values shown are from env fallback. Saving will create a row.
                </p>
              )}
              {roles.persisted && roles.updated_by_email && roles.updated_at && (
                <p className="text-xs text-gray-500 mt-0.5">
                  Last updated {new Date(roles.updated_at).toLocaleString()} by {roles.updated_by_email}
                </p>
              )}
            </div>
            <div className="flex gap-1.5">
              <StatusBadge value={roles.is_admin} label="admin" />
              <StatusBadge value={roles.is_uploader} label="uploader" />
              <StatusBadge value={roles.is_reviewer} label="reviewer" />
            </div>
          </div>

          <fieldset className="border-t border-gray-200 pt-4 space-y-2">
            <legend className="sr-only">Roles</legend>
            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                checked={formAdmin}
                onChange={(e) => setFormAdmin(e.target.checked)}
                disabled={Boolean(isSelf) && roles.is_admin}
                className="mt-1"
              />
              <span>
                <span className="font-medium">Admin</span>
                <span className="block text-xs text-gray-500">
                  Full access; implicitly grants uploader and reviewer.
                  {isSelf && roles.is_admin && (
                    <span className="ml-1 text-amber-700">
                      Locked — you cannot demote yourself.
                    </span>
                  )}
                </span>
              </span>
            </label>
            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                checked={formAdmin || formUploader}
                onChange={(e) => setFormUploader(e.target.checked)}
                disabled={formAdmin}
                className="mt-1"
              />
              <span>
                <span className="font-medium">Uploader</span>
                <span className="block text-xs text-gray-500">
                  Bypasses the 3-uploads-per-day rate limit.
                </span>
              </span>
            </label>
            <label className="flex items-start gap-2 text-sm">
              <input
                type="checkbox"
                checked={formAdmin || formReviewer}
                onChange={(e) => setFormReviewer(e.target.checked)}
                disabled={formAdmin}
                className="mt-1"
              />
              <span>
                <span className="font-medium">Reviewer</span>
                <span className="block text-xs text-gray-500">
                  Access to Review Queue and Diagnosis Review tabs.
                </span>
              </span>
            </label>
          </fieldset>

          <div className="flex items-center gap-3 pt-2 border-t border-gray-200">
            <button
              onClick={save}
              disabled={saving || !dirty}
              className="px-4 py-2 text-sm font-medium bg-[var(--color-teal)] text-white rounded hover:bg-[var(--color-teal-dark)] disabled:opacity-50"
            >
              {saving ? 'Saving…' : 'Save'}
            </button>
            {savedAt && (
              <span className="text-xs text-emerald-700">Saved at {savedAt}</span>
            )}
            {dirty && !saving && !savedAt && (
              <span className="text-xs text-gray-500">Unsaved changes.</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
