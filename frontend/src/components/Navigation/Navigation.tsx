import { useEffect, useState } from 'react';
import type { TabType } from '../../types';
import { TABS } from '../../types';
import { useAuth } from '../../contexts/AuthContext';
import { LoginModal } from '../LoginModal/LoginModal';
import { fetchPendingCount, fetchPendingRoleRequestCount, fetchPendingExportRequestCount } from '../../api/client';

const PENDING_POLL_MS = 30_000;

interface NavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

export function Navigation({ activeTab, onTabChange }: NavigationProps) {
  const { user, isAdmin, isReviewer, signOut, loading, getAccessToken } = useAuth();
  const [showLogin, setShowLogin] = useState(false);
  const [pendingCount, setPendingCount] = useState<number | null>(null);
  const [pendingRoleCount, setPendingRoleCount] = useState<number | null>(null);
  const [pendingExportCount, setPendingExportCount] = useState<number | null>(null);

  // Poll pending diagnosis count for the badge (admins + reviewers). Users
  // without review access never see the tab so we leave stale state alone.
  useEffect(() => {
    if (!(isAdmin || isReviewer)) return;
    let cancelled = false;
    const tick = async () => {
      const token = await getAccessToken();
      if (!token || cancelled) return;
      try {
        const r = await fetchPendingCount(token);
        if (!cancelled) setPendingCount(r.count);
      } catch {
        // Silent — badge is non-critical UI.
      }
    };
    tick();
    const id = window.setInterval(tick, PENDING_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [isAdmin, isReviewer, getAccessToken]);

  // Poll pending role request count for the User Management badge (admin-only).
  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    const tick = async () => {
      const token = await getAccessToken();
      if (!token || cancelled) return;
      try {
        const r = await fetchPendingRoleRequestCount(token);
        if (!cancelled) setPendingRoleCount(r.count);
      } catch {
        // Silent — badge is non-critical UI.
      }
    };
    tick();
    const id = window.setInterval(tick, PENDING_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [isAdmin, getAccessToken]);

  // Poll pending export request count for the User Management badge (admin-only).
  useEffect(() => {
    if (!isAdmin) return;
    let cancelled = false;
    const tick = async () => {
      const token = await getAccessToken();
      if (!token || cancelled) return;
      try {
        const r = await fetchPendingExportRequestCount(token);
        if (!cancelled) setPendingExportCount(r.count);
      } catch {
        // Silent — badge is non-critical UI.
      }
    };
    tick();
    const id = window.setInterval(tick, PENDING_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [isAdmin, getAccessToken]);

  // Review queue and diagnosis review are visible to admins and reviewers.
  // User management is admin-only.
  const visibleTabs = TABS.filter(tab => {
    if (tab.id === 'review-queue' || tab.id === 'diagnosis-review') {
      return isAdmin || isReviewer;
    }
    if (tab.id === 'user-management') return isAdmin;
    return true;
  });

  return (
    <header className="bg-white border-b border-gray-200">
      {/* Top banner */}
      <div className="bg-[var(--color-teal-dark)] text-white py-2 px-6">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <span className="text-sm font-medium tracking-wide">
            UC Davis Veterinary Medicine
          </span>
          <div className="flex items-center gap-4">
            {!loading && (
              user ? (
                <div className="flex items-center gap-3">
                  <span className="text-sm opacity-80">{user.email}</span>
                  {isAdmin && (
                    <span className="text-xs bg-white/20 px-1.5 py-0.5 rounded">Admin</span>
                  )}
                  <button
                    onClick={() => { signOut(); onTabChange('overview'); }}
                    className="text-sm opacity-80 hover:opacity-100 underline transition-opacity"
                  >
                    Sign Out
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setShowLogin(true)}
                  className="text-sm opacity-80 hover:opacity-100 underline transition-opacity"
                >
                  Sign In
                </button>
              )
            )}
            <span className="text-sm opacity-80">
              Veterinary Medical Teaching Hospital
            </span>
          </div>
        </div>
      </div>

      {/* Main header */}
      <div className="py-4 px-6 border-b border-gray-100">
        <div className="max-w-[1400px] mx-auto">
          <div className="flex items-center gap-3">
            <img
              src="/ucdavisvetmed_logo.jpeg"
              alt="UC Davis Veterinary Medicine logo"
              className="h-10 w-10 object-contain"
            />
            <div>
              <h1 className="text-2xl font-semibold text-[var(--color-text-primary)] tracking-tight">
                California Canine Cancer Registry Dashboard
              </h1>
              <p className="text-sm text-[var(--color-text-secondary)] mt-1">
                Cancer incidence data for dogs in California
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation tabs */}
      <nav className="px-6">
        <div className="max-w-[1400px] mx-auto">
          <div className="flex gap-1">
            {visibleTabs.map((tab) => {
              const showDiagnosisBadge =
                (isAdmin || isReviewer) &&
                tab.id === 'diagnosis-review' &&
                pendingCount !== null &&
                pendingCount > 0;
              const userMgmtTotal = (pendingRoleCount ?? 0) + (pendingExportCount ?? 0);
              const showRoleBadge =
                isAdmin &&
                tab.id === 'user-management' &&
                userMgmtTotal > 0;
              const showBadge = showDiagnosisBadge || showRoleBadge;
              const badgeCount = showDiagnosisBadge ? pendingCount : showRoleBadge ? userMgmtTotal : null;
              return (
                <button
                  key={tab.id}
                  onClick={() => onTabChange(tab.id)}
                  className={`
                    px-5 py-3 text-sm font-medium transition-all duration-200
                    border-b-3 -mb-[1px] inline-flex items-center gap-2
                    ${activeTab === tab.id
                      ? 'bg-[var(--color-primary-orange)] text-[var(--color-teal-dark)] border-[var(--color-primary-orange)] rounded-t-md'
                      : 'text-[var(--color-teal)] hover:bg-gray-50 border-transparent hover:border-[var(--color-teal-light)]'
                    }
                  `}
                >
                  {tab.label}
                  {showBadge && badgeCount != null && (
                    <span className="inline-flex items-center justify-center min-w-[20px] h-5 px-1.5 text-xs font-semibold rounded-full bg-amber-500 text-white">
                      {badgeCount}
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {showLogin && <LoginModal onClose={() => setShowLogin(false)} />}
    </header>
  );
}
