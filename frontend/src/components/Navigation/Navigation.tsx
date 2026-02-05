import type { TabType } from '../../types';
import { TABS } from '../../types';

interface NavigationProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
}

export function Navigation({ activeTab, onTabChange }: NavigationProps) {
  return (
    <header className="bg-white border-b border-gray-200">
      {/* Top banner */}
      <div className="bg-[#052049] text-white py-2 px-6">
        <div className="max-w-[1400px] mx-auto flex items-center justify-between">
          <span className="text-sm font-medium tracking-wide">
            UC Davis Veterinary Medicine
          </span>
          <span className="text-sm opacity-80">
            Veterinary Medical Teaching Hospital
          </span>
        </div>
      </div>
      
      {/* Main header */}
      <div className="py-4 px-6 border-b border-gray-100">
        <div className="max-w-[1400px] mx-auto">
          <h1 className="text-2xl font-semibold text-[var(--color-text-primary)] tracking-tight">
            California Canine Cancer Registry Dashboard
          </h1>
          <p className="text-sm text-[var(--color-text-secondary)] mt-1">
            Population-based cancer incidence and mortality data for dogs in California
          </p>
        </div>
      </div>
      
      {/* Navigation tabs */}
      <nav className="px-6">
        <div className="max-w-[1400px] mx-auto">
          <div className="flex gap-1">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`
                  px-5 py-3 text-sm font-medium transition-all duration-200
                  border-b-3 -mb-[1px]
                  ${activeTab === tab.id 
                    ? 'bg-[var(--color-primary-orange)] text-white border-[var(--color-primary-orange)] rounded-t-md' 
                    : 'text-[var(--color-teal)] hover:bg-gray-50 border-transparent hover:border-[var(--color-teal-light)]'
                  }
                `}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>
    </header>
  );
}
