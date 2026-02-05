import { useState } from 'react';
import { Navigation, Filters, SummaryTable, CountyTable, ChoroplethMap, Footer } from './components';
import { useFilteredData } from './hooks/useFilteredData';
import type { TabType, FilterState } from './types';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [hoveredCounty, setHoveredCounty] = useState<string | null>(null);
  const [selectedCounty, setSelectedCounty] = useState<string | null>(null);
  const [filters, setFilters] = useState<FilterState>({
    rateType: 'incidence',
    sex: 'all',
    cancerType: 'All Types',
    breed: 'All Breeds',
  });

  const { countyData, regionSummary, rateRange, loading, error } = useFilteredData(filters);

  const handleCountyClick = (county: string) => {
    setSelectedCounty(selectedCounty === county ? null : county);
  };

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {/* Intro text */}
        <div className="mb-6 bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
            This dashboard provides a customized interface for querying and visualizing population-based 
            cancer incidence and mortality data for dogs across California. The data covers the 
            UC Davis VMTH catchment area, which includes Northern California and the Central Valley 
            regions. Use the filters on the right to explore data by cancer type, breed, and sex.
        </p>
      </div>

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Tables */}
          <div className="lg:col-span-2 space-y-6">
            <SummaryTable data={regionSummary} />
            <CountyTable 
              data={countyData} 
              rateRange={rateRange}
              onCountyHover={setHoveredCounty}
              selectedCounty={selectedCounty}
            />
          </div>

          {/* Right column - Filters and Map */}
          <div className="space-y-6">
            <Filters filters={filters} onFilterChange={setFilters} />
            <ChoroplethMap 
              data={countyData} 
              rateRange={rateRange}
              hoveredCounty={hoveredCounty}
              onCountyHover={setHoveredCounty}
              onCountyClick={handleCountyClick}
            />
          </div>
        </div>

        {/* Tab-specific content */}
        {activeTab === 'breed-disparities' && (
          <div className="mt-6 bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
              Breed Disparities Analysis
            </h2>
            <p className="text-sm text-[var(--color-text-secondary)] mb-4">
              This section will display breed-specific cancer incidence rates and comparisons. 
              Select a specific breed from the filter to see detailed statistics.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['Golden Retriever', 'Boxer', 'Rottweiler', 'Labrador Retriever'].map(breed => (
                <div key={breed} className="bg-gray-50 rounded-lg p-4 text-center">
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">{breed}</p>
                  <p className="text-2xl font-bold text-[var(--color-teal-dark)] mt-2">
                    {(Math.random() * 50 + 30).toFixed(1)}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">cases per 10,000</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'cancer-types' && (
          <div className="mt-6 bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
              Cancer Types Distribution
            </h2>
            <p className="text-sm text-[var(--color-text-secondary)] mb-4">
              Overview of cancer types by incidence rate across California.
            </p>
            <div className="space-y-3">
              {['Mast Cell Tumor', 'Lymphoma', 'Mammary Carcinoma', 'Hemangiosarcoma', 'Osteosarcoma', 'Soft Tissue Sarcoma'].map((type, i) => {
                const rate = 50 - i * 6 + Math.random() * 5;
                const width = (rate / 55) * 100;
                return (
                  <div key={type} className="flex items-center gap-4">
                    <span className="w-40 text-sm text-[var(--color-text-primary)]">{type}</span>
                    <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                      <div 
                        className="h-full bg-gradient-to-r from-[var(--color-teal)] to-[var(--color-teal-dark)] rounded-full flex items-center justify-end pr-2"
                        style={{ width: `${width}%` }}
                      >
                        <span className="text-xs font-semibold text-white">{rate.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {activeTab === 'regional-comparison' && (
          <div className="mt-6 bg-white rounded-lg border border-gray-200 p-6">
            <h2 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
              Regional Comparison
            </h2>
            <p className="text-sm text-[var(--color-text-secondary)] mb-4">
              Compare cancer incidence rates across California regions.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                { name: 'Bay Area', rate: 42.3, trend: 'up' },
                { name: 'Northern CA', rate: 38.7, trend: 'down' },
                { name: 'Central Valley', rate: 35.2, trend: 'stable' },
                { name: 'Central Coast', rate: 40.1, trend: 'up' },
                { name: 'Southern CA', rate: 44.8, trend: 'up' },
              ].map(region => (
                <div key={region.name} className="bg-gray-50 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-[var(--color-text-primary)]">{region.name}</p>
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      region.trend === 'up' ? 'bg-red-100 text-red-600' :
                      region.trend === 'down' ? 'bg-green-100 text-green-600' :
                      'bg-gray-200 text-gray-600'
                    }`}>
                      {region.trend === 'up' ? '↑' : region.trend === 'down' ? '↓' : '→'}
                    </span>
                  </div>
                  <p className="text-3xl font-bold text-[var(--color-teal-dark)] mt-2">
                    {region.rate}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">per 10,000 dogs</p>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;
