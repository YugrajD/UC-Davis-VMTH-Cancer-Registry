import { useState } from 'react';
import { Navigation, Filters, SummaryTable, CountyTable, ChoroplethMap, Footer, DataUpload } from './components';
import { useFilteredData } from './hooks/useFilteredData';
import { useCancerTypesData } from './hooks/useCancerTypesData';
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

  const { countyData, regionSummary, countRange, loading, error } = useFilteredData(filters);
  const cancerTypesState = useCancerTypesData(filters);

  const handleCountyClick = (county: string) => {
    setSelectedCounty(selectedCounty === county ? null : county);
  };

  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      <Navigation activeTab={activeTab} onTabChange={setActiveTab} />
      
      <main className="max-w-[1400px] mx-auto px-6 py-6">
        {activeTab === 'data-upload' ? (
          <DataUpload />
        ) : activeTab === 'cancer-types' ? (
          <div className="space-y-6">
            <div className="bg-white rounded-lg border border-gray-200 p-4">
              <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
                This view shows the distribution of cancer types across all ingested PetBERT cases.
                Use the filters above to focus on specific sex or cancer types.
              </p>
            </div>
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-[var(--color-text-primary)] mb-4">
                Cancer Types Distribution
              </h2>
              <p className="text-sm text-[var(--color-text-secondary)] mb-4">
                Overview of cancer types by case count across California.
              </p>

              {cancerTypesState.loading ? (
                <p className="text-sm text-[var(--color-text-secondary)]">Loading cancer type data...</p>
              ) : cancerTypesState.error ? (
                <p className="text-sm text-red-600">Error: {cancerTypesState.error}</p>
              ) : cancerTypesState.data.length === 0 ? (
                <p className="text-sm text-[var(--color-text-secondary)]">
                  No cancer type data found for the selected filters.
                </p>
              ) : (
                <div className="space-y-4">
                  {cancerTypesState.data
                    .slice()
                    .sort((a, b) => b.count - a.count)
                    .slice(0, 10)
                    .map((record) => {
                      const maxCount = cancerTypesState.data[0]?.count || 1;
                      const width = Math.max(5, (record.count / maxCount) * 100);
                      return (
                        <div key={record.cancer_type} className="flex items-center gap-4">
                          <span className="w-48 text-sm text-[var(--color-text-primary)]">
                            {record.cancer_type}
                          </span>
                          <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-[var(--color-teal)] to-[var(--color-teal-dark)] rounded-full flex items-center justify-end pr-2"
                              style={{ width: `${width}%` }}
                            >
                              <span className="text-xs font-semibold text-white">
                                {record.count.toLocaleString()}
                              </span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                </div>
              )}
            </div>
          </div>
        ) : (
        <>
        {/* Intro text */}
        <div className="mb-6 bg-white rounded-lg border border-gray-200 p-4">
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
            This dashboard provides a customized interface for querying and visualizing
            cancer incidence data for dogs across California. The data covers the
            UC Davis VMTH catchment area, which includes Northern California and the Central Valley
            regions. Use the filters on the right to explore data by cancer type, breed, and sex.
        </p>
      </div>

        {/* Error banner */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-3">
            <svg className="w-5 h-5 text-red-500 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}

        {/* Main content grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Tables */}
          <div className="lg:col-span-2 space-y-6">
            {loading ? (
              <div className="bg-white rounded-lg border border-gray-200 p-12 flex flex-col items-center justify-center">
                <div className="w-8 h-8 border-4 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
                <p className="mt-4 text-sm text-[var(--color-text-secondary)]">Loading data...</p>
              </div>
            ) : countyData.length === 0 && !error ? (
              <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
                <p className="text-sm text-[var(--color-text-secondary)]">
                  No data found for the selected filters. Try adjusting your filter criteria.
                </p>
              </div>
            ) : (
              <>
                <SummaryTable data={regionSummary} />
                <CountyTable
                  data={countyData}
                  countRange={countRange}
                  onCountyHover={setHoveredCounty}
                  selectedCounty={selectedCounty}
                />
              </>
            )}
          </div>

          {/* Right column - Filters and Map */}
          <div className="space-y-6" id="filters-panel">
            <Filters filters={filters} onFilterChange={setFilters} />
            <ChoroplethMap
              data={countyData}
              countRange={countRange}
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
              This section will display breed-specific cancer case counts. 
              Select a specific breed from the filter to see detailed statistics.
            </p>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {['Golden Retriever', 'Boxer', 'Rottweiler', 'Labrador Retriever'].map(breed => (
                <div key={breed} className="bg-gray-50 rounded-lg p-4 text-center">
                  <p className="text-sm font-medium text-[var(--color-text-primary)]">{breed}</p>
                  <p className="text-2xl font-bold text-[var(--color-teal-dark)] mt-2">
                    {Math.floor(Math.random() * 80 + 20)}
                  </p>
                  <p className="text-xs text-[var(--color-text-secondary)]">cases</p>
                </div>
              ))}
            </div>
          </div>
        )}

        </>
        )}
      </main>

      <Footer />
    </div>
  );
}

export default App;
