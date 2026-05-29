import { useEffect, useMemo, useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { fetchAgeDetail } from '../../api/client';
import type { AgeDetail } from '../../api/client';
import { AGE_GROUP_OPTIONS } from '../../types';
import type { AgeGroup } from '../../types';

const GEO_URL =
  'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson';

const MAP_PROJECTION_CONFIG = {
  scale: 2400,
  center: [-119.5, 37.5] as [number, number],
};

const AGE_GROUP_DISPLAY_OPTIONS = AGE_GROUP_OPTIONS.filter(o => o.value !== 'all');

export function AgeDisparitiesView() {
  const [selectedAgeGroup, setSelectedAgeGroup] = useState<AgeGroup | ''>('');
  const [loadedAgeGroup, setLoadedAgeGroup] = useState<string>('');
  const [detail, setDetail] = useState<AgeDetail | null>(null);

  const loadingDetail = selectedAgeGroup !== '' && selectedAgeGroup !== loadedAgeGroup;

  const [tooltip, setTooltip] = useState<{
    county: string;
    count: number;
    x: number;
    y: number;
  } | null>(null);

  useEffect(() => {
    if (!selectedAgeGroup) return;
    fetchAgeDetail(selectedAgeGroup)
      .then((data) => {
        setDetail(data);
        setLoadedAgeGroup(selectedAgeGroup);
      })
      .catch(() => {
        setDetail(null);
        setLoadedAgeGroup(selectedAgeGroup);
      });
  }, [selectedAgeGroup]);

  const countyMap = useMemo(() => {
    const m = new Map<string, number>();
    detail?.county_cases.forEach((c) => m.set(c.county_name.toLowerCase(), c.count));
    return m;
  }, [detail]);

  const countRange = useMemo(() => {
    if (!detail || detail.county_cases.length === 0) return { min: 0, max: 1 };
    const counts = detail.county_cases.map((c) => c.count);
    return { min: Math.min(...counts), max: Math.max(...counts) };
  }, [detail]);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([countRange.min, (countRange.min + countRange.max) / 2, countRange.max])
      .range(['#E6F3F5', '#6BB5BF', '#1A6B77']);
  }, [countRange]);

  const maxCancerCount = detail?.cancer_types[0]?.count || 1;

  const selectedOption = AGE_GROUP_DISPLAY_OPTIONS.find(o => o.value === selectedAgeGroup);
  const displayLabel = selectedOption
    ? `${selectedOption.label} (${selectedOption.range})`
    : '';

  return (
    <div className="space-y-6">
      {/* Age Group Selector */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 flex items-center gap-4">
        <label
          htmlFor="age-group-select"
          className="text-sm font-medium text-[var(--color-text-primary)] whitespace-nowrap"
        >
          Select Age Group:
        </label>
        <select
          id="age-group-select"
          value={selectedAgeGroup}
          onChange={(e) => setSelectedAgeGroup(e.target.value as AgeGroup)}
          className="text-sm border border-gray-300 rounded-md px-3 py-2 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent w-64"
        >
          <option value="">— Choose an age group —</option>
          {AGE_GROUP_DISPLAY_OPTIONS.map(opt => (
            <option key={opt.value} value={opt.value}>
              {opt.label} ({opt.range})
            </option>
          ))}
        </select>
      </div>

      {loadingDetail && (
        <div className="flex items-center justify-center h-32">
          <p className="text-sm text-[var(--color-text-secondary)]">Loading age group data…</p>
        </div>
      )}

      {!loadingDetail && detail && (
        <>
          {/* Summary Stats */}
          <div className="bg-white rounded-lg border border-gray-200 p-4">
            <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
              <span className="font-semibold text-[var(--color-text-primary)]">
                {displayLabel}
              </span>{' '}
              has{' '}
              <span className="font-semibold text-[var(--color-teal-dark)]">
                {detail.total_cases.toLocaleString()}
              </span>{' '}
              total case diagnoses across{' '}
              <span className="font-semibold">{detail.county_cases.length}</span> counties.
              {detail.sex_breakdown.length > 0 && (
                <>
                  {' '}
                  Sex distribution:{' '}
                  {detail.sex_breakdown
                    .map((s) => `${s.sex} (${s.count})`)
                    .join(', ')}
                  .
                </>
              )}
            </p>
          </div>

          {/* Two-column layout: bar chart + map */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left: Cancer Type Breakdown */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                  Cancer Type Breakdown
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                  Top cancer types for {displayLabel}
                </p>
              </div>
              <div className="p-6">
                {detail.cancer_types.length === 0 ? (
                  <p className="text-sm text-[var(--color-text-secondary)]">
                    No cancer type data available for this age group.
                  </p>
                ) : (
                  <div className="space-y-4">
                    {detail.cancer_types.slice(0, 15).map((ct) => {
                      const width = Math.max(5, (ct.count / maxCancerCount) * 100);
                      return (
                        <div key={ct.cancer_type} className="flex items-center gap-4">
                          <span className="w-48 text-sm text-[var(--color-text-primary)] truncate">
                            {ct.cancer_type}
                          </span>
                          <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-[var(--color-teal)] to-[var(--color-teal-dark)] rounded-full flex items-center justify-end pr-2"
                              style={{ width: `${width}%` }}
                            >
                              <span className="text-xs font-semibold text-white">
                                {ct.count.toLocaleString()}
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

            {/* Right: County Distribution Map */}
            <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                  County Distribution
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                  Where {displayLabel} cases are located
                </p>
              </div>
              <div className="relative" style={{ minHeight: '400px', backgroundColor: '#f8fafc' }}>
                <ComposableMap
                  projection="geoMercator"
                  projectionConfig={MAP_PROJECTION_CONFIG}
                  width={400}
                  height={400}
                  style={{ width: '100%', height: '100%' }}
                >
                  <Geographies geography={GEO_URL}>
                    {({ geographies }) =>
                      geographies.map((geo) => {
                        const name = (geo.properties.name || '') as string;
                        const count = countyMap.get(name.toLowerCase()) ?? 0;
                        const fill = count > 0 ? colorScale(count) : '#E5E7EB';

                        return (
                          <Geography
                            key={geo.rsmKey}
                            geography={geo}
                            fill={fill}
                            stroke="#FFFFFF"
                            strokeWidth={0.5}
                            style={{
                              default: { outline: 'none' },
                              hover: {
                                fill: '#F5A623',
                                stroke: '#E87722',
                                strokeWidth: 1.5,
                                outline: 'none',
                                cursor: 'pointer',
                              },
                              pressed: { fill: '#E87722', outline: 'none' },
                            }}
                            onMouseEnter={(e) => {
                              const event = e as unknown as React.MouseEvent;
                              setTooltip({
                                county: name,
                                count,
                                x: event.clientX,
                                y: event.clientY,
                              });
                            }}
                            onMouseLeave={() => setTooltip(null)}
                          />
                        );
                      })
                    }
                  </Geographies>
                </ComposableMap>

                {/* Legend */}
                <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
                  <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">
                    Cases
                  </p>
                  <div
                    className="w-28 h-3 rounded"
                    style={{
                      background: 'linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)',
                    }}
                  />
                  <div className="flex justify-between mt-1">
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      {countRange.min}
                    </span>
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      {countRange.max}
                    </span>
                  </div>
                  <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
                    <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
                    <span className="text-[10px] text-[var(--color-text-secondary)]">
                      No data
                    </span>
                  </div>
                </div>

                {/* Tooltip */}
                {tooltip && (
                  <div
                    className="fixed z-50 pointer-events-none"
                    style={{
                      left: tooltip.x + 12,
                      top: tooltip.y - 12,
                      transform: 'translateY(-100%)',
                    }}
                  >
                    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
                      <p className="font-semibold text-sm text-[var(--color-text-primary)]">
                        {tooltip.county}
                      </p>
                      <p className="text-xs text-[var(--color-text-secondary)] mt-1">
                        {tooltip.count > 0
                          ? `${tooltip.count.toLocaleString()} cases`
                          : 'No data'}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </>
      )}

      {!loadingDetail && !detail && !selectedAgeGroup && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                Cancer Type Breakdown
              </h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Top cancer types for selected age group
              </p>
            </div>
            <div className="p-6 flex items-center justify-center h-48">
              <p className="text-sm text-[var(--color-text-secondary)]">Select an age group to view data</p>
            </div>
          </div>
          <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                County Distribution
              </h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Where selected age group cases are located
              </p>
            </div>
            <div className="relative" style={{ minHeight: '400px', backgroundColor: '#f8fafc' }}>
              <ComposableMap
                projection="geoMercator"
                projectionConfig={MAP_PROJECTION_CONFIG}
                width={400}
                height={400}
                style={{ width: '100%', height: '100%' }}
              >
                <Geographies geography={GEO_URL}>
                  {({ geographies }) =>
                    geographies.map((geo) => (
                      <Geography
                        key={geo.rsmKey}
                        geography={geo}
                        fill="#E5E7EB"
                        stroke="#FFFFFF"
                        strokeWidth={0.5}
                        style={{
                          default: { outline: 'none' },
                          hover: { outline: 'none' },
                          pressed: { outline: 'none' },
                        }}
                      />
                    ))
                  }
                </Geographies>
              </ComposableMap>
              <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
                  <span className="text-[10px] text-[var(--color-text-secondary)]">No data</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {!loadingDetail && !detail && selectedAgeGroup && (
        <div className="bg-white rounded-lg border border-gray-200 p-8 text-center">
          <p className="text-sm text-[var(--color-text-secondary)]">
            No data available for {displayLabel}.
          </p>
        </div>
      )}
    </div>
  );
}
