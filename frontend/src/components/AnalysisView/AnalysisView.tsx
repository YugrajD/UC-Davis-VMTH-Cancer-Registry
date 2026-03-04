import { useMemo, useState, useCallback } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { useCalEnviroScreenData } from '../../hooks/useCalEnviroScreenData';
import type { CountyData, CESIndicator, CalEnviroScreenData } from '../../types';
import { CES_INDICATORS } from '../../types';

const GEO_URL = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson';

const MAP_PROJECTION_CONFIG = {
  scale: 2400,
  center: [-119.5, 37.5] as [number, number],
};

interface AnalysisViewProps {
  countyData: CountyData[];
  countRange: { min: number; max: number };
}

function CancerMap({
  countyData,
  countRange,
  hoveredCounty,
  onHoverCounty,
}: {
  countyData: CountyData[];
  countRange: { min: number; max: number };
  hoveredCounty: string | null;
  onHoverCounty: (county: string | null) => void;
}) {
  const countyDataMap = useMemo(() => {
    const map = new Map<string, CountyData>();
    countyData.forEach(c => map.set(c.county.toLowerCase(), c));
    return map;
  }, [countyData]);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([countRange.min, (countRange.min + countRange.max) / 2, countRange.max])
      .range(['#E6F3F5', '#6BB5BF', '#1A6B77']);
  }, [countRange]);

  return (
    <div className="relative">
      <div style={{ minHeight: '400px', backgroundColor: '#f8fafc' }}>
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
                const nameLower = name.toLowerCase();
                const info = countyDataMap.get(nameLower);
                const count = info?.count ?? 0;
                const baseFill = count > 0 ? colorScale(count) : '#E5E7EB';
                const isHovered = hoveredCounty === nameLower;

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={isHovered ? '#F5A623' : baseFill}
                    stroke={isHovered ? '#E87722' : '#FFFFFF'}
                    strokeWidth={isHovered ? 1.5 : 0.5}
                    style={{
                      default: { outline: 'none' },
                      hover: { fill: '#F5A623', stroke: '#E87722', strokeWidth: 1.5, outline: 'none', cursor: 'pointer' },
                      pressed: { fill: '#E87722', outline: 'none' },
                    }}
                    onMouseEnter={() => onHoverCounty(nameLower)}
                    onMouseLeave={() => onHoverCounty(null)}
                  />
                );
              })
            }
          </Geographies>
        </ComposableMap>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
        <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">Cases</p>
        <div className="w-28 h-3 rounded" style={{ background: 'linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)' }} />
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-[var(--color-text-secondary)]">{countRange.min}</span>
          <span className="text-[10px] text-[var(--color-text-secondary)]">{countRange.max}</span>
        </div>
        <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
          <span className="text-[10px] text-[var(--color-text-secondary)]">No data</span>
        </div>
      </div>
    </div>
  );
}

function EnviroScreenMap({
  data,
  indicator,
  hoveredCounty,
  onHoverCounty,
}: {
  data: CalEnviroScreenData[];
  indicator: CESIndicator;
  hoveredCounty: string | null;
  onHoverCounty: (county: string | null) => void;
}) {
  const countyValueMap = useMemo(() => {
    const map = new Map<string, number | null>();
    data.forEach(d => map.set(d.county_name.toLowerCase(), d[indicator]));
    return map;
  }, [data, indicator]);

  const valueRange = useMemo(() => {
    const vals = data.map(d => d[indicator]).filter((v): v is number => v !== null);
    if (vals.length === 0) return { min: 0, max: 100 };
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [data, indicator]);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([valueRange.min, (valueRange.min + valueRange.max) / 2, valueRange.max])
      .range(['#4CAF50', '#FFC107', '#F44336']);
  }, [valueRange]);

  return (
    <div className="relative">
      <div style={{ minHeight: '400px', backgroundColor: '#f8fafc' }}>
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
                const nameLower = name.toLowerCase();
                const val = countyValueMap.get(nameLower);
                const baseFill = val != null ? colorScale(val) : '#E5E7EB';
                const isHovered = hoveredCounty === nameLower;

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={isHovered ? '#F5A623' : baseFill}
                    stroke={isHovered ? '#E87722' : '#FFFFFF'}
                    strokeWidth={isHovered ? 1.5 : 0.5}
                    style={{
                      default: { outline: 'none' },
                      hover: { fill: '#F5A623', stroke: '#E87722', strokeWidth: 1.5, outline: 'none', cursor: 'pointer' },
                      pressed: { fill: '#E87722', outline: 'none' },
                    }}
                    onMouseEnter={() => onHoverCounty(nameLower)}
                    onMouseLeave={() => onHoverCounty(null)}
                  />
                );
              })
            }
          </Geographies>
        </ComposableMap>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
        <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">Percentile (0–100)</p>
        <div className="w-28 h-3 rounded" style={{ background: 'linear-gradient(to right, #4CAF50, #FFC107, #F44336)' }} />
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-[var(--color-text-secondary)]">{valueRange.min.toFixed(0)}</span>
          <span className="text-[10px] text-[var(--color-text-secondary)]">{valueRange.max.toFixed(0)}</span>
        </div>
        <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
          <span className="text-[10px] text-[var(--color-text-secondary)]">No data</span>
        </div>
      </div>
    </div>
  );
}

export function AnalysisView({ countyData, countRange }: AnalysisViewProps) {
  const [selectedIndicator, setSelectedIndicator] = useState<CESIndicator>('ces_score');
  const [hoveredCounty, setHoveredCounty] = useState<string | null>(null);
  const { data: cesData, loading, error } = useCalEnviroScreenData();

  const handleHoverCounty = useCallback((county: string | null) => {
    setHoveredCounty(county);
  }, []);

  // Build the info panel data for the hovered county
  const hoveredInfo = useMemo(() => {
    if (!hoveredCounty) return null;

    const countyCase = countyData.find(c => c.county.toLowerCase() === hoveredCounty);
    const cesEntry = cesData.find(d => d.county_name.toLowerCase() === hoveredCounty);
    const indicatorLabel = CES_INDICATORS.find(i => i.value === selectedIndicator)?.label ?? selectedIndicator;
    const indicatorVal = cesEntry ? cesEntry[selectedIndicator] : null;

    // Title-case the county name
    const displayName = hoveredCounty.replace(/\b\w/g, c => c.toUpperCase());

    return {
      name: displayName,
      cases: countyCase?.count ?? 0,
      indicatorLabel,
      indicatorVal,
    };
  }, [hoveredCounty, countyData, cesData, selectedIndicator]);

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          Compare cancer incidence across California counties with{' '}
          <a
            href="https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]"
          >
            CalEnviroScreen 4.0
          </a>{' '}
          environmental health indicators. CalEnviroScreen ranks communities based on pollution
          exposure and population vulnerability. Higher percentiles indicate greater environmental burden.
          Use the dropdown to explore different indicators. Hover over a county on either map to compare.
        </p>
      </div>

      {/* Hover info bar */}
      <div
        className="bg-white rounded-lg border border-gray-200 px-4 py-3 flex items-center justify-between transition-opacity duration-150"
        style={{ opacity: hoveredInfo ? 1 : 0.4 }}
      >
        {hoveredInfo ? (
          <>
            <span className="font-semibold text-sm text-[var(--color-text-primary)]">
              {hoveredInfo.name} County
            </span>
            <div className="flex items-center gap-6 text-sm">
              <span className="text-[var(--color-text-secondary)]">
                Cancer cases: <span className="font-medium text-[var(--color-text-primary)]">{hoveredInfo.cases.toLocaleString()}</span>
              </span>
              <span className="text-[var(--color-text-secondary)]">
                {hoveredInfo.indicatorLabel}:{' '}
                <span className="font-medium text-[var(--color-text-primary)]">
                  {hoveredInfo.indicatorVal != null ? hoveredInfo.indicatorVal.toFixed(1) : 'N/A'}
                </span>
              </span>
            </div>
          </>
        ) : (
          <span className="text-sm text-[var(--color-text-secondary)]">
            Hover over a county to compare cancer incidence and environmental data
          </span>
        )}
      </div>

      {/* Side-by-side maps */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Cancer incidence map */}
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
              Cancer Incidence
            </h3>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
              Case count by county
            </p>
          </div>
          <CancerMap
            countyData={countyData}
            countRange={countRange}
            hoveredCounty={hoveredCounty}
            onHoverCounty={handleHoverCounty}
          />
        </div>

        {/* Right: CalEnviroScreen map */}
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                CalEnviroScreen 4.0
              </h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Environmental health percentile
              </p>
            </div>
            <select
              value={selectedIndicator}
              onChange={(e) => setSelectedIndicator(e.target.value as CESIndicator)}
              className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
            >
              {CES_INDICATORS.map((ind) => (
                <option key={ind.value} value={ind.value}>
                  {ind.label}
                </option>
              ))}
            </select>
          </div>

          {loading ? (
            <div className="flex flex-col items-center justify-center py-24">
              <div className="w-8 h-8 border-4 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
              <p className="mt-4 text-sm text-[var(--color-text-secondary)]">Loading CalEnviroScreen data...</p>
            </div>
          ) : error ? (
            <div className="p-6">
              <p className="text-sm text-red-600">Error: {error}</p>
            </div>
          ) : (
            <EnviroScreenMap
              data={cesData}
              indicator={selectedIndicator}
              hoveredCounty={hoveredCounty}
              onHoverCounty={handleHoverCounty}
            />
          )}
        </div>
      </div>
    </div>
  );
}
