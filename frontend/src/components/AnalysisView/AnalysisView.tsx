import { useMemo, useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { useCalEnviroScreenData } from '../../hooks/useCalEnviroScreenData';
import type { CountyData, CESIndicator, CalEnviroScreenData } from '../../types';
import { CES_INDICATORS } from '../../types';
import { HUMAN_CANCER_RATES } from '../../data/humanCancerRates';

const GEO_URL = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson';

const MAP_PROJECTION_CONFIG = {
  scale: 2400,
  center: [-119.5, 37.5] as [number, number],
};

interface AnalysisViewProps {
  countyData: CountyData[];
  countRange: { min: number; max: number };
}

interface MapTooltip {
  county: string;
  value: string;
  x: number;
  y: number;
}

function CancerMap({ countyData, countRange }: { countyData: CountyData[]; countRange: { min: number; max: number } }) {
  const [tooltip, setTooltip] = useState<MapTooltip | null>(null);

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
                const info = countyDataMap.get(name.toLowerCase());
                const count = info?.count ?? 0;
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
                      hover: { fill: '#F5A623', stroke: '#E87722', strokeWidth: 1.5, outline: 'none', cursor: 'pointer' },
                      pressed: { fill: '#E87722', outline: 'none' },
                    }}
                    onMouseEnter={(e) => {
                      const event = e as unknown as React.MouseEvent;
                      setTooltip({
                        county: name,
                        value: `${count.toLocaleString()} cases`,
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

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}
        >
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function EnviroScreenMap({
  data,
  indicator,
}: {
  data: CalEnviroScreenData[];
  indicator: CESIndicator;
}) {
  const [tooltip, setTooltip] = useState<MapTooltip | null>(null);

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
      .range(['#4CAF50', '#FFC107', '#F44336']); // green → yellow → red
  }, [valueRange]);

  const indicatorLabel = CES_INDICATORS.find(i => i.value === indicator)?.label ?? indicator;

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
                const val = countyValueMap.get(name.toLowerCase());
                const fill = val != null ? colorScale(val) : '#E5E7EB';

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke="#FFFFFF"
                    strokeWidth={0.5}
                    style={{
                      default: { outline: 'none' },
                      hover: { fill: '#F5A623', stroke: '#E87722', strokeWidth: 1.5, outline: 'none', cursor: 'pointer' },
                      pressed: { fill: '#E87722', outline: 'none' },
                    }}
                    onMouseEnter={(e) => {
                      const event = e as unknown as React.MouseEvent;
                      setTooltip({
                        county: name,
                        value: val != null ? `${indicatorLabel}: ${val.toFixed(1)}` : 'No data',
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

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}
        >
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

function HumanCancerMap() {
  const [tooltip, setTooltip] = useState<MapTooltip | null>(null);

  const rateMap = useMemo(() => {
    const map = new Map<string, { rate: number | null; cases: number | null }>();
    HUMAN_CANCER_RATES.forEach(d => map.set(d.county.toLowerCase(), { rate: d.rate, cases: d.cases }));
    return map;
  }, []);

  const rateRange = useMemo(() => {
    const vals = HUMAN_CANCER_RATES.map(d => d.rate).filter((v): v is number => v !== null);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, []);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([rateRange.min, (rateRange.min + rateRange.max) / 2, rateRange.max])
      .range(['#F3E5F5', '#9C27B0', '#4A148C']);
  }, [rateRange]);

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
                const info = rateMap.get(name.toLowerCase());
                const rate = info?.rate;
                const fill = rate != null ? colorScale(rate) : '#E5E7EB';

                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={fill}
                    stroke="#FFFFFF"
                    strokeWidth={0.5}
                    style={{
                      default: { outline: 'none' },
                      hover: { fill: '#F5A623', stroke: '#E87722', strokeWidth: 1.5, outline: 'none', cursor: 'pointer' },
                      pressed: { fill: '#E87722', outline: 'none' },
                    }}
                    onMouseEnter={(e) => {
                      const event = e as unknown as React.MouseEvent;
                      const casesStr = info?.cases != null ? ` (${info.cases.toLocaleString()}/yr)` : '';
                      setTooltip({
                        county: name,
                        value: rate != null ? `${rate.toFixed(1)} per 100K${casesStr}` : 'Suppressed',
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
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
        <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">Rate per 100K</p>
        <div className="w-28 h-3 rounded" style={{ background: 'linear-gradient(to right, #F3E5F5, #9C27B0, #4A148C)' }} />
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-[var(--color-text-secondary)]">{rateRange.min.toFixed(0)}</span>
          <span className="text-[10px] text-[var(--color-text-secondary)]">{rateRange.max.toFixed(0)}</span>
        </div>
        <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
          <span className="text-[10px] text-[var(--color-text-secondary)]">Suppressed</span>
        </div>
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}
        >
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

export function AnalysisView({ countyData, countRange }: AnalysisViewProps) {
  const [selectedIndicator, setSelectedIndicator] = useState<CESIndicator>('ces_score');
  const { data: cesData, loading, error } = useCalEnviroScreenData();

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed">
          Compare veterinary cancer incidence, human cancer incidence from the{' '}
          <a
            href="https://www.californiahealthmaps.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]"
          >
            California Cancer Registry
          </a>
          , and environmental health indicators from{' '}
          <a
            href="https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40"
            target="_blank"
            rel="noopener noreferrer"
            className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]"
          >
            CalEnviroScreen 4.0
          </a>{' '}
          across California counties. Identifying geographic overlap between animal and human cancer
          patterns alongside environmental burden may reveal shared risk factors.
        </p>
      </div>

      {/* Side-by-side maps */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
          <CancerMap countyData={countyData} countRange={countRange} />
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
            <EnviroScreenMap data={cesData} indicator={selectedIndicator} />
          )}
        </div>

        {/* Human Cancer Registry map */}
        <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
              Human Cancer Registry
            </h3>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
              Age-adjusted rate per 100K (2017–2021) &middot;{' '}
              <a
                href="https://statecancerprofiles.cancer.gov/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]"
              >
                Source
              </a>
            </p>
          </div>
          <HumanCancerMap />
        </div>
      </div>
    </div>
  );
}
