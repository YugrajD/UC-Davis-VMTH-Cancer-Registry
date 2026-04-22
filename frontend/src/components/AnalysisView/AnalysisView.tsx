import { useMemo, useState } from 'react';
import { ComposableMap, Geographies, Geography, Marker } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import { useCalEnviroScreenData } from '../../hooks/useCalEnviroScreenData';
import type { CountyData, CESIndicator, CalEnviroScreenData } from '../../types';
import { CES_INDICATORS } from '../../types';
import { HUMAN_CANCER_RATES } from '../../data/humanCancerRates';
import {
  MOCK_SUPERFUND_SITES,
  SUPERFUND_BY_COUNTY,
  type SuperfundSite,
} from '../../data/superfundData';
import { MOCK_PESTICIDE_DATA, PESTICIDE_BY_COUNTY } from '../../data/pesticideData';

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

// ---------------------------------------------------------------------------
// Superfund dot overlay — rendered inside any ComposableMap
// ---------------------------------------------------------------------------

function SuperfundOverlay({ sites }: { sites: SuperfundSite[] }) {
  const [hovered, setHovered] = useState<SuperfundSite | null>(null);

  return (
    <>
      {sites.map((site) => (
        <Marker key={site.name} coordinates={site.coordinates}>
          <circle
            r={5}
            fill={site.status === 'active' ? '#EF4444' : site.status === 'proposed' ? '#F97316' : '#22C55E'}
            stroke="#FFFFFF"
            strokeWidth={1}
            opacity={0.85}
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => setHovered(site)}
            onMouseLeave={() => setHovered(null)}
          />
        </Marker>
      ))}
      {hovered && (
        <Marker coordinates={hovered.coordinates}>
          <foreignObject x={8} y={-40} width={200} height={80} style={{ overflow: 'visible' }}>
            <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-2 text-xs pointer-events-none" style={{ width: '180px' }}>
              <p className="font-semibold text-[var(--color-text-primary)]">{hovered.name}</p>
              <p className="text-[var(--color-text-secondary)] mt-0.5">
                {hovered.county} Co. · <span className={hovered.status === 'active' ? 'text-red-500' : hovered.status === 'proposed' ? 'text-orange-500' : 'text-green-600'}>{hovered.status}</span>
              </p>
              <p className="text-[var(--color-text-secondary)] mt-0.5 truncate">{hovered.contaminants.join(', ')}</p>
            </div>
          </foreignObject>
        </Marker>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// VMTH Cancer Incidence choropleth
// ---------------------------------------------------------------------------

function CancerMap({
  countyData,
  countRange,
  showSuperfund,
}: {
  countyData: CountyData[];
  countRange: { min: number; max: number };
  showSuperfund: boolean;
}) {
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
                      const sf = SUPERFUND_BY_COUNTY[name];
                      const sfStr = sf ? ` · ${sf.total} Superfund site${sf.total !== 1 ? 's' : ''}` : '';
                      setTooltip({ county: name, value: `${count.toLocaleString()} cases${sfStr}`, x: event.clientX, y: event.clientY });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  />
                );
              })
            }
          </Geographies>
          {showSuperfund && <SuperfundOverlay sites={MOCK_SUPERFUND_SITES} />}
        </ComposableMap>
      </div>

      <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
        <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">Cases</p>
        <div className="w-28 h-3 rounded" style={{ background: 'linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)' }} />
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-[var(--color-text-secondary)]">{countRange.min}</span>
          <span className="text-[10px] text-[var(--color-text-secondary)]">{countRange.max}</span>
        </div>
        {showSuperfund && (
          <div className="mt-2 pt-2 border-t border-gray-100 space-y-1">
            {[{ color: '#EF4444', label: 'Active' }, { color: '#F97316', label: 'Proposed' }, { color: '#22C55E', label: 'Remediated' }].map(({ color, label }) => (
              <div key={label} className="flex items-center gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full border border-white" style={{ backgroundColor: color }} />
                <span className="text-[10px] text-[var(--color-text-secondary)]">{label} Superfund</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {tooltip && (
        <div className="fixed z-50 pointer-events-none" style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}>
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// CalEnviroScreen choropleth
// ---------------------------------------------------------------------------

function EnviroScreenMap({
  data,
  indicator,
  showSuperfund,
}: {
  data: CalEnviroScreenData[];
  indicator: CESIndicator;
  showSuperfund: boolean;
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
      .range(['#4CAF50', '#FFC107', '#F44336']);
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
                      setTooltip({ county: name, value: val != null ? `${indicatorLabel}: ${val.toFixed(1)}` : 'No data', x: event.clientX, y: event.clientY });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  />
                );
              })
            }
          </Geographies>
          {showSuperfund && <SuperfundOverlay sites={MOCK_SUPERFUND_SITES} />}
        </ComposableMap>
      </div>

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

      {tooltip && (
        <div className="fixed z-50 pointer-events-none" style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}>
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Human Cancer Registry choropleth
// ---------------------------------------------------------------------------

function HumanCancerMap({ showSuperfund }: { showSuperfund: boolean }) {
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
                      setTooltip({ county: name, value: rate != null ? `${rate.toFixed(1)} per 100K${casesStr}` : 'Suppressed', x: event.clientX, y: event.clientY });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  />
                );
              })
            }
          </Geographies>
          {showSuperfund && <SuperfundOverlay sites={MOCK_SUPERFUND_SITES} />}
        </ComposableMap>
      </div>

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

      {tooltip && (
        <div className="fixed z-50 pointer-events-none" style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}>
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pesticide Use choropleth
// ---------------------------------------------------------------------------

function PesticideMap({ showSuperfund }: { showSuperfund: boolean }) {
  const [tooltip, setTooltip] = useState<MapTooltip | null>(null);

  const valueRange = useMemo(() => {
    const vals = MOCK_PESTICIDE_DATA.map(d => d.lbs_per_sq_mile);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, []);

  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([valueRange.min, (valueRange.min + valueRange.max) / 2, valueRange.max])
      .range(['#FEF9C3', '#F97316', '#7F1D1D']); // pale yellow → orange → dark red
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
                const data = PESTICIDE_BY_COUNTY[name];
                const val = data?.lbs_per_sq_mile;
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
                      hover: { fill: '#60A5FA', stroke: '#2563EB', strokeWidth: 1.5, outline: 'none', cursor: 'pointer' },
                      pressed: { fill: '#2563EB', outline: 'none' },
                    }}
                    onMouseEnter={(e) => {
                      const event = e as unknown as React.MouseEvent;
                      setTooltip({
                        county: name,
                        value: val != null
                          ? `${val.toLocaleString()} lbs/sq mi · ${data.top_pesticide_class}`
                          : 'No data',
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
          {showSuperfund && <SuperfundOverlay sites={MOCK_SUPERFUND_SITES} />}
        </ComposableMap>
      </div>

      <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
        <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">lbs / sq mile</p>
        <div className="w-28 h-3 rounded" style={{ background: 'linear-gradient(to right, #FEF9C3, #F97316, #7F1D1D)' }} />
        <div className="flex justify-between mt-1">
          <span className="text-[10px] text-[var(--color-text-secondary)]">{valueRange.min}</span>
          <span className="text-[10px] text-[var(--color-text-secondary)]">{valueRange.max}</span>
        </div>
        <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
          <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
          <span className="text-[10px] text-[var(--color-text-secondary)]">No data</span>
        </div>
      </div>

      {tooltip && (
        <div className="fixed z-50 pointer-events-none" style={{ left: tooltip.x + 12, top: tooltip.y - 12, transform: 'translateY(-100%)' }}>
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[160px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">{tooltip.county}</p>
            <p className="text-xs text-[var(--color-text-secondary)] mt-1">{tooltip.value}</p>
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Correlation Scatter Plot
// ---------------------------------------------------------------------------

type ScatterXVar = 'pesticide' | 'superfund' | 'ces_score';

const X_VAR_OPTIONS: { value: ScatterXVar; label: string; unit: string }[] = [
  { value: 'pesticide', label: 'Pesticide Use', unit: 'lbs/sq mi' },
  { value: 'superfund', label: 'Superfund Sites', unit: 'sites' },
  { value: 'ces_score', label: 'CES Score', unit: 'percentile' },
];

function CorrelationScatterPlot({
  countyData,
  cesData,
  xVar,
}: {
  countyData: CountyData[];
  cesData: CalEnviroScreenData[];
  xVar: ScatterXVar;
}) {
  const [hovered, setHovered] = useState<string | null>(null);

  const points = useMemo(() => {
    const cesMap = new Map(cesData.map(d => [d.county_name, d]));
    return countyData.flatMap(c => {
      let x: number | null = null;
      if (xVar === 'pesticide') {
        x = PESTICIDE_BY_COUNTY[c.county]?.lbs_per_sq_mile ?? null;
      } else if (xVar === 'superfund') {
        x = SUPERFUND_BY_COUNTY[c.county]?.total ?? 0;
      } else if (xVar === 'ces_score') {
        x = cesMap.get(c.county)?.ces_score ?? null;
      }
      if (x === null) return [];
      return [{ county: c.county, x, y: c.count }];
    });
  }, [countyData, cesData, xVar]);

  const margin = { top: 20, right: 20, bottom: 50, left: 60 };
  const width = 520;
  const height = 320;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xScale = useMemo(() => {
    if (points.length === 0) return scaleLinear().domain([0, 1]).range([0, innerW]);
    const xVals = points.map(p => p.x);
    return scaleLinear().domain([0, Math.max(...xVals) * 1.05]).range([0, innerW]).nice();
  }, [points, innerW]);

  const yScale = useMemo(() => {
    if (points.length === 0) return scaleLinear().domain([0, 1]).range([innerH, 0]);
    const yVals = points.map(p => p.y);
    return scaleLinear().domain([0, Math.max(...yVals) * 1.1]).range([innerH, 0]).nice();
  }, [points, innerH]);

  // Simple linear regression for trend line
  const trendLine = useMemo(() => {
    if (points.length < 2) return null;
    const n = points.length;
    const meanX = points.reduce((s, p) => s + p.x, 0) / n;
    const meanY = points.reduce((s, p) => s + p.y, 0) / n;
    const slope = points.reduce((s, p) => s + (p.x - meanX) * (p.y - meanY), 0) /
      points.reduce((s, p) => s + (p.x - meanX) ** 2, 0);
    const intercept = meanY - slope * meanX;
    const xMin = Math.min(...points.map(p => p.x));
    const xMax = Math.max(...points.map(p => p.x));
    return { x1: xMin, y1: slope * xMin + intercept, x2: xMax, y2: slope * xMax + intercept };
  }, [points]);

  const xVarMeta = X_VAR_OPTIONS.find(o => o.value === xVar)!;
  const xTicks = xScale.ticks(5);
  const yTicks = yScale.ticks(5);

  if (points.length === 0) {
    return (
      <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-secondary)]">
        No overlapping county data for this variable.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <svg
        width="100%"
        viewBox={`0 0 ${width} ${height}`}
        style={{ maxWidth: `${width}px`, display: 'block', margin: '0 auto' }}
      >
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Grid lines */}
          {yTicks.map(t => (
            <line key={t} x1={0} x2={innerW} y1={yScale(t)} y2={yScale(t)} stroke="#E5E7EB" strokeWidth={1} />
          ))}
          {xTicks.map(t => (
            <line key={t} x1={xScale(t)} x2={xScale(t)} y1={0} y2={innerH} stroke="#E5E7EB" strokeWidth={1} />
          ))}

          {/* Trend line */}
          {trendLine && (
            <line
              x1={xScale(trendLine.x1)} y1={yScale(trendLine.y1)}
              x2={xScale(trendLine.x2)} y2={yScale(trendLine.y2)}
              stroke="#94A3B8" strokeWidth={1.5} strokeDasharray="4 3"
            />
          )}

          {/* Data points */}
          {points.map(p => (
            <g key={p.county} onMouseEnter={() => setHovered(p.county)} onMouseLeave={() => setHovered(null)}>
              <circle
                cx={xScale(p.x)}
                cy={yScale(p.y)}
                r={hovered === p.county ? 7 : 5}
                fill={hovered === p.county ? '#E87722' : '#1A6B77'}
                opacity={0.8}
                style={{ cursor: 'pointer', transition: 'r 0.1s' }}
              />
              {hovered === p.county && (
                <g>
                  <rect
                    x={xScale(p.x) + 8}
                    y={yScale(p.y) - 30}
                    width={130}
                    height={38}
                    rx={4}
                    fill="white"
                    stroke="#E5E7EB"
                    strokeWidth={1}
                    filter="drop-shadow(0 1px 3px rgba(0,0,0,0.15))"
                  />
                  <text x={xScale(p.x) + 14} y={yScale(p.y) - 14} fontSize={10} fontWeight="600" fill="#1F2937">{p.county}</text>
                  <text x={xScale(p.x) + 14} y={yScale(p.y) + 2} fontSize={9} fill="#6B7280">
                    {xVarMeta.label}: {p.x.toLocaleString()} · Cases: {p.y.toLocaleString()}
                  </text>
                </g>
              )}
            </g>
          ))}

          {/* X axis */}
          <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
          {xTicks.map(t => (
            <g key={t} transform={`translate(${xScale(t)},${innerH})`}>
              <line y2={4} stroke="#9CA3AF" />
              <text y={16} textAnchor="middle" fontSize={9} fill="#6B7280">{t.toLocaleString()}</text>
            </g>
          ))}
          <text x={innerW / 2} y={innerH + 40} textAnchor="middle" fontSize={10} fill="#374151">
            {xVarMeta.label} ({xVarMeta.unit})
          </text>

          {/* Y axis */}
          <line x1={0} x2={0} y1={0} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
          {yTicks.map(t => (
            <g key={t} transform={`translate(0,${yScale(t)})`}>
              <line x2={-4} stroke="#9CA3AF" />
              <text x={-8} textAnchor="end" dominantBaseline="middle" fontSize={9} fill="#6B7280">{t.toLocaleString()}</text>
            </g>
          ))}
          <text
            transform={`translate(${-44},${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
            fontSize={10}
            fill="#374151"
          >
            Cancer Cases
          </text>
        </g>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// AnalysisView — root component
// ---------------------------------------------------------------------------

type MapId = 'vmth' | 'enviro' | 'human' | 'pesticide';
type MapCount = 2 | 3 | 4;

const MAP_OPTIONS: { id: MapId; label: string }[] = [
  { id: 'vmth', label: 'VMTH Cancer Incidence' },
  { id: 'enviro', label: 'CalEnviroScreen 4.0' },
  { id: 'human', label: 'Human Cancer Registry' },
  { id: 'pesticide', label: 'Pesticide Use' },
];

export function AnalysisView({ countyData, countRange }: AnalysisViewProps) {
  const [selectedIndicator, setSelectedIndicator] = useState<CESIndicator>('ces_score');
  const [mapCount, setMapCount] = useState<MapCount>(3);
  const [twoMapSelection, setTwoMapSelection] = useState<[MapId, MapId]>(['vmth', 'enviro']);
  const [threeMapSelection, setThreeMapSelection] = useState<[MapId, MapId, MapId]>(['vmth', 'enviro', 'human']);
  const [showSuperfund, setShowSuperfund] = useState(false);
  const [scatterXVar, setScatterXVar] = useState<ScatterXVar>('pesticide');

  const { data: cesData, loading, error } = useCalEnviroScreenData();

  const visibleMaps: MapId[] = mapCount === 2
    ? twoMapSelection
    : mapCount === 3
      ? threeMapSelection
      : ['vmth', 'enviro', 'human', 'pesticide'];

  const handleSlotChange = (slot: number, value: MapId, count: MapCount) => {
    if (count === 2) {
      setTwoMapSelection(prev => {
        const next: [MapId, MapId] = [...prev] as [MapId, MapId];
        next[slot] = value;
        return next;
      });
    } else {
      setThreeMapSelection(prev => {
        const next: [MapId, MapId, MapId] = [...prev] as [MapId, MapId, MapId];
        next[slot] = value;
        return next;
      });
    }
  };

  const renderMap = (id: MapId) => {
    switch (id) {
      case 'vmth':
        return (
          <div key={id} className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">Cancer Incidence</h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">Case count by county</p>
            </div>
            <CancerMap countyData={countyData} countRange={countRange} showSuperfund={showSuperfund} />
          </div>
        );
      case 'enviro':
        return (
          <div key={id} className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">CalEnviroScreen 4.0</h3>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">Environmental health percentile</p>
              </div>
              <select
                value={selectedIndicator}
                onChange={(e) => setSelectedIndicator(e.target.value as CESIndicator)}
                className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
              >
                {CES_INDICATORS.map((ind) => (
                  <option key={ind.value} value={ind.value}>{ind.label}</option>
                ))}
              </select>
            </div>
            {loading ? (
              <div className="flex flex-col items-center justify-center py-24">
                <div className="w-8 h-8 border-4 border-gray-200 border-t-[var(--color-teal)] rounded-full animate-spin" />
                <p className="mt-4 text-sm text-[var(--color-text-secondary)]">Loading CalEnviroScreen data...</p>
              </div>
            ) : error ? (
              <div className="p-6"><p className="text-sm text-red-600">Error: {error}</p></div>
            ) : (
              <EnviroScreenMap data={cesData} indicator={selectedIndicator} showSuperfund={showSuperfund} />
            )}
          </div>
        );
      case 'human':
        return (
          <div key={id} className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">Human Cancer Registry</h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Age-adjusted rate per 100K (2017–2021) &middot;{' '}
                <a href="https://statecancerprofiles.cancer.gov/" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">Source</a>
              </p>
            </div>
            <HumanCancerMap showSuperfund={showSuperfund} />
          </div>
        );
      case 'pesticide':
        return (
          <div key={id} className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
              <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">Pesticide Use</h3>
              <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                Avg annual lbs active ingredient / sq mile (2015–2019) &middot;{' '}
                <a href="https://trackingcalifornia.org/data-and-tools/pesticide-mapping-tool" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">Source</a>
              </p>
            </div>
            <PesticideMap showSuperfund={showSuperfund} />
          </div>
        );
    }
  };

  const gridCols = mapCount === 2 ? 'lg:grid-cols-2' : mapCount === 3 ? 'lg:grid-cols-3' : 'lg:grid-cols-2 xl:grid-cols-4';

  return (
    <div className="space-y-6">
      {/* Description + controls */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-start gap-4">
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed flex-1">
            Compare veterinary cancer incidence, human cancer incidence, environmental indicators from{' '}
            <a href="https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">CalEnviroScreen 4.0</a>
            , and pesticide use from{' '}
            <a href="https://trackingcalifornia.org/data-and-tools/pesticide-mapping-tool" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">CDPR</a>
            . Toggle Superfund sites to overlay{' '}
            <a href="https://www.epa.gov/superfund/search-superfund-sites-where-you-live" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">EPA NPL sites</a>
            {' '}on any map.
          </p>
          {/* Map count toggle */}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden shrink-0">
            {([2, 3, 4] as MapCount[]).map((n) => (
              <button
                key={n}
                onClick={() => setMapCount(n)}
                className={`px-3 py-1.5 text-xs font-medium border-l border-gray-300 first:border-l-0 transition-colors ${
                  mapCount === n
                    ? 'bg-[var(--color-teal)] text-white'
                    : 'bg-white text-[var(--color-text-secondary)] hover:bg-gray-50'
                }`}
              >
                {n} Maps
              </button>
            ))}
          </div>
        </div>

        {/* Controls row */}
        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-gray-100">
          {/* Superfund toggle */}
          <button
            onClick={() => setShowSuperfund(v => !v)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border font-medium transition-colors ${
              showSuperfund
                ? 'bg-red-50 border-red-300 text-red-700'
                : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${showSuperfund ? 'bg-red-500' : 'bg-gray-400'}`} />
            Superfund Sites
          </button>

          {/* Map slot selectors for 2/3-map modes */}
          {mapCount < 4 && (
            <>
              <span className="text-xs font-medium text-[var(--color-text-secondary)]">Show:</span>
              {Array.from({ length: mapCount }, (_, i) => (
                <select
                  key={i}
                  value={mapCount === 2 ? twoMapSelection[i] : threeMapSelection[i]}
                  onChange={(e) => handleSlotChange(i, e.target.value as MapId, mapCount)}
                  className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
                >
                  {MAP_OPTIONS.map(opt => (
                    <option key={opt.id} value={opt.id}>{opt.label}</option>
                  ))}
                </select>
              ))}
            </>
          )}
        </div>
      </div>

      {/* Maps grid */}
      <div className={`grid grid-cols-1 ${gridCols} gap-6`}>
        {visibleMaps.map(renderMap)}
      </div>

      {/* Correlation scatter plot */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
              Environmental Correlation
            </h3>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
              Cancer cases vs. environmental exposure by county
            </p>
          </div>
          <select
            value={scatterXVar}
            onChange={(e) => setScatterXVar(e.target.value as ScatterXVar)}
            className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
          >
            {X_VAR_OPTIONS.map(o => (
              <option key={o.value} value={o.value}>{o.label} ({o.unit})</option>
            ))}
          </select>
        </div>
        <div className="p-4">
          <CorrelationScatterPlot countyData={countyData} cesData={cesData} xVar={scatterXVar} />
          <p className="text-xs text-[var(--color-text-secondary)] mt-3 text-center">
            Dashed line shows linear trend. Each dot is one county. Hover for details.
          </p>
        </div>
      </div>
    </div>
  );
}
