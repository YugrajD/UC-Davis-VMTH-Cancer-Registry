import { useEffect, useMemo, useRef, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer } from '@deck.gl/layers';
import type { MapViewState, PickingInfo } from '@deck.gl/core';
import { scaleLinear } from 'd3-scale';
import { useCalEnviroScreenData } from '../../hooks/useCalEnviroScreenData';
import { useFilteredData } from '../../hooks/useFilteredData';
import { useYearlyTrendsData } from '../../hooks/useYearlyTrendsData';
import { fetchFilterOptions } from '../../api/client';
import { yearRange, countForYear, OTHER_SERIES_NAME } from '../../lib/trends';
import { MapResetButton } from '../MapResetButton/MapResetButton';
import type { CountyData, CESIndicator, CalEnviroScreenData, FilterState } from '../../types';
import { CES_INDICATORS, CANCER_TYPES, BREEDS, SEX_OPTIONS } from '../../types';
import {
  HUMAN_CANCER_RATES,
  HUMAN_CANCER_SITES,
  HUMAN_CANCER_SEX_OPTIONS,
  getHumanCancerRateMap,
  type HumanCancerSite,
  type HumanCancerSex,
} from '../../data/humanCancerRates';
import {
  GEO_URLS,
  INITIAL_VIEW_STATE,
  NO_DATA_COLOR,
  HOVER_COLOR,
  hexToRgba,
  countyFromFeature,
  hoverKeyFromFeature,
  type GeoLevel,
} from '../../lib/mapUtils';

// ---------------------------------------------------------------------------
// Scatter variable definitions (pair-wise plot)
// ---------------------------------------------------------------------------

type ScatterVar =
  | 'cancer_cases'
  | 'human_cancer_rate'
  | CESIndicator;

interface ScatterVarOption {
  value: ScatterVar;
  label: string;
  unit: string;
  group: string;
}

const SCATTER_VAR_OPTIONS: ScatterVarOption[] = [
  // VMTH
  { value: 'cancer_cases', label: 'Cancer Cases', unit: 'cases', group: 'VMTH' },
  // CCR
  { value: 'human_cancer_rate', label: 'Human Cancer Rate', unit: 'per 100K', group: 'CCR' },
  // CES — all 24 indicators
  ...CES_INDICATORS.map(ind => ({
    value: ind.value as ScatterVar,
    label: ind.label,
    unit: 'percentile',
    group: 'CES',
  })),
];

const SCATTER_VAR_GROUPS = ['VMTH', 'CCR', 'CES'] as const;

function isCesVar(v: ScatterVar): v is CESIndicator {
  return CES_INDICATORS.some(ind => ind.value === v);
}

function getVarValue(
  county: string,
  v: ScatterVar,
  countyDataMap: Map<string, CountyData>,
  cesMap: Map<string, CalEnviroScreenData>,
): number | null {
  switch (v) {
    case 'cancer_cases':
      return countyDataMap.get(county.toLowerCase())?.count ?? null;
    case 'human_cancer_rate': {
      const hrMap = getHumanCancerRateMap('All Cancer Sites', 'Both Sexes');
      return hrMap.get(county.toLowerCase())?.rate ?? null;
    }
    default:
      // CES indicator
      if (isCesVar(v)) {
        return cesMap.get(county)?.[v] ?? null;
      }
      return null;
  }
}

function getVarMeta(v: ScatterVar): ScatterVarOption {
  return SCATTER_VAR_OPTIONS.find(o => o.value === v) ?? { value: v, label: String(v), unit: '', group: '' };
}

// ---------------------------------------------------------------------------
// Tooltip header helper (shared by all 4 maps)
// ---------------------------------------------------------------------------

function tooltipHeader(props: Record<string, unknown>, geoLevel: GeoLevel, county: string): string {
  switch (geoLevel) {
    case 'county':
      return `<strong style="font-size:13px">${county}</strong>`;
    case 'tract':
      return `<strong style="font-size:13px">Tract ${props.NAME as string}</strong><br/><span style="color:#6b7280">${county} County</span>`;
    case 'zcta':
      return `<strong style="font-size:13px">ZIP ${props.ZCTA5CE20 as string}</strong><br/><span style="color:#6b7280">${county} County</span>`;
  }
}

// ---------------------------------------------------------------------------
// Filter popover button — small icon button that toggles a dropdown panel
// ---------------------------------------------------------------------------

function MapFilterButton({ children, label }: { children: React.ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen(v => !v)}
        className={`inline-flex items-center gap-1 px-2 py-1.5 rounded-md text-xs font-medium border transition-colors ${open ? 'bg-[var(--color-teal)] text-white border-[var(--color-teal)]' : 'border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
        title={label ?? 'Filters'}
      >
        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
        </svg>
        <span className="hidden sm:inline">Filters</span>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-20 bg-white border border-gray-200 rounded-lg shadow-lg p-3 min-w-[200px] space-y-2">
          {children}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared DeckGL map container
// ---------------------------------------------------------------------------

interface DeckMapProps {
  layers: GeoJsonLayer[];
  getTooltip: (info: PickingInfo) => { html: string; style?: Record<string, string | undefined> } | null;
  title: string;
  subtitle?: React.ReactNode;
  headerRight?: React.ReactNode;
  legend: React.ReactNode;
}

function DeckMap({ layers, getTooltip, title, subtitle, headerRight, legend }: DeckMapProps) {
  // Controlled view state so the "Reset view" button can snap back to
  // INITIAL_VIEW_STATE (CA-wide framing).  Each map manages its own camera.
  const [viewState, setViewState] = useState<MapViewState>(INITIAL_VIEW_STATE);
  const resetView = () => setViewState(INITIAL_VIEW_STATE);
  const isDefaultView =
    viewState.longitude === INITIAL_VIEW_STATE.longitude &&
    viewState.latitude === INITIAL_VIEW_STATE.latitude &&
    viewState.zoom === INITIAL_VIEW_STATE.zoom &&
    viewState.pitch === INITIAL_VIEW_STATE.pitch &&
    viewState.bearing === INITIAL_VIEW_STATE.bearing;

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
            {title}
          </h3>
          {subtitle && (
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">{subtitle}</p>
          )}
        </div>
        {headerRight}
      </div>
      <div className="relative" style={{ height: '400px', backgroundColor: '#f1f5f9' }}>
        <DeckGL
          viewState={viewState}
          onViewStateChange={({ viewState: nextViewState }) =>
            setViewState(nextViewState as MapViewState)
          }
          controller
          layers={layers}
          getTooltip={getTooltip}
          style={{ position: 'absolute', inset: '0', background: '#f1f5f9' }}
        />
        {/* Legend */}
        <div className="absolute bottom-4 left-4 z-10 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm pointer-events-none">
          {legend}
        </div>
        <MapResetButton onClick={resetView} disabled={isDefaultView} />
      </div>
    </div>
  );
}

function GradientLegend({
  label,
  gradient,
  min,
  max,
  noDataLabel = 'No data',
  extra,
}: {
  label: string;
  gradient: string;
  min: string;
  max: string;
  noDataLabel?: string;
  extra?: React.ReactNode;
}) {
  return (
    <>
      <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">{label}</p>
      <div className="w-28 h-3 rounded" style={{ background: gradient }} />
      <div className="flex justify-between mt-1">
        <span className="text-[10px] text-[var(--color-text-secondary)]">{min}</span>
        <span className="text-[10px] text-[var(--color-text-secondary)]">{max}</span>
      </div>
      <div className="mt-2 pt-2 border-t border-gray-100 flex items-center gap-2">
        <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
        <span className="text-[10px] text-[var(--color-text-secondary)]">{noDataLabel}</span>
      </div>
      {extra}
    </>
  );
}

// ---------------------------------------------------------------------------
// Geo level segmented control
// ---------------------------------------------------------------------------

const GEO_LEVEL_OPTIONS: { value: GeoLevel; label: string }[] = [
  { value: 'county', label: 'County' },
  { value: 'tract', label: 'Tract' },
  { value: 'zcta', label: 'ZCTA' },
];

// ---------------------------------------------------------------------------
// VMTH Cancer Incidence map
// ---------------------------------------------------------------------------

function CancerMap({
  geoLevel,
}: {
  geoLevel: GeoLevel;
}) {
  const [filters, setFilters] = useState<FilterState>({
    rateType: 'incidence',
    sex: 'all',
    cancerType: 'All Types',
    breed: 'All Breeds',
  });
  const [yearOptions, setYearOptions] = useState<number[]>([]);

  useEffect(() => {
    fetchFilterOptions()
      .then(opts => {
        const [min, max] = opts.year_range;
        const years: number[] = [];
        for (let y = min; y <= max; y++) years.push(y);
        setYearOptions(years);
      })
      .catch(() => {});
  }, []);

  const handleYearChange = (key: 'yearStart' | 'yearEnd', value: string) => {
    setFilters(f => ({ ...f, [key]: value ? Number(value) : undefined }));
  };

  const { countyData, countRange } = useFilteredData(filters);

  const countyDataMap = useMemo(() => {
    const m = new Map<string, CountyData>();
    countyData.forEach(c => m.set(c.county.toLowerCase(), c));
    return m;
  }, [countyData]);

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([0, countRange.max / 2, countRange.max])
        .range(['#E6F3F5', '#6BB5BF', '#1A6B77']),
    [countRange],
  );

  const [hovered, setHovered] = useState<string | null>(null);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'cancer-counties',
        data: GEO_URLS[geoLevel],
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          if (key && key === hovered) return HOVER_COLOR;
          const county = countyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          const info = countyDataMap.get(county.toLowerCase());
          const count = info?.count ?? 0;
          return count > 0 ? hexToRgba(colorScale(count)) : NO_DATA_COLOR;
        },
        getLineColor: geoLevel !== 'county' ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: geoLevel !== 'county' ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, geoLevel) : null),
        updateTriggers: { getFillColor: [countyDataMap, colorScale, hovered, geoLevel], data: [geoLevel] },
      }),
    [countyDataMap, colorScale, hovered, geoLevel],
  );

  const layers = useMemo(() => [geoLayer], [geoLayer]);

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'cancer-counties') {
      const props = info.object.properties as Record<string, unknown>;
      const county = countyFromFeature(props, geoLevel);
      const count = countyDataMap.get(county.toLowerCase())?.count ?? 0;
      const header = tooltipHeader(props, geoLevel, county);
      return {
        html: `${header}<br/>${count.toLocaleString()} cases`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    return null;
  };

  const subtitle = geoLevel === 'county'
    ? 'Case count by county'
    : geoLevel === 'tract'
      ? 'Case count by county · census tract boundaries'
      : 'Case count by county · ZCTA boundaries';

  return (
    <DeckMap
      layers={layers}
      getTooltip={getTooltip}
      title="Cancer Incidence"
      subtitle={subtitle}
      headerRight={
        <div className="flex items-center gap-1.5 flex-wrap justify-end">
          {filters.cancerType !== 'All Types' && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200 max-w-[120px] truncate">
              {filters.cancerType}
            </span>
          )}
          {filters.breed !== 'All Breeds' && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200 max-w-[120px] truncate">
              {filters.breed}
            </span>
          )}
          {filters.sex !== 'all' && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200">
              {SEX_OPTIONS.find(s => s.value === filters.sex)?.label}
            </span>
          )}
          {(filters.yearStart || filters.yearEnd) && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200">
              {filters.yearStart && filters.yearEnd ? `${filters.yearStart}–${filters.yearEnd}` :
               filters.yearStart ? `≥${filters.yearStart}` : `≤${filters.yearEnd}`}
            </span>
          )}
          <MapFilterButton>
            <label className="block">
              <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Cancer Type</span>
              <select value={filters.cancerType} onChange={e => setFilters(f => ({ ...f, cancerType: e.target.value }))} className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]">
                {CANCER_TYPES.map(ct => <option key={ct} value={ct}>{ct}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Breed</span>
              <select value={filters.breed} onChange={e => setFilters(f => ({ ...f, breed: e.target.value }))} className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]">
                {BREEDS.map(b => <option key={b} value={b}>{b}</option>)}
              </select>
            </label>
            <label className="block">
              <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Sex</span>
              <select value={filters.sex} onChange={e => setFilters(f => ({ ...f, sex: e.target.value as FilterState['sex'] }))} className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]">
                {SEX_OPTIONS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
              </select>
            </label>
            {yearOptions.length > 0 && (
              <label className="block">
                <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Year Start</span>
                <select value={filters.yearStart ?? ''} onChange={e => handleYearChange('yearStart', e.target.value)} className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]">
                  <option value="">All Years</option>
                  {yearOptions.filter(y => !filters.yearEnd || y <= filters.yearEnd).map(y => <option key={y} value={y}>{y}</option>)}
                </select>
              </label>
            )}
            {yearOptions.length > 0 && (
              <label className="block">
                <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Year End</span>
                <select value={filters.yearEnd ?? ''} onChange={e => handleYearChange('yearEnd', e.target.value)} className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]">
                  <option value="">All Years</option>
                  {yearOptions.filter(y => !filters.yearStart || y >= filters.yearStart).map(y => <option key={y} value={y}>{y}</option>)}
                </select>
              </label>
            )}
          </MapFilterButton>
        </div>
      }
      legend={
        <GradientLegend
          label="Cases"
          gradient="linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)"
          min={String(countRange.min)}
          max={String(countRange.max)}
        />
      }
    />
  );
}

// ---------------------------------------------------------------------------
// CalEnviroScreen map
// ---------------------------------------------------------------------------

function EnviroScreenMap({
  data,
  indicator,
  geoLevel,
  onIndicatorChange,
}: {
  data: CalEnviroScreenData[];
  indicator: CESIndicator;
  geoLevel: GeoLevel;
  onIndicatorChange: (v: CESIndicator) => void;
}) {
  const countyValueMap = useMemo(() => {
    const m = new Map<string, number | null>();
    data.forEach(d => m.set(d.county_name.toLowerCase(), d[indicator]));
    return m;
  }, [data, indicator]);

  const valueRange = useMemo(() => {
    const vals = data.map(d => d[indicator]).filter((v): v is number => v !== null);
    if (vals.length === 0) return { min: 0, max: 100 };
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [data, indicator]);

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([valueRange.min, (valueRange.min + valueRange.max) / 2, valueRange.max])
        .range(['#4CAF50', '#FFC107', '#F44336']),
    [valueRange],
  );

  const [hovered, setHovered] = useState<string | null>(null);
  const indicatorLabel = CES_INDICATORS.find(i => i.value === indicator)?.label ?? indicator;

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'enviro-counties',
        data: GEO_URLS[geoLevel],
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          if (key && key === hovered) return HOVER_COLOR;
          const county = countyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          const val = countyValueMap.get(county.toLowerCase());
          return val != null ? hexToRgba(colorScale(val)) : NO_DATA_COLOR;
        },
        getLineColor: geoLevel !== 'county' ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: geoLevel !== 'county' ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, geoLevel) : null),
        updateTriggers: { getFillColor: [countyValueMap, colorScale, hovered, geoLevel], data: [geoLevel] },
      }),
    [countyValueMap, colorScale, hovered, geoLevel],
  );

  const layers = useMemo(() => [geoLayer], [geoLayer]);

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'enviro-counties') {
      const props = info.object.properties as Record<string, unknown>;
      const county = countyFromFeature(props, geoLevel);
      const val = countyValueMap.get(county.toLowerCase());
      const header = tooltipHeader(props, geoLevel, county);
      return {
        html: `${header}<br/>${val != null ? `${indicatorLabel}: <strong>${val.toFixed(1)}</strong>` : 'No data'}`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    return null;
  };

  return (
    <DeckMap
      layers={layers}
      getTooltip={getTooltip}
      title="CalEnviroScreen 4.0"
      subtitle="Environmental health percentile"
      headerRight={
        <div className="flex items-center gap-1.5">
          <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200 max-w-[140px] truncate">
            {indicatorLabel}
          </span>
          <MapFilterButton>
            <label className="block">
              <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Indicator</span>
              <select
                value={indicator}
                onChange={(e) => onIndicatorChange(e.target.value as CESIndicator)}
                className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]"
              >
                {CES_INDICATORS.map((ind) => (
                  <option key={ind.value} value={ind.value}>{ind.label}</option>
                ))}
              </select>
            </label>
          </MapFilterButton>
        </div>
      }
      legend={
        <GradientLegend
          label="Percentile (0–100)"
          gradient="linear-gradient(to right, #4CAF50, #FFC107, #F44336)"
          min={valueRange.min.toFixed(0)}
          max={valueRange.max.toFixed(0)}
        />
      }
    />
  );
}

// ---------------------------------------------------------------------------
// Human Cancer Registry map
// ---------------------------------------------------------------------------

function HumanCancerMap({ geoLevel }: { geoLevel: GeoLevel }) {
  const [selectedSite, setSelectedSite] = useState<HumanCancerSite>('All Cancer Sites');
  const [selectedSex, setSelectedSex] = useState<HumanCancerSex>('Both Sexes');

  // Filter out sex options that are invalid for the selected cancer site
  const availableSexOptions = useMemo(() => {
    if (selectedSite === 'Prostate') return HUMAN_CANCER_SEX_OPTIONS.filter(o => o.value === 'Male');
    if (selectedSite === 'Breast (Female)' || selectedSite === 'Cervix' || selectedSite === 'Ovary' || selectedSite === 'Uterus (Corpus & Uterus, NOS)')
      return HUMAN_CANCER_SEX_OPTIONS.filter(o => o.value === 'Female');
    return HUMAN_CANCER_SEX_OPTIONS;
  }, [selectedSite]);

  // Auto-correct sex when site changes to a sex-specific cancer
  const effectiveSex = useMemo(() => {
    if (availableSexOptions.some(o => o.value === selectedSex)) return selectedSex;
    return availableSexOptions[0].value;
  }, [availableSexOptions, selectedSex]);

  const rateMap = useMemo(
    () => getHumanCancerRateMap(selectedSite, effectiveSex),
    [selectedSite, effectiveSex],
  );

  const rateRange = useMemo(() => {
    const vals = Array.from(rateMap.values()).map(d => d.rate).filter((v): v is number => v !== null);
    if (vals.length === 0) return { min: 0, max: 500 };
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, [rateMap]);

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([rateRange.min, (rateRange.min + rateRange.max) / 2, rateRange.max])
        .range(['#F3E5F5', '#9C27B0', '#4A148C']),
    [rateRange],
  );

  const [hovered, setHovered] = useState<string | null>(null);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'human-counties',
        data: GEO_URLS[geoLevel],
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          if (key && key === hovered) return HOVER_COLOR;
          const county = countyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          const info = rateMap.get(county.toLowerCase());
          const rate = info?.rate;
          return rate != null ? hexToRgba(colorScale(rate)) : NO_DATA_COLOR;
        },
        getLineColor: geoLevel !== 'county' ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: geoLevel !== 'county' ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, geoLevel) : null),
        updateTriggers: { getFillColor: [rateMap, colorScale, hovered, geoLevel], data: [geoLevel] },
      }),
    [rateMap, colorScale, hovered, geoLevel],
  );

  const layers = useMemo(() => [geoLayer], [geoLayer]);

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'human-counties') {
      const props = info.object.properties as Record<string, unknown>;
      const county = countyFromFeature(props, geoLevel);
      const info2 = rateMap.get(county.toLowerCase());
      const rate = info2?.rate;
      const casesStr = info2?.cases != null ? ` (${info2.cases.toLocaleString()}/yr)` : '';
      const header = tooltipHeader(props, geoLevel, county);
      return {
        html: `${header}<br/>${rate != null ? `${rate.toFixed(1)} per 100K${casesStr}` : 'Suppressed'}`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    return null;
  };

  return (
    <DeckMap
      layers={layers}
      getTooltip={getTooltip}
      title="Human Cancer Registry"
      subtitle={
        <>
          Age-adjusted rate per 100K (2017–2021) &middot;{' '}
          <a href="https://statecancerprofiles.cancer.gov/" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">Source</a>
        </>
      }
      headerRight={
        <div className="flex items-center gap-1.5 flex-wrap justify-end">
          {selectedSite !== 'All Cancer Sites' && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200 max-w-[120px] truncate">
              {selectedSite}
            </span>
          )}
          {effectiveSex !== 'Both Sexes' && (
            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-medium bg-teal-50 text-teal-700 border border-teal-200">
              {effectiveSex}
            </span>
          )}
          <MapFilterButton>
            <label className="block">
              <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Cancer Site</span>
              <select
                value={selectedSite}
                onChange={e => setSelectedSite(e.target.value as HumanCancerSite)}
                className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]"
              >
                {HUMAN_CANCER_SITES.map(s => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="text-[10px] font-medium text-[var(--color-text-secondary)] uppercase tracking-wider">Sex</span>
              <select
                value={effectiveSex}
                onChange={e => setSelectedSex(e.target.value as HumanCancerSex)}
                className="mt-0.5 w-full text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)]"
              >
                {availableSexOptions.map(s => (
                  <option key={s.value} value={s.value}>{s.label}</option>
                ))}
              </select>
            </label>
          </MapFilterButton>
        </div>
      }
      legend={
        <GradientLegend
          label="Rate per 100K"
          gradient="linear-gradient(to right, #F3E5F5, #9C27B0, #4A148C)"
          min={rateRange.min.toFixed(0)}
          max={rateRange.max.toFixed(0)}
          noDataLabel="Suppressed"
        />
      }
    />
  );
}

// ---------------------------------------------------------------------------
// Correlation Scatter Plot (pair-wise)
// ---------------------------------------------------------------------------

function CorrelationScatterPlot({
  countyData,
  cesData,
  xVar,
  yVar,
}: {
  countyData: CountyData[];
  cesData: CalEnviroScreenData[];
  xVar: ScatterVar;
  yVar: ScatterVar;
}) {
  const [hovered, setHovered] = useState<string | null>(null);

  const countyDataMap = useMemo(() => {
    const m = new Map<string, CountyData>();
    countyData.forEach(c => m.set(c.county.toLowerCase(), c));
    return m;
  }, [countyData]);

  const cesMap = useMemo(
    () => new Map(cesData.map(d => [d.county_name, d])),
    [cesData],
  );

  // Build a set of all county names across data sources
  const allCounties = useMemo(() => {
    const names = new Set<string>();
    countyData.forEach(c => names.add(c.county));
    cesData.forEach(d => names.add(d.county_name));
    HUMAN_CANCER_RATES.forEach(d => names.add(d.county));
    return Array.from(names);
  }, [countyData, cesData]);

  const points = useMemo(() => {
    return allCounties.flatMap(county => {
      const x = getVarValue(county, xVar, countyDataMap, cesMap);
      const y = getVarValue(county, yVar, countyDataMap, cesMap);
      if (x === null || y === null) return [];
      return [{ county, x, y }];
    });
  }, [allCounties, xVar, yVar, countyDataMap, cesMap]);

  const margin = { top: 20, right: 20, bottom: 50, left: 60 };
  const width = 520;
  const height = 320;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const xScale = useMemo(() => {
    if (points.length === 0) return scaleLinear().domain([0, 1]).range([0, innerW]);
    return scaleLinear().domain([0, Math.max(...points.map(p => p.x)) * 1.05]).range([0, innerW]).nice();
  }, [points, innerW]);

  const yScale = useMemo(() => {
    if (points.length === 0) return scaleLinear().domain([0, 1]).range([innerH, 0]);
    return scaleLinear().domain([0, Math.max(...points.map(p => p.y)) * 1.1]).range([innerH, 0]).nice();
  }, [points, innerH]);

  const trendLine = useMemo(() => {
    if (points.length < 2) return null;
    const n = points.length;
    const meanX = points.reduce((s, p) => s + p.x, 0) / n;
    const meanY = points.reduce((s, p) => s + p.y, 0) / n;
    const denom = points.reduce((s, p) => s + (p.x - meanX) ** 2, 0);
    if (denom === 0) return null;
    const slope =
      points.reduce((s, p) => s + (p.x - meanX) * (p.y - meanY), 0) / denom;
    const intercept = meanY - slope * meanX;
    const xMin = Math.min(...points.map(p => p.x));
    const xMax = Math.max(...points.map(p => p.x));
    return { x1: xMin, y1: slope * xMin + intercept, x2: xMax, y2: slope * xMax + intercept };
  }, [points]);

  const xMeta = getVarMeta(xVar);
  const yMeta = getVarMeta(yVar);
  const xTicks = xScale.ticks(5);
  const yTicks = yScale.ticks(5);

  if (points.length === 0) {
    return <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-secondary)]">No overlapping county data for these variables.</div>;
  }

  return (
    <div className="overflow-x-auto">
      <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ maxWidth: `${width}px`, display: 'block', margin: '0 auto' }}>
        <g transform={`translate(${margin.left},${margin.top})`}>
          {yTicks.map(t => <line key={t} x1={0} x2={innerW} y1={yScale(t)} y2={yScale(t)} stroke="#E5E7EB" strokeWidth={1} />)}
          {xTicks.map(t => <line key={t} x1={xScale(t)} x2={xScale(t)} y1={0} y2={innerH} stroke="#E5E7EB" strokeWidth={1} />)}
          {trendLine && (
            <line x1={xScale(trendLine.x1)} y1={yScale(trendLine.y1)} x2={xScale(trendLine.x2)} y2={yScale(trendLine.y2)} stroke="#94A3B8" strokeWidth={1.5} strokeDasharray="4 3" />
          )}
          {points.map(p => (
            <g key={p.county} onMouseEnter={() => setHovered(p.county)} onMouseLeave={() => setHovered(null)}>
              <circle cx={xScale(p.x)} cy={yScale(p.y)} r={hovered === p.county ? 7 : 5} fill={hovered === p.county ? '#E87722' : '#1A6B77'} opacity={0.8} style={{ cursor: 'pointer', transition: 'r 0.1s' }} />
              {hovered === p.county && (
                <g>
                  <rect x={xScale(p.x) + 8} y={yScale(p.y) - 30} width={180} height={38} rx={4} fill="white" stroke="#E5E7EB" strokeWidth={1} filter="drop-shadow(0 1px 3px rgba(0,0,0,0.15))" />
                  <text x={xScale(p.x) + 14} y={yScale(p.y) - 14} fontSize={10} fontWeight="600" fill="#1F2937">{p.county}</text>
                  <text x={xScale(p.x) + 14} y={yScale(p.y) + 2} fontSize={9} fill="#6B7280">X: {p.x.toLocaleString()} · Y: {p.y.toLocaleString()}</text>
                </g>
              )}
            </g>
          ))}
          <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
          {xTicks.map(t => (
            <g key={t} transform={`translate(${xScale(t)},${innerH})`}>
              <line y2={4} stroke="#9CA3AF" />
              <text y={16} textAnchor="middle" fontSize={9} fill="#6B7280">{t.toLocaleString()}</text>
            </g>
          ))}
          <text x={innerW / 2} y={innerH + 40} textAnchor="middle" fontSize={10} fill="#374151">{xMeta.label} ({xMeta.unit})</text>
          <line x1={0} x2={0} y1={0} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
          {yTicks.map(t => (
            <g key={t} transform={`translate(0,${yScale(t)})`}>
              <line x2={-4} stroke="#9CA3AF" />
              <text x={-8} textAnchor="end" dominantBaseline="middle" fontSize={9} fill="#6B7280">{t.toLocaleString()}</text>
            </g>
          ))}
          <text transform={`translate(${-44},${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="#374151">{yMeta.label} ({yMeta.unit})</text>
        </g>
      </svg>
    </div>
  );
}

const TREND_COLORS = ['#1A6B77', '#E87722', '#9C27B0', '#EF4444', '#2563EB', '#059669', '#D97706', '#6366F1'];

// ---------------------------------------------------------------------------
// Yearly cancer trend chart (US-11 / US-NTH-2)
//
// One line per top-5 cancer type plus an aggregated "Other" line; a
// dropdown lets the user toggle individual lines on/off via the legend.
// Hand-built SVG to match the visual style of the surrounding charts.
// ---------------------------------------------------------------------------

function CancerTrendChart() {
  const { series, loading, error } = useYearlyTrendsData();

  const allNames = useMemo(() => series.map((s) => s.name), [series]);
  // null = "use the default (show everything)". When the user toggles a line
  // we record their explicit choice as an array. This avoids syncing default
  // state from data via an effect or a ref-during-render.
  const [userSelection, setUserSelection] = useState<string[] | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const selectedNames = userSelection ?? allNames;

  const margin = { top: 20, right: 140, bottom: 40, left: 60 };
  const width = 600;
  const height = 300;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  const visible = useMemo(
    () => series.filter((s) => selectedNames.includes(s.name)),
    [series, selectedNames],
  );

  const years = useMemo(() => yearRange(series), [series]);

  const yMax = useMemo(() => {
    let max = 0;
    for (const s of visible) {
      for (const p of s.data) {
        if (p.count > max) max = p.count;
      }
    }
    return max || 1;
  }, [visible]);

  const xScale = useMemo(() => {
    if (years.length === 0) {
      return scaleLinear().domain([2020, 2024]).range([0, innerW]);
    }
    if (years.length === 1) {
      // Single-year corner case: pad domain by ±1 so the dot has somewhere to land.
      return scaleLinear()
        .domain([years[0] - 1, years[0] + 1])
        .range([0, innerW]);
    }
    return scaleLinear()
      .domain([years[0], years[years.length - 1]])
      .range([0, innerW]);
  }, [years, innerW]);

  const yScale = useMemo(
    () => scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]).nice(),
    [innerH, yMax],
  );

  const yTicks = yScale.ticks(5);

  // X-axis labels: thin out so 4-digit year labels don't collide.  d3-scale's
  // .ticks() returns nicely-rounded values inside the domain (e.g. every 5 yrs
  // for a 30-year range) — exactly what we want for the visible labels.
  // Data points and the line are still drawn at every actual year in `years`.
  const xTicks = useMemo(() => {
    if (years.length === 0) return [];
    if (years.length === 1) return years;
    return xScale.ticks(Math.min(8, years.length));
  }, [xScale, years]);

  const toggleName = (name: string) => {
    const current = userSelection ?? allNames;
    setUserSelection(
      current.includes(name) ? current.filter((n) => n !== name) : [...current, name],
    );
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
            Cancer Cases by Year
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
            Top 5 cancer types plus "Other" · annual case counts
          </p>
        </div>
        <div className="relative">
          <button
            onClick={() => setDropdownOpen((v) => !v)}
            disabled={loading || allNames.length === 0}
            className="text-xs border border-gray-300 rounded-md px-3 py-1.5 bg-white text-[var(--color-text-primary)] hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Cancer types ({selectedNames.length})
            <svg className="inline-block w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          {dropdownOpen && (
            <div className="absolute right-0 top-full mt-1 z-20 bg-white border border-gray-200 rounded-lg shadow-lg py-1 w-56 max-h-72 overflow-y-auto">
              {allNames.map((name) => (
                <label
                  key={name}
                  className="flex items-center gap-2 px-3 py-1.5 hover:bg-gray-50 cursor-pointer text-xs"
                >
                  <input
                    type="checkbox"
                    checked={selectedNames.includes(name)}
                    onChange={() => toggleName(name)}
                    className="rounded border-gray-300 text-[var(--color-teal)] focus:ring-[var(--color-teal)]"
                  />
                  <span
                    className={`text-[var(--color-text-primary)] ${
                      name === OTHER_SERIES_NAME ? 'italic' : ''
                    }`}
                  >
                    {name}
                  </span>
                </label>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="p-4">
        {loading ? (
          <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-secondary)]">
            Loading trend data…
          </div>
        ) : error ? (
          <div className="flex items-center justify-center h-40 text-sm text-red-600">
            {error}
          </div>
        ) : series.length === 0 ? (
          <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-secondary)]">
            No yearly trend data available.
          </div>
        ) : visible.length === 0 ? (
          <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-secondary)]">
            Select cancer types to view trends.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <svg
              width="100%"
              viewBox={`0 0 ${width} ${height}`}
              style={{ maxWidth: `${width}px`, display: 'block', margin: '0 auto' }}
            >
              <g transform={`translate(${margin.left},${margin.top})`}>
                {/* Grid lines */}
                {yTicks.map((t) => (
                  <line
                    key={t}
                    x1={0}
                    x2={innerW}
                    y1={yScale(t)}
                    y2={yScale(t)}
                    stroke="#E5E7EB"
                    strokeWidth={1}
                  />
                ))}
                {xTicks.map((yr) => (
                  <line
                    key={yr}
                    x1={xScale(yr)}
                    x2={xScale(yr)}
                    y1={0}
                    y2={innerH}
                    stroke="#E5E7EB"
                    strokeWidth={1}
                  />
                ))}

                {/* Lines */}
                {visible.map((s) => {
                  const colorIndex = allNames.indexOf(s.name);
                  const color = TREND_COLORS[colorIndex % TREND_COLORS.length];
                  const isHov = hovered === s.name;
                  const pathD = years
                    .map((yr, j) => {
                      const x = xScale(yr);
                      const y = yScale(countForYear(s, yr));
                      return `${j === 0 ? 'M' : 'L'}${x},${y}`;
                    })
                    .join(' ');
                  const lastYear = years[years.length - 1];
                  const lastCount = countForYear(s, lastYear);
                  return (
                    <g
                      key={s.name}
                      onMouseEnter={() => setHovered(s.name)}
                      onMouseLeave={() => setHovered(null)}
                      style={{ cursor: 'pointer' }}
                    >
                      <path
                        d={pathD}
                        fill="none"
                        stroke={color}
                        strokeWidth={isHov ? 3 : 1.5}
                        strokeDasharray={s.name === OTHER_SERIES_NAME ? '4 3' : undefined}
                        opacity={hovered && !isHov ? 0.3 : 1}
                      />
                      {years.map((yr) => (
                        <circle
                          key={yr}
                          cx={xScale(yr)}
                          cy={yScale(countForYear(s, yr))}
                          r={isHov ? 5 : 3}
                          fill={color}
                          opacity={hovered && !isHov ? 0.3 : 1}
                        />
                      ))}
                      {isHov && (
                        <g>
                          <rect
                            x={xScale(lastYear) + 8}
                            y={yScale(lastCount) - 18}
                            width={140}
                            height={24}
                            rx={4}
                            fill="white"
                            stroke="#E5E7EB"
                            strokeWidth={1}
                            filter="drop-shadow(0 1px 3px rgba(0,0,0,0.15))"
                          />
                          <text
                            x={xScale(lastYear) + 14}
                            y={yScale(lastCount) - 2}
                            fontSize={10}
                            fontWeight="600"
                            fill="#1F2937"
                          >
                            {s.name}: {lastCount}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}

                {/* X axis — labels only at thinned ticks; the line draws
                    through every year, but labels every year would collide. */}
                <line x1={0} x2={innerW} y1={innerH} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
                {xTicks.map((yr) => (
                  <g key={yr} transform={`translate(${xScale(yr)},${innerH})`}>
                    <line y2={4} stroke="#9CA3AF" />
                    <text y={18} textAnchor="middle" fontSize={10} fill="#6B7280">
                      {Math.round(yr)}
                    </text>
                  </g>
                ))}
                <text x={innerW / 2} y={innerH + 34} textAnchor="middle" fontSize={10} fill="#374151">
                  Year
                </text>

                {/* Y axis */}
                <line x1={0} x2={0} y1={0} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
                {yTicks.map((t) => (
                  <g key={t} transform={`translate(0,${yScale(t)})`}>
                    <line x2={-4} stroke="#9CA3AF" />
                    <text x={-8} textAnchor="end" dominantBaseline="middle" fontSize={9} fill="#6B7280">
                      {t.toLocaleString()}
                    </text>
                  </g>
                ))}
                <text
                  transform={`translate(${-44},${innerH / 2}) rotate(-90)`}
                  textAnchor="middle"
                  fontSize={10}
                  fill="#374151"
                >
                  Cases
                </text>

                {/* Legend on the right */}
                {visible.map((s, i) => {
                  const colorIndex = allNames.indexOf(s.name);
                  return (
                    <g
                      key={s.name}
                      transform={`translate(${innerW + 12},${i * 18 + 4})`}
                    >
                      <line
                        x1={0}
                        x2={14}
                        y1={0}
                        y2={0}
                        stroke={TREND_COLORS[colorIndex % TREND_COLORS.length]}
                        strokeWidth={2}
                        strokeDasharray={s.name === OTHER_SERIES_NAME ? '3 2' : undefined}
                      />
                      <text
                        x={18}
                        dominantBaseline="middle"
                        fontSize={9}
                        fill="#374151"
                        fontStyle={s.name === OTHER_SERIES_NAME ? 'italic' : undefined}
                      >
                        {s.name}
                      </text>
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Axis dropdown with optgroup support
// ---------------------------------------------------------------------------

function ScatterVarSelect({
  value,
  onChange,
  label,
}: {
  value: ScatterVar;
  onChange: (v: ScatterVar) => void;
  label: string;
}) {
  return (
    <label className="flex items-center gap-1.5 text-xs text-[var(--color-text-secondary)]">
      <span className="font-medium">{label}:</span>
      <select
        value={value}
        onChange={e => onChange(e.target.value as ScatterVar)}
        className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent max-w-[180px]"
      >
        {SCATTER_VAR_GROUPS.map(group => {
          const opts = SCATTER_VAR_OPTIONS.filter(o => o.group === group);
          if (opts.length === 0) return null;
          return (
            <optgroup key={group} label={group}>
              {opts.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </optgroup>
          );
        })}
      </select>
    </label>
  );
}

// ---------------------------------------------------------------------------
// Root AnalysisView
// ---------------------------------------------------------------------------

type MapId = 'vmth' | 'enviro' | 'human';
type MapCount = 2 | 3;

const MAP_OPTIONS: { id: MapId; label: string }[] = [
  { id: 'vmth', label: 'VMTH Cancer Incidence' },
  { id: 'enviro', label: 'CalEnviroScreen 4.0' },
  { id: 'human', label: 'Human Cancer Registry' },
];

export function AnalysisView() {
  // Unfiltered VMTH data for scatter plot (CancerMap owns its own filters now)
  const { countyData: unfilteredCountyData } = useFilteredData({
    rateType: 'incidence',
    sex: 'all',
    cancerType: 'All Types',
    breed: 'All Breeds',
  });

  const [selectedIndicator, setSelectedIndicator] = useState<CESIndicator>('ces_score');
  const [mapCount, setMapCount] = useState<MapCount>(3);
  const [twoMapSelection, setTwoMapSelection] = useState<[MapId, MapId]>(['vmth', 'enviro']);
  const [threeMapSelection, setThreeMapSelection] = useState<[MapId, MapId, MapId]>(['vmth', 'enviro', 'human']);
  const [geoLevel, setGeoLevel] = useState<GeoLevel>('county');
  const [scatterXVar, setScatterXVar] = useState<ScatterVar>('pesticides');
  const [scatterYVar, setScatterYVar] = useState<ScatterVar>('cancer_cases');
  const [autoSync, setAutoSync] = useState(true);

  const { data: cesData } = useCalEnviroScreenData();

  // Auto-sync handlers
  const handleIndicatorChange = (newIndicator: CESIndicator) => {
    setSelectedIndicator(newIndicator);
    if (autoSync && isCesVar(scatterXVar)) {
      setScatterXVar(newIndicator);
    }
  };

  const handleScatterXChange = (newVar: ScatterVar) => {
    setScatterXVar(newVar);
    if (autoSync && isCesVar(newVar)) {
      setSelectedIndicator(newVar as CESIndicator);
    }
  };

  const visibleMaps: MapId[] =
    mapCount === 2 ? twoMapSelection :
    threeMapSelection;

  const handleSlotChange = (slot: number, value: MapId) => {
    if (mapCount === 2) {
      setTwoMapSelection(prev => { const next = [...prev] as [MapId, MapId]; next[slot] = value; return next; });
    } else {
      setThreeMapSelection(prev => { const next = [...prev] as [MapId, MapId, MapId]; next[slot] = value; return next; });
    }
  };

  const renderMap = (id: MapId) => {
    switch (id) {
      case 'vmth':    return <CancerMap key={id} geoLevel={geoLevel} />;
      case 'enviro':  return <EnviroScreenMap key={id} data={cesData} indicator={selectedIndicator} geoLevel={geoLevel} onIndicatorChange={handleIndicatorChange} />;
      case 'human':   return <HumanCancerMap key={id} geoLevel={geoLevel} />;
    }
  };

  const gridCols =
    mapCount === 2 ? 'lg:grid-cols-2' :
    'lg:grid-cols-3';

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-start gap-4">
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed flex-1">
            Compare veterinary cancer incidence, human cancer incidence, and environmental indicators from{' '}
            <a href="https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">CalEnviroScreen 4.0</a>
            . Maps support pan &amp; zoom.
          </p>
          <div className="flex rounded-lg border border-gray-300 overflow-hidden shrink-0">
            {([2, 3] as MapCount[]).map(n => (
              <button
                key={n}
                onClick={() => setMapCount(n)}
                className={`px-3 py-1.5 text-xs font-medium border-l border-gray-300 first:border-l-0 transition-colors ${mapCount === n ? 'bg-[var(--color-teal)] text-white' : 'bg-white text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
              >
                {n} Maps
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3 pt-2 border-t border-gray-100">
          {/* Geo level segmented control */}
          <div className="flex rounded-lg border border-gray-300 overflow-hidden">
            {GEO_LEVEL_OPTIONS.map(opt => (
              <button
                key={opt.value}
                onClick={() => setGeoLevel(opt.value)}
                className={`px-3 py-1.5 text-xs font-medium border-l border-gray-300 first:border-l-0 transition-colors ${geoLevel === opt.value ? 'bg-blue-500 text-white' : 'bg-white text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          <>
            <div className="w-px h-6 bg-gray-300" />
            <span className="text-xs font-medium text-[var(--color-text-secondary)]">Show:</span>
            {Array.from({ length: mapCount }, (_, i) => (
              <select
                key={i}
                value={mapCount === 2 ? twoMapSelection[i] : threeMapSelection[i]}
                onChange={e => handleSlotChange(i, e.target.value as MapId)}
                className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
              >
                {MAP_OPTIONS.map(opt => <option key={opt.id} value={opt.id}>{opt.label}</option>)}
              </select>
            ))}
          </>
        </div>
      </div>

      {/* Maps grid */}
      <div className={`grid grid-cols-1 ${gridCols} gap-6`}>
        {visibleMaps.map(renderMap)}
      </div>

      {/* Scatter plot */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">Environmental Correlation</h3>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">Pair-wise comparison by county</p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <ScatterVarSelect label="Y" value={scatterYVar} onChange={setScatterYVar} />
            <ScatterVarSelect label="X" value={scatterXVar} onChange={handleScatterXChange} />
            <button
              onClick={() => setAutoSync(v => !v)}
              className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border font-medium transition-colors ${autoSync ? 'bg-teal-50 border-teal-300 text-teal-700' : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
              title="When enabled, changing the CES map indicator also updates the scatter X axis (and vice versa)"
            >
              <span className={`w-2 h-2 rounded-full ${autoSync ? 'bg-teal-500' : 'bg-gray-400'}`} />
              Sync
            </button>
          </div>
        </div>
        <div className="p-4">
          <CorrelationScatterPlot countyData={unfilteredCountyData} cesData={cesData} xVar={scatterXVar} yVar={scatterYVar} />
          <p className="text-xs text-[var(--color-text-secondary)] mt-3 text-center">
            Dashed line shows linear trend · Each dot is one county · Hover for details
          </p>
        </div>
      </div>

      {/* Cancer Cases by Year */}
      <CancerTrendChart />
    </div>
  );
}
