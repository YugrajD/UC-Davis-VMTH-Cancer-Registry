import { useMemo, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import type { PickingInfo } from '@deck.gl/core';
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
import {
  COUNTY_GEO_URL,
  TRACT_GEO_URL,
  INITIAL_VIEW_STATE,
  NO_DATA_COLOR,
  HOVER_COLOR,
  hexToRgba,
  countyFromFeature,
  hoverKeyFromFeature,
} from '../../lib/mapUtils';

// ---------------------------------------------------------------------------
// Shared DeckGL map container
// ---------------------------------------------------------------------------

interface DeckMapProps {
  layers: (GeoJsonLayer | ScatterplotLayer)[];
  getTooltip: (info: PickingInfo) => { html: string; style?: Record<string, string | undefined> } | null;
  title: string;
  subtitle?: React.ReactNode;
  headerRight?: React.ReactNode;
  legend: React.ReactNode;
}

function DeckMap({ layers, getTooltip, title, subtitle, headerRight, legend }: DeckMapProps) {
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
      <div className="relative" style={{ height: '400px', backgroundColor: '#f8fafc' }}>
        <DeckGL
          initialViewState={INITIAL_VIEW_STATE}
          controller
          layers={layers}
          getTooltip={getTooltip}
          style={{ position: 'absolute', inset: '0' }}
        />
        {/* Legend */}
        <div className="absolute bottom-4 left-4 z-10 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm pointer-events-none">
          {legend}
        </div>
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
// Superfund ScatterplotLayer factory (shared across all maps)
// ---------------------------------------------------------------------------

function useSuperfundLayer(enabled: boolean) {
  return useMemo(() => {
    if (!enabled) return null;
    return new ScatterplotLayer<SuperfundSite>({
      id: 'superfund',
      data: MOCK_SUPERFUND_SITES,
      getPosition: d => d.coordinates,
      getFillColor: d =>
        d.status === 'active'
          ? [239, 68, 68, 230]
          : d.status === 'proposed'
            ? [249, 115, 22, 230]
            : [34, 197, 94, 230],
      getLineColor: [255, 255, 255, 255],
      lineWidthMinPixels: 1,
      radiusMinPixels: 5,
      radiusMaxPixels: 14,
      getRadius: 3000,
      pickable: true,
      updateTriggers: { getFillColor: [] },
    });
  }, [enabled]);
}

// ---------------------------------------------------------------------------
// VMTH Cancer Incidence map
// ---------------------------------------------------------------------------

function CancerMap({
  countyData,
  countRange,
  showSuperfund,
  tractLevel,
}: {
  countyData: CountyData[];
  countRange: { min: number; max: number };
  showSuperfund: boolean;
  tractLevel: boolean;
}) {
  const countyDataMap = useMemo(() => {
    const m = new Map<string, CountyData>();
    countyData.forEach(c => m.set(c.county.toLowerCase(), c));
    return m;
  }, [countyData]);

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([countRange.min, (countRange.min + countRange.max) / 2, countRange.max])
        .range(['#E6F3F5', '#6BB5BF', '#1A6B77']),
    [countRange],
  );

  const [hovered, setHovered] = useState<string | null>(null);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'cancer-counties',
        data: tractLevel ? TRACT_GEO_URL : COUNTY_GEO_URL,
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          if (key && key === hovered) return HOVER_COLOR;
          const county = countyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          const info = countyDataMap.get(county.toLowerCase());
          const count = info?.count ?? 0;
          return count > 0 ? hexToRgba(colorScale(count)) : NO_DATA_COLOR;
        },
        getLineColor: tractLevel ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: tractLevel ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, tractLevel) : null),
        updateTriggers: { getFillColor: [countyDataMap, colorScale, hovered, tractLevel], data: [tractLevel] },
      }),
    [countyDataMap, colorScale, hovered, tractLevel],
  );

  const superfundLayer = useSuperfundLayer(showSuperfund);
  const layers = useMemo(
    () => [geoLayer, ...(superfundLayer ? [superfundLayer] : [])],
    [geoLayer, superfundLayer],
  );

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'cancer-counties') {
      const county = countyFromFeature(info.object.properties as Record<string, unknown>, tractLevel);
      const count = countyDataMap.get(county.toLowerCase())?.count ?? 0;
      const sf = SUPERFUND_BY_COUNTY[county];
      const sfStr = sf ? `<br/><span style="color:#6b7280">${sf.total} Superfund site${sf.total !== 1 ? 's' : ''}</span>` : '';
      const header = tractLevel
        ? `<strong style="font-size:13px">Tract ${info.object.properties?.NAME as string}</strong><br/><span style="color:#6b7280">${county} County</span>`
        : `<strong style="font-size:13px">${county}</strong>`;
      return {
        html: `${header}<br/>${count.toLocaleString()} cases${sfStr}`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    if (info.layer?.id === 'superfund') {
      const site = info.object as SuperfundSite;
      return {
        html: `<strong style="font-size:13px">${site.name}</strong><br/><span style="color:#6b7280">${site.county} Co. · ${site.status}</span><br/><span style="color:#9ca3af;font-size:11px">${site.contaminants.join(', ')}</span>`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)', maxWidth: '200px' },
      };
    }
    return null;
  };

  return (
    <DeckMap
      layers={layers}
      getTooltip={getTooltip}
      title="Cancer Incidence"
      subtitle={tractLevel ? 'Case count by county · census tract boundaries' : 'Case count by county'}
      legend={
        <GradientLegend
          label="Cases"
          gradient="linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)"
          min={String(countRange.min)}
          max={String(countRange.max)}
          extra={
            showSuperfund ? (
              <div className="mt-2 pt-2 border-t border-gray-100 space-y-1">
                {[{ color: '#EF4444', label: 'Active' }, { color: '#F97316', label: 'Proposed' }, { color: '#22C55E', label: 'Remediated' }].map(({ color, label }) => (
                  <div key={label} className="flex items-center gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full border border-white shadow-sm" style={{ backgroundColor: color }} />
                    <span className="text-[10px] text-[var(--color-text-secondary)]">{label} Superfund</span>
                  </div>
                ))}
              </div>
            ) : null
          }
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
  showSuperfund,
  tractLevel,
  onIndicatorChange,
}: {
  data: CalEnviroScreenData[];
  indicator: CESIndicator;
  showSuperfund: boolean;
  tractLevel: boolean;
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
        data: tractLevel ? TRACT_GEO_URL : COUNTY_GEO_URL,
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          if (key && key === hovered) return HOVER_COLOR;
          const county = countyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          const val = countyValueMap.get(county.toLowerCase());
          return val != null ? hexToRgba(colorScale(val)) : NO_DATA_COLOR;
        },
        getLineColor: tractLevel ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: tractLevel ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, tractLevel) : null),
        updateTriggers: { getFillColor: [countyValueMap, colorScale, hovered, tractLevel], data: [tractLevel] },
      }),
    [countyValueMap, colorScale, hovered, tractLevel],
  );

  const superfundLayer = useSuperfundLayer(showSuperfund);
  const layers = useMemo(
    () => [geoLayer, ...(superfundLayer ? [superfundLayer] : [])],
    [geoLayer, superfundLayer],
  );

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'enviro-counties') {
      const county = countyFromFeature(info.object.properties as Record<string, unknown>, tractLevel);
      const val = countyValueMap.get(county.toLowerCase());
      const header = tractLevel
        ? `<strong style="font-size:13px">Tract ${info.object.properties?.NAME as string}</strong><br/><span style="color:#6b7280">${county} County</span>`
        : `<strong style="font-size:13px">${county}</strong>`;
      return {
        html: `${header}<br/>${val != null ? `${indicatorLabel}: <strong>${val.toFixed(1)}</strong>` : 'No data'}`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    if (info.layer?.id === 'superfund') {
      const site = info.object as SuperfundSite;
      return {
        html: `<strong style="font-size:13px">${site.name}</strong><br/><span style="color:#6b7280">${site.county} Co. · ${site.status}</span>`,
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
        <select
          value={indicator}
          onChange={(e) => onIndicatorChange(e.target.value as CESIndicator)}
          className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
        >
          {CES_INDICATORS.map((ind) => (
            <option key={ind.value} value={ind.value}>{ind.label}</option>
          ))}
        </select>
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

function HumanCancerMap({ showSuperfund, tractLevel }: { showSuperfund: boolean; tractLevel: boolean }) {
  const rateMap = useMemo(() => {
    const m = new Map<string, { rate: number | null; cases: number | null }>();
    HUMAN_CANCER_RATES.forEach(d => m.set(d.county.toLowerCase(), { rate: d.rate, cases: d.cases }));
    return m;
  }, []);

  const rateRange = useMemo(() => {
    const vals = HUMAN_CANCER_RATES.map(d => d.rate).filter((v): v is number => v !== null);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, []);

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
        data: tractLevel ? TRACT_GEO_URL : COUNTY_GEO_URL,
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          if (key && key === hovered) return HOVER_COLOR;
          const county = countyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          const info = rateMap.get(county.toLowerCase());
          const rate = info?.rate;
          return rate != null ? hexToRgba(colorScale(rate)) : NO_DATA_COLOR;
        },
        getLineColor: tractLevel ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: tractLevel ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, tractLevel) : null),
        updateTriggers: { getFillColor: [rateMap, colorScale, hovered, tractLevel], data: [tractLevel] },
      }),
    [rateMap, colorScale, hovered, tractLevel],
  );

  const superfundLayer = useSuperfundLayer(showSuperfund);
  const layers = useMemo(
    () => [geoLayer, ...(superfundLayer ? [superfundLayer] : [])],
    [geoLayer, superfundLayer],
  );

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'human-counties') {
      const county = countyFromFeature(info.object.properties as Record<string, unknown>, tractLevel);
      const info2 = rateMap.get(county.toLowerCase());
      const rate = info2?.rate;
      const casesStr = info2?.cases != null ? ` (${info2.cases.toLocaleString()}/yr)` : '';
      const header = tractLevel
        ? `<strong style="font-size:13px">Tract ${info.object.properties?.NAME as string}</strong><br/><span style="color:#6b7280">${county} County</span>`
        : `<strong style="font-size:13px">${county}</strong>`;
      return {
        html: `${header}<br/>${rate != null ? `${rate.toFixed(1)} per 100K${casesStr}` : 'Suppressed'}`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    if (info.layer?.id === 'superfund') {
      const site = info.object as SuperfundSite;
      return {
        html: `<strong style="font-size:13px">${site.name}</strong><br/><span style="color:#6b7280">${site.county} Co. · ${site.status}</span>`,
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
// Pesticide Use map
// ---------------------------------------------------------------------------

function PesticideMap({ showSuperfund, tractLevel }: { showSuperfund: boolean; tractLevel: boolean }) {
  const valueRange = useMemo(() => {
    const vals = MOCK_PESTICIDE_DATA.map(d => d.lbs_per_sq_mile);
    return { min: Math.min(...vals), max: Math.max(...vals) };
  }, []);

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([valueRange.min, (valueRange.min + valueRange.max) / 2, valueRange.max])
        .range(['#FEF9C3', '#F97316', '#7F1D1D']),
    [valueRange],
  );

  const [hovered, setHovered] = useState<string | null>(null);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'pesticide-counties',
        data: tractLevel ? TRACT_GEO_URL : COUNTY_GEO_URL,
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          if (key && key === hovered) return [96, 165, 250, 220] as [number, number, number, number];
          const county = countyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          const data = PESTICIDE_BY_COUNTY[county];
          return data ? hexToRgba(colorScale(data.lbs_per_sq_mile)) : NO_DATA_COLOR;
        },
        getLineColor: tractLevel ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: tractLevel ? 0.3 : 0.5,
        onHover: ({ object }) =>
          setHovered(object ? hoverKeyFromFeature(object.properties as Record<string, unknown>, tractLevel) : null),
        updateTriggers: { getFillColor: [colorScale, hovered, tractLevel], data: [tractLevel] },
      }),
    [colorScale, hovered, tractLevel],
  );

  const superfundLayer = useSuperfundLayer(showSuperfund);
  const layers = useMemo(
    () => [geoLayer, ...(superfundLayer ? [superfundLayer] : [])],
    [geoLayer, superfundLayer],
  );

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;
    if (info.layer?.id === 'pesticide-counties') {
      const county = countyFromFeature(info.object.properties as Record<string, unknown>, tractLevel);
      const data = PESTICIDE_BY_COUNTY[county];
      const header = tractLevel
        ? `<strong style="font-size:13px">Tract ${info.object.properties?.NAME as string}</strong><br/><span style="color:#6b7280">${county} County</span>`
        : `<strong style="font-size:13px">${county}</strong>`;
      return {
        html: `${header}<br/>${data ? `${data.lbs_per_sq_mile.toLocaleString()} lbs/sq mi<br/><span style="color:#6b7280">${data.top_pesticide_class}</span>` : 'No data'}`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    if (info.layer?.id === 'superfund') {
      const site = info.object as SuperfundSite;
      return {
        html: `<strong style="font-size:13px">${site.name}</strong><br/><span style="color:#6b7280">${site.county} Co. · ${site.status}</span>`,
        style: { backgroundColor: 'white', color: '#1f2937', padding: '8px 12px', borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px', boxShadow: '0 2px 8px rgba(0,0,0,0.12)' },
      };
    }
    return null;
  };

  return (
    <DeckMap
      layers={layers}
      getTooltip={getTooltip}
      title="Pesticide Use"
      subtitle={
        <>
          Avg annual lbs active ingredient / sq mi (2015–2019) &middot;{' '}
          <a href="https://trackingcalifornia.org/data-and-tools/pesticide-mapping-tool" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">Source</a>
        </>
      }
      legend={
        <GradientLegend
          label="lbs / sq mile"
          gradient="linear-gradient(to right, #FEF9C3, #F97316, #7F1D1D)"
          min={String(valueRange.min)}
          max={String(valueRange.max)}
        />
      }
    />
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
      if (xVar === 'pesticide') x = PESTICIDE_BY_COUNTY[c.county]?.lbs_per_sq_mile ?? null;
      else if (xVar === 'superfund') x = SUPERFUND_BY_COUNTY[c.county]?.total ?? 0;
      else if (xVar === 'ces_score') x = cesMap.get(c.county)?.ces_score ?? null;
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
    const slope =
      points.reduce((s, p) => s + (p.x - meanX) * (p.y - meanY), 0) /
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
    return <div className="flex items-center justify-center h-40 text-sm text-[var(--color-text-secondary)]">No overlapping county data for this variable.</div>;
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
                  <rect x={xScale(p.x) + 8} y={yScale(p.y) - 30} width={140} height={38} rx={4} fill="white" stroke="#E5E7EB" strokeWidth={1} filter="drop-shadow(0 1px 3px rgba(0,0,0,0.15))" />
                  <text x={xScale(p.x) + 14} y={yScale(p.y) - 14} fontSize={10} fontWeight="600" fill="#1F2937">{p.county}</text>
                  <text x={xScale(p.x) + 14} y={yScale(p.y) + 2} fontSize={9} fill="#6B7280">{xVarMeta.label}: {p.x.toLocaleString()} · Cases: {p.y.toLocaleString()}</text>
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
          <text x={innerW / 2} y={innerH + 40} textAnchor="middle" fontSize={10} fill="#374151">{xVarMeta.label} ({xVarMeta.unit})</text>
          <line x1={0} x2={0} y1={0} y2={innerH} stroke="#9CA3AF" strokeWidth={1} />
          {yTicks.map(t => (
            <g key={t} transform={`translate(0,${yScale(t)})`}>
              <line x2={-4} stroke="#9CA3AF" />
              <text x={-8} textAnchor="end" dominantBaseline="middle" fontSize={9} fill="#6B7280">{t.toLocaleString()}</text>
            </g>
          ))}
          <text transform={`translate(${-44},${innerH / 2}) rotate(-90)`} textAnchor="middle" fontSize={10} fill="#374151">Cancer Cases</text>
        </g>
      </svg>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Root AnalysisView
// ---------------------------------------------------------------------------

type MapId = 'vmth' | 'enviro' | 'human' | 'pesticide';
type MapCount = 2 | 3 | 4;

const MAP_OPTIONS: { id: MapId; label: string }[] = [
  { id: 'vmth', label: 'VMTH Cancer Incidence' },
  { id: 'enviro', label: 'CalEnviroScreen 4.0' },
  { id: 'human', label: 'Human Cancer Registry' },
  { id: 'pesticide', label: 'Pesticide Use' },
];

interface AnalysisViewProps {
  countyData: CountyData[];
  countRange: { min: number; max: number };
}

export function AnalysisView({ countyData, countRange }: AnalysisViewProps) {
  const [selectedIndicator, setSelectedIndicator] = useState<CESIndicator>('ces_score');
  const [mapCount, setMapCount] = useState<MapCount>(3);
  const [twoMapSelection, setTwoMapSelection] = useState<[MapId, MapId]>(['vmth', 'enviro']);
  const [threeMapSelection, setThreeMapSelection] = useState<[MapId, MapId, MapId]>(['vmth', 'enviro', 'human']);
  const [showSuperfund, setShowSuperfund] = useState(false);
  const [tractLevel, setTractLevel] = useState(false);
  const [scatterXVar, setScatterXVar] = useState<ScatterXVar>('pesticide');

  const { data: cesData } = useCalEnviroScreenData();

  const visibleMaps: MapId[] =
    mapCount === 2 ? twoMapSelection :
    mapCount === 3 ? threeMapSelection :
    ['vmth', 'enviro', 'human', 'pesticide'];

  const handleSlotChange = (slot: number, value: MapId) => {
    if (mapCount === 2) {
      setTwoMapSelection(prev => { const next = [...prev] as [MapId, MapId]; next[slot] = value; return next; });
    } else {
      setThreeMapSelection(prev => { const next = [...prev] as [MapId, MapId, MapId]; next[slot] = value; return next; });
    }
  };

  const renderMap = (id: MapId) => {
    switch (id) {
      case 'vmth':    return <CancerMap key={id} countyData={countyData} countRange={countRange} showSuperfund={showSuperfund} tractLevel={tractLevel} />;
      case 'enviro':  return <EnviroScreenMap key={id} data={cesData} indicator={selectedIndicator} showSuperfund={showSuperfund} tractLevel={tractLevel} onIndicatorChange={setSelectedIndicator} />;
      case 'human':   return <HumanCancerMap key={id} showSuperfund={showSuperfund} tractLevel={tractLevel} />;
      case 'pesticide': return <PesticideMap key={id} showSuperfund={showSuperfund} tractLevel={tractLevel} />;
    }
  };

  const gridCols =
    mapCount === 2 ? 'lg:grid-cols-2' :
    mapCount === 3 ? 'lg:grid-cols-3' :
    'lg:grid-cols-2 xl:grid-cols-4';

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-start gap-4">
          <p className="text-sm text-[var(--color-text-secondary)] leading-relaxed flex-1">
            Compare veterinary cancer incidence, human cancer incidence, environmental indicators from{' '}
            <a href="https://oehha.ca.gov/calenviroscreen/report/calenviroscreen-40" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">CalEnviroScreen 4.0</a>
            , and pesticide use from{' '}
            <a href="https://trackingcalifornia.org/data-and-tools/pesticide-mapping-tool" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">CDPR</a>
            . Maps support pan &amp; zoom. Toggle{' '}
            <a href="https://www.epa.gov/superfund/search-superfund-sites-where-you-live" target="_blank" rel="noopener noreferrer" className="text-[var(--color-teal)] underline hover:text-[var(--color-teal-dark)]">EPA Superfund sites</a>
            {' '}as an overlay on any map.
          </p>
          <div className="flex rounded-lg border border-gray-300 overflow-hidden shrink-0">
            {([2, 3, 4] as MapCount[]).map(n => (
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
          <button
            onClick={() => setShowSuperfund(v => !v)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border font-medium transition-colors ${showSuperfund ? 'bg-red-50 border-red-300 text-red-700' : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
          >
            <span className={`w-2 h-2 rounded-full ${showSuperfund ? 'bg-red-500' : 'bg-gray-400'}`} />
            Superfund Sites
          </button>

          <button
            onClick={() => setTractLevel(v => !v)}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full border font-medium transition-colors ${tractLevel ? 'bg-blue-50 border-blue-300 text-blue-700' : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
          >
            <span className={`w-2 h-2 rounded-full ${tractLevel ? 'bg-blue-500' : 'bg-gray-400'}`} />
            Census Tract Level
          </button>

          {mapCount < 4 && (
            <>
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
          )}
        </div>
      </div>

      {/* Maps grid */}
      <div className={`grid grid-cols-1 ${gridCols} gap-6`}>
        {visibleMaps.map(renderMap)}
      </div>

      {/* Scatter plot */}
      <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">Environmental Correlation</h3>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">Cancer cases vs. environmental exposure by county</p>
          </div>
          <select
            value={scatterXVar}
            onChange={e => setScatterXVar(e.target.value as ScatterXVar)}
            className="text-xs border border-gray-300 rounded-md px-2 py-1.5 bg-white text-[var(--color-text-primary)] focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent"
          >
            {X_VAR_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label} ({o.unit})</option>)}
          </select>
        </div>
        <div className="p-4">
          <CorrelationScatterPlot countyData={countyData} cesData={cesData} xVar={scatterXVar} />
          <p className="text-xs text-[var(--color-text-secondary)] mt-3 text-center">
            Dashed line shows linear trend · Each dot is one county · Hover for details
          </p>
        </div>
      </div>
    </div>
  );
}
