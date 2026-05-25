import { useEffect, useMemo, useState } from 'react';
import DeckGL from '@deck.gl/react';
import { GeoJsonLayer, ScatterplotLayer } from '@deck.gl/layers';
import type { PickingInfo } from '@deck.gl/core';
import { scaleLinear } from 'd3-scale';
import type { CountyData } from '../../types';
import {
  MOCK_SUPERFUND_SITES,
  SUPERFUND_BY_COUNTY,
  type SuperfundSite,
} from '../../data/superfundData';
import {
  GEO_URLS,
  INITIAL_VIEW_STATE,
  HOVER_COLOR,
  hexToRgba,
  countyFromFeature,
  hoverKeyFromFeature,
  type GeoLevel,
} from '../../lib/mapUtils';
import { MapResetButton } from '../MapResetButton/MapResetButton';

// Local NO_DATA_COLOR — slightly transparent so the map background shows through.
const NO_DATA_COLOR: [number, number, number, number] = [229, 231, 235, 180];

// Background applied to both the container div and the DeckGL canvas container.
const MAP_BG_CSS = '#f1f5f9';

// Both the normal and expanded maps default to INITIAL_VIEW_STATE, which is
// already sized to fit all of California with margin.

function isAtDefaultView(v: typeof INITIAL_VIEW_STATE): boolean {
  return (
    v.longitude === INITIAL_VIEW_STATE.longitude &&
    v.latitude === INITIAL_VIEW_STATE.latitude &&
    v.zoom === INITIAL_VIEW_STATE.zoom &&
    v.pitch === INITIAL_VIEW_STATE.pitch &&
    v.bearing === INITIAL_VIEW_STATE.bearing
  );
}

const GEO_LEVEL_OPTIONS: { value: GeoLevel; label: string }[] = [
  { value: 'county', label: 'County' },
  { value: 'tract', label: 'Tract' },
  { value: 'zcta', label: 'ZCTA' },
];

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

function makeColorScale(countRange: { min: number; max: number }) {
  return scaleLinear<string>()
    .domain([countRange.min, (countRange.min + countRange.max) / 2, countRange.max])
    .range(['#E6F3F5', '#6BB5BF', '#1A6B77']);
}

function makeCountyDataMap(data: CountyData[]) {
  const m = new Map<string, CountyData>();
  data.forEach(c => m.set(c.county.toLowerCase(), c));
  return m;
}

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

function GeoLevelSelector({ value, onChange }: { value: GeoLevel; onChange: (v: GeoLevel) => void }) {
  return (
    <div className="flex rounded-lg border border-gray-300 overflow-hidden">
      {GEO_LEVEL_OPTIONS.map(opt => (
        <button
          key={opt.value}
          onClick={() => onChange(opt.value)}
          className={`px-2 py-1 text-[10px] font-medium border-l border-gray-300 first:border-l-0 transition-colors ${value === opt.value ? 'bg-blue-500 text-white' : 'bg-white text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
        >
          {opt.label}
        </button>
      ))}
    </div>
  );
}

function MapLegend({
  countRange,
  showSuperfund,
}: {
  countRange: { min: number; max: number };
  showSuperfund: boolean;
}) {
  return (
    <div className="absolute bottom-4 left-4 z-10 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm pointer-events-none">
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
      {showSuperfund && (
        <div className="mt-2 pt-2 border-t border-gray-100 space-y-1">
          {[
            { color: '#EF4444', label: 'Active' },
            { color: '#F97316', label: 'Proposed' },
            { color: '#22C55E', label: 'Remediated' },
          ].map(({ color, label }) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full border border-white shadow-sm" style={{ backgroundColor: color }} />
              <span className="text-[10px] text-[var(--color-text-secondary)]">{label} Superfund</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Expanded modal — owns its own state so toggles don't affect the normal map
// ---------------------------------------------------------------------------

interface ExpandedMapProps {
  data: CountyData[];
  countRange: { min: number; max: number };
  onClose: () => void;
}

function ExpandedMap({ data, countRange, onClose }: ExpandedMapProps) {
  const [showSuperfund, setShowSuperfund] = useState(false);
  const [geoLevel, setGeoLevel] = useState<GeoLevel>('county');
  const [localHovered, setLocalHovered] = useState<string | null>(null);
  const [expandedViewState, setExpandedViewState] =
    useState<typeof INITIAL_VIEW_STATE>(INITIAL_VIEW_STATE);

  const countyDataMap = useMemo(() => makeCountyDataMap(data), [data]);
  const colorScale = useMemo(() => makeColorScale(countRange), [countRange]);

  // DeckGL canvas may not size itself correctly when mounted inside a modal
  // before CSS layout settles. Dispatching a resize event fixes this.
  useEffect(() => {
    const id = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')));
    return () => cancelAnimationFrame(id);
  }, []);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'expanded-choropleth-counties',
        data: GEO_URLS[geoLevel],
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          const county = countyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          if (key && key === localHovered) return HOVER_COLOR;
          const info = countyDataMap.get(county.toLowerCase());
          const count = info?.count ?? 0;
          return count > 0 ? hexToRgba(colorScale(count)) : NO_DATA_COLOR;
        },
        getLineColor: geoLevel !== 'county' ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: geoLevel !== 'county' ? 0.3 : 0.5,
        onHover: ({ object }) => {
          const props = object?.properties as Record<string, unknown> | null ?? null;
          setLocalHovered(props ? hoverKeyFromFeature(props, geoLevel) : null);
        },
        updateTriggers: {
          getFillColor: [countyDataMap, colorScale, localHovered, geoLevel],
          data: [geoLevel],
        },
      }),
    [countyDataMap, colorScale, localHovered, geoLevel],
  );

  const superfundLayer = useMemo(() => {
    if (!showSuperfund) return null;
    return new ScatterplotLayer<SuperfundSite>({
      id: 'expanded-choropleth-superfund',
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
    });
  }, [showSuperfund]);

  const layers = useMemo(
    () => [geoLayer, ...(superfundLayer ? [superfundLayer] : [])],
    [geoLayer, superfundLayer],
  );

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;

    if (info.layer?.id === 'expanded-choropleth-counties') {
      const props = info.object.properties as Record<string, unknown>;
      const county = countyFromFeature(props, geoLevel);
      const countyInfo = countyDataMap.get(county.toLowerCase());
      const sf = SUPERFUND_BY_COUNTY[county];
      const sfStr = sf
        ? `<br/><span style="color:#6b7280">${sf.total} Superfund site${sf.total !== 1 ? 's' : ''}</span>`
        : '';
      const header = tooltipHeader(props, geoLevel, county);
      const body = countyInfo
        ? `${countyInfo.count.toLocaleString()} cases${sfStr}`
        : `<span style="color:#6b7280">No data</span>`;
      return {
        html: `${header}<br/>${body}`,
        style: {
          backgroundColor: 'white', color: '#1f2937', padding: '8px 12px',
          borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
        },
      };
    }

    if (info.layer?.id === 'expanded-choropleth-superfund') {
      const site = info.object as SuperfundSite;
      return {
        html: `<strong style="font-size:13px">${site.name}</strong><br/><span style="color:#6b7280">${site.county} Co. · ${site.status}</span><br/><span style="color:#9ca3af;font-size:11px">${site.contaminants.join(', ')}</span>`,
        style: {
          backgroundColor: 'white', color: '#1f2937', padding: '8px 12px',
          borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)', maxWidth: '220px',
        },
      };
    }

    return null;
  };

  const subtitleText = geoLevel === 'county'
    ? 'Case counts by county (expanded view)'
    : geoLevel === 'tract'
      ? 'Case count by county · census tract boundaries'
      : 'Case count by county · ZCTA boundaries';

  return (
    <div className="fixed inset-0 z-40 bg-black/50 flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-xl border border-gray-200 max-w-5xl w-full mx-4 max-h-[90vh] flex flex-col overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
          <div>
            <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
              California County Map
            </h3>
            <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
              {subtitleText}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={() => setShowSuperfund(v => !v)}
                className={`flex items-center gap-1 text-[10px] px-2 py-1 rounded-full border font-medium transition-colors ${showSuperfund ? 'bg-red-50 border-red-300 text-red-700' : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
              >
                <span className={`w-1.5 h-1.5 rounded-full ${showSuperfund ? 'bg-red-500' : 'bg-gray-400'}`} />
                Superfund
              </button>
              <GeoLevelSelector value={geoLevel} onChange={setGeoLevel} />
            </div>
            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center justify-center w-7 h-7 rounded-full border border-gray-300 text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
            >
              <span className="sr-only">Close</span>
              <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
        <div className="relative" style={{ height: 560, backgroundColor: MAP_BG_CSS }}>
          <DeckGL
            viewState={expandedViewState}
            onViewStateChange={(params) => setExpandedViewState(params.viewState as typeof INITIAL_VIEW_STATE)}
            controller
            layers={layers}
            getTooltip={getTooltip}
            style={{ position: 'absolute', top: '0', left: '0', right: '0', bottom: '0', background: MAP_BG_CSS }}
          />
          <MapLegend countRange={countRange} showSuperfund={showSuperfund} />
          <MapResetButton
            onClick={() => setExpandedViewState(INITIAL_VIEW_STATE)}
            disabled={isAtDefaultView(expandedViewState)}
          />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main ChoroplethMap — normal (non-expanded) view only
// ---------------------------------------------------------------------------

interface ChoroplethMapProps {
  data: CountyData[];
  countRange: { min: number; max: number };
  hoveredCounty?: string | null;
  onCountyHover?: (county: string | null) => void;
  onCountyClick?: (county: string) => void;
}

export function ChoroplethMap({
  data,
  countRange,
  hoveredCounty,
  onCountyHover,
  onCountyClick,
}: ChoroplethMapProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [geoLevel, setGeoLevel] = useState<GeoLevel>('county');
  const [localHovered, setLocalHovered] = useState<string | null>(null);
  const [viewState, setViewState] =
    useState<typeof INITIAL_VIEW_STATE>(INITIAL_VIEW_STATE);

  const countyDataMap = useMemo(() => makeCountyDataMap(data), [data]);
  const colorScale = useMemo(() => makeColorScale(countRange), [countRange]);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'choropleth-counties',
        data: GEO_URLS[geoLevel],
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          const county = countyFromFeature(feature.properties as Record<string, unknown>, geoLevel);
          const isHovered =
            (key && key === localHovered) ||
            (hoveredCounty != null && county.toLowerCase() === hoveredCounty.toLowerCase());
          if (isHovered) return HOVER_COLOR;
          const info = countyDataMap.get(county.toLowerCase());
          const count = info?.count ?? 0;
          return count > 0 ? hexToRgba(colorScale(count)) : NO_DATA_COLOR;
        },
        getLineColor: geoLevel !== 'county' ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: geoLevel !== 'county' ? 0.3 : 0.5,
        onHover: ({ object }) => {
          const props = object?.properties as Record<string, unknown> | null ?? null;
          const key = props ? hoverKeyFromFeature(props, geoLevel) : null;
          const county = props ? countyFromFeature(props, geoLevel) : null;
          setLocalHovered(key);
          onCountyHover?.(county ?? null);
        },
        onClick: ({ object }) => {
          if (!object) return;
          const county = countyFromFeature(object.properties as Record<string, unknown>, geoLevel);
          if (county) onCountyClick?.(county);
        },
        updateTriggers: {
          getFillColor: [countyDataMap, colorScale, localHovered, hoveredCounty, geoLevel],
          data: [geoLevel],
        },
      }),
    [countyDataMap, colorScale, localHovered, hoveredCounty, geoLevel, onCountyHover, onCountyClick],
  );

  const layers = useMemo(() => [geoLayer], [geoLayer]);

  const getTooltip = (info: PickingInfo) => {
    if (!info.object) return null;

    if (info.layer?.id === 'choropleth-counties') {
      const props = info.object.properties as Record<string, unknown>;
      const county = countyFromFeature(props, geoLevel);
      const countyInfo = countyDataMap.get(county.toLowerCase());
      const header = tooltipHeader(props, geoLevel, county);
      const body = countyInfo
        ? `${countyInfo.count.toLocaleString()} cases`
        : `<span style="color:#6b7280">No data</span>`;
      return {
        html: `${header}<br/>${body}`,
        style: {
          backgroundColor: 'white', color: '#1f2937', padding: '8px 12px',
          borderRadius: '8px', border: '1px solid #e5e7eb', fontSize: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
        },
      };
    }

    return null;
  };

  const subtitleText = geoLevel === 'county'
    ? 'Case count by county'
    : geoLevel === 'tract'
      ? 'Case count by county · census tract boundaries'
      : 'Case count by county · ZCTA boundaries';

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden relative">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
            California County Map
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
            {subtitleText}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <GeoLevelSelector value={geoLevel} onChange={setGeoLevel} />
          <button
            type="button"
            onClick={() => setIsExpanded(true)}
            className="inline-flex items-center gap-1 px-3 py-1.5 rounded-md text-xs font-medium border border-[var(--color-teal)] text-[var(--color-teal)] hover:bg-[var(--color-teal)] hover:text-white transition-colors"
          >
            Expand
            <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4h4M16 4h4v4M4 16v4h4M16 20h4v-4" />
            </svg>
          </button>
        </div>
      </div>

      {/* Normal map — always mounted so it renders immediately */}
      <div className="relative" style={{ height: 450, backgroundColor: MAP_BG_CSS }}>
        <DeckGL
          viewState={viewState}
          onViewStateChange={(params) => setViewState(params.viewState as typeof INITIAL_VIEW_STATE)}
          controller
          layers={layers}
          getTooltip={getTooltip}
          style={{ position: 'absolute', top: '0', left: '0', right: '0', bottom: '0', background: MAP_BG_CSS }}
        />
        <MapLegend countRange={countRange} showSuperfund={false} />
        <MapResetButton
          onClick={() => setViewState(INITIAL_VIEW_STATE)}
          disabled={isAtDefaultView(viewState)}
        />
      </div>

      {/* Expanded modal — separate component with its own independent state */}
      {isExpanded && (
        <ExpandedMap
          data={data}
          countRange={countRange}
          onClose={() => setIsExpanded(false)}
        />
      )}
    </div>
  );
}
