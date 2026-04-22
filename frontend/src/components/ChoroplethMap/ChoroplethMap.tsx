import { useMemo, useState } from 'react';
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
  COUNTY_GEO_URL,
  TRACT_GEO_URL,
  INITIAL_VIEW_STATE,
  HOVER_COLOR,
  hexToRgba,
  countyFromFeature,
  hoverKeyFromFeature,
} from '../../lib/mapUtils';

// Fully opaque — matches the original react-simple-maps appearance.
// Using the global NO_DATA_COLOR (alpha=180) with a transparent WebGL canvas
// can make no-data counties appear darker than expected.
const NO_DATA_COLOR: [number, number, number, number] = [229, 231, 235, 255];

// Background applied to both the container div and the DeckGL canvas container
// so the backdrop is consistently light regardless of WebGL compositing mode.
const MAP_BG_CSS = '#f1f5f9';

const EXPANDED_VIEW_STATE = { ...INITIAL_VIEW_STATE, zoom: 4.9 };

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
  const [showSuperfund, setShowSuperfund] = useState(false);
  const [tractLevel, setTractLevel] = useState(false);

  const countyDataMap = useMemo(() => {
    const m = new Map<string, CountyData>();
    data.forEach(c => m.set(c.county.toLowerCase(), c));
    return m;
  }, [data]);

  const colorScale = useMemo(
    () =>
      scaleLinear<string>()
        .domain([countRange.min, (countRange.min + countRange.max) / 2, countRange.max])
        .range(['#E6F3F5', '#6BB5BF', '#1A6B77']),
    [countRange],
  );

  // Local hover key (GEOID in tract mode, county name in county mode)
  const [localHovered, setLocalHovered] = useState<string | null>(null);

  const geoLayer = useMemo(
    () =>
      new GeoJsonLayer({
        id: 'choropleth-counties',
        data: tractLevel ? TRACT_GEO_URL : COUNTY_GEO_URL,
        pickable: true,
        stroked: true,
        filled: true,
        getFillColor: (feature) => {
          const key = hoverKeyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          const county = countyFromFeature(feature.properties as Record<string, unknown>, tractLevel);
          // Highlight from local hover OR from external hoveredCounty prop
          const isHovered =
            (key && key === localHovered) ||
            (hoveredCounty != null && county.toLowerCase() === hoveredCounty.toLowerCase());
          if (isHovered) return HOVER_COLOR;
          const info = countyDataMap.get(county.toLowerCase());
          const count = info?.count ?? 0;
          return count > 0 ? hexToRgba(colorScale(count), 255) : NO_DATA_COLOR;
        },
        getLineColor: tractLevel ? [255, 255, 255, 100] : [255, 255, 255, 255],
        lineWidthMinPixels: tractLevel ? 0.3 : 0.5,
        onHover: ({ object }) => {
          const props = object?.properties as Record<string, unknown> | null ?? null;
          const key = props ? hoverKeyFromFeature(props, tractLevel) : null;
          const county = props ? countyFromFeature(props, tractLevel) : null;
          setLocalHovered(key);
          onCountyHover?.(county ?? null);
        },
        onClick: ({ object }) => {
          if (!object) return;
          const county = countyFromFeature(object.properties as Record<string, unknown>, tractLevel);
          if (county) onCountyClick?.(county);
        },
        updateTriggers: {
          getFillColor: [countyDataMap, colorScale, localHovered, hoveredCounty, tractLevel],
          data: [tractLevel],
        },
      }),
    [countyDataMap, colorScale, localHovered, hoveredCounty, tractLevel, onCountyHover, onCountyClick],
  );

  const superfundLayer = useMemo(() => {
    if (!showSuperfund) return null;
    return new ScatterplotLayer<SuperfundSite>({
      id: 'choropleth-superfund',
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

    if (info.layer?.id === 'choropleth-counties') {
      const county = countyFromFeature(info.object.properties as Record<string, unknown>, tractLevel);
      const countyInfo = countyDataMap.get(county.toLowerCase());
      const sf = SUPERFUND_BY_COUNTY[county];
      const sfStr = sf
        ? `<br/><span style="color:#6b7280">${sf.total} Superfund site${sf.total !== 1 ? 's' : ''}</span>`
        : '';
      const header = tractLevel
        ? `<strong style="font-size:13px">Tract ${info.object.properties?.NAME as string}</strong><br/><span style="color:#6b7280">${county} County</span>`
        : `<strong style="font-size:13px">${county}</strong>`;
      const body = countyInfo
        ? `${countyInfo.count.toLocaleString()} cases${sfStr}`
        : `<span style="color:#6b7280">No data</span>`;
      return {
        html: `${header}<br/>${body}`,
        style: {
          backgroundColor: 'white',
          color: '#1f2937',
          padding: '8px 12px',
          borderRadius: '8px',
          border: '1px solid #e5e7eb',
          fontSize: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
        },
      };
    }

    if (info.layer?.id === 'choropleth-superfund') {
      const site = info.object as SuperfundSite;
      return {
        html: `<strong style="font-size:13px">${site.name}</strong><br/><span style="color:#6b7280">${site.county} Co. · ${site.status}</span><br/><span style="color:#9ca3af;font-size:11px">${site.contaminants.join(', ')}</span>`,
        style: {
          backgroundColor: 'white',
          color: '#1f2937',
          padding: '8px 12px',
          borderRadius: '8px',
          border: '1px solid #e5e7eb',
          fontSize: '12px',
          boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
          maxWidth: '220px',
        },
      };
    }

    return null;
  };

  const controls = (
    <div className="flex items-center gap-2 flex-wrap">
      <button
        onClick={() => setShowSuperfund(v => !v)}
        className={`flex items-center gap-1 text-[10px] px-2 py-1 rounded-full border font-medium transition-colors ${showSuperfund ? 'bg-red-50 border-red-300 text-red-700' : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${showSuperfund ? 'bg-red-500' : 'bg-gray-400'}`} />
        Superfund
      </button>
      <button
        onClick={() => setTractLevel(v => !v)}
        className={`flex items-center gap-1 text-[10px] px-2 py-1 rounded-full border font-medium transition-colors ${tractLevel ? 'bg-blue-50 border-blue-300 text-blue-700' : 'bg-white border-gray-300 text-[var(--color-text-secondary)] hover:bg-gray-50'}`}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${tractLevel ? 'bg-blue-500' : 'bg-gray-400'}`} />
        Census Tracts
      </button>
    </div>
  );

  const legend = (
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

  const deckMap = (mapKey: 'normal' | 'expanded') => {
    const height = mapKey === 'expanded' ? 560 : 450;
    const viewState = mapKey === 'expanded' ? EXPANDED_VIEW_STATE : INITIAL_VIEW_STATE;
    return (
      <div className="relative" style={{ height, backgroundColor: MAP_BG_CSS }}>
        <DeckGL
          key={mapKey}
          initialViewState={viewState}
          controller
          layers={layers}
          getTooltip={getTooltip}
          // background on the DeckGL container div composites behind the
          // transparent WebGL canvas, giving the map a light backdrop.
          style={{ position: 'absolute', top: '0', left: '0', right: '0', bottom: '0', background: MAP_BG_CSS }}
        />
        {legend}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden relative">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
            California County Map
          </h3>
          <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
            {tractLevel ? 'Case count by county · census tract boundaries' : 'Case count by county'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {controls}
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

      {/* Only mount one DeckGL instance at a time — two instances sharing the same
          layer objects causes the second canvas to render blank. */}
      {!isExpanded && deckMap('normal')}

      {/* Expanded modal */}
      {isExpanded && (
        <div className="fixed inset-0 z-40 bg-black/50 flex items-center justify-center">
          <div className="bg-white rounded-lg shadow-xl border border-gray-200 max-w-5xl w-full mx-4 max-h-[90vh] flex flex-col overflow-hidden">
            <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
                  California County Map
                </h3>
                <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
                  {tractLevel ? 'Case count by county · census tract boundaries' : 'Case counts by county (expanded view)'}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {controls}
                <button
                  type="button"
                  onClick={() => setIsExpanded(false)}
                  className="inline-flex items-center justify-center w-7 h-7 rounded-full border border-gray-300 text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
                >
                  <span className="sr-only">Close</span>
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>
            {deckMap('expanded')}
          </div>
        </div>
      )}
    </div>
  );
}
