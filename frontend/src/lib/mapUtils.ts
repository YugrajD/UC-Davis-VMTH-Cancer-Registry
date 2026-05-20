// Shared constants and helpers for all Deck.gl map components.

export type GeoLevel = 'county' | 'tract' | 'zcta';

export const COUNTY_GEO_URL = '/california-counties.geojson';

export const TRACT_GEO_URL = '/california-tracts.geojson';

export const ZCTA_GEO_URL = '/california-zctas.geojson';

export const GEO_URLS: Record<GeoLevel, string> = {
  county: COUNTY_GEO_URL,
  tract: TRACT_GEO_URL,
  zcta: ZCTA_GEO_URL,
};

// Default Web Mercator camera that fits the entire state of California
// with visible margin on all four sides.
//
// California's extent: 32.5°N–42°N latitude, -124.4°W to -114.1°W longitude.
// Latitude midpoint in Mercator is ~37.4°N (slightly north of the
// geographic midpoint because Mercator stretches higher latitudes).
// Longitude midpoint is -119.25°W.  Zoom 4.5 leaves the full state
// visible even on narrow ~290 px-wide map cells in the Analysis 4-across
// grid layout.
export const INITIAL_VIEW_STATE = {
  longitude: -119.25,
  latitude: 37.4,
  zoom: 4.5,
  pitch: 0,
  bearing: 0,
};

export const NO_DATA_COLOR: [number, number, number, number] = [229, 231, 235, 180];
export const HOVER_COLOR: [number, number, number, number] = [245, 166, 35, 220];

/**
 * Convert a d3-scale color string to a DeckGL RGBA tuple.
 * d3.scaleLinear<string>() returns "rgb(r, g, b)" via d3-interpolate, not hex.
 * This handles both formats so fills render correctly.
 */
export function hexToRgba(color: string, alpha = 200): [number, number, number, number] {
  const rgb = color.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)/);
  if (rgb) {
    return [parseInt(rgb[1]), parseInt(rgb[2]), parseInt(rgb[3]), alpha];
  }
  const h = color.replace('#', '');
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
    alpha,
  ];
}

// California county FIPS → county name (used with tract GeoJSON COUNTYFP property)
export const CA_COUNTY_FIPS: Record<string, string> = {
  '001': 'Alameda',        '003': 'Alpine',        '005': 'Amador',
  '007': 'Butte',          '009': 'Calaveras',     '011': 'Colusa',
  '013': 'Contra Costa',   '015': 'Del Norte',     '017': 'El Dorado',
  '019': 'Fresno',         '021': 'Glenn',          '023': 'Humboldt',
  '025': 'Imperial',       '027': 'Inyo',           '029': 'Kern',
  '031': 'Kings',          '033': 'Lake',           '035': 'Lassen',
  '037': 'Los Angeles',    '039': 'Madera',         '041': 'Marin',
  '043': 'Mariposa',       '045': 'Mendocino',     '047': 'Merced',
  '049': 'Modoc',          '051': 'Mono',           '053': 'Monterey',
  '055': 'Napa',           '057': 'Nevada',         '059': 'Orange',
  '061': 'Placer',         '063': 'Plumas',         '065': 'Riverside',
  '067': 'Sacramento',     '069': 'San Benito',     '071': 'San Bernardino',
  '073': 'San Diego',      '075': 'San Francisco',  '077': 'San Joaquin',
  '079': 'San Luis Obispo','081': 'San Mateo',      '083': 'Santa Barbara',
  '085': 'Santa Clara',    '087': 'Santa Cruz',     '089': 'Shasta',
  '091': 'Sierra',         '093': 'Siskiyou',      '095': 'Solano',
  '097': 'Sonoma',         '099': 'Stanislaus',    '101': 'Sutter',
  '103': 'Tehama',         '105': 'Trinity',        '107': 'Tulare',
  '109': 'Tuolumne',       '111': 'Ventura',       '113': 'Yolo',
  '115': 'Yuba',
};

/**
 * Get the county name from a GeoJSON feature.
 * County GeoJSON: feature.properties.name
 * Tract  GeoJSON: CA_COUNTY_FIPS[feature.properties.COUNTYFP]
 * ZCTA   GeoJSON: feature.properties.COUNTY_NAME
 */
export function countyFromFeature(
  props: Record<string, unknown> | null,
  geoLevel: GeoLevel,
): string {
  if (!props) return '';
  switch (geoLevel) {
    case 'county':
      return (props.name as string) ?? '';
    case 'tract':
      return CA_COUNTY_FIPS[props.COUNTYFP as string] ?? '';
    case 'zcta':
      return (props.COUNTY_NAME as string) ?? '';
  }
}

/**
 * Hover tracking key for a feature.
 * County: county name
 * Tract:  GEOID (unique per tract)
 * ZCTA:   ZCTA5CE20 (zip code)
 */
export function hoverKeyFromFeature(
  props: Record<string, unknown> | null,
  geoLevel: GeoLevel,
): string {
  if (!props) return '';
  switch (geoLevel) {
    case 'county':
      return (props.name as string) ?? '';
    case 'tract':
      return (props.GEOID as string) ?? '';
    case 'zcta':
      return (props.ZCTA5CE20 as string) ?? '';
  }
}
