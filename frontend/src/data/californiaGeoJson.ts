// Simplified California county boundaries for the choropleth map
// This is a simplified version - for production use, you'd want the full TopoJSON from US Census

export const CALIFORNIA_GEOJSON = {
  type: 'FeatureCollection' as const,
  features: [
    // Bay Area
    { type: 'Feature', properties: { name: 'Alameda', fips: '06001' }, geometry: { type: 'Polygon', coordinates: [[[-122.37, 37.9], [-122.12, 37.9], [-121.72, 37.63], [-121.87, 37.46], [-122.15, 37.51], [-122.37, 37.68], [-122.37, 37.9]]] } },
    { type: 'Feature', properties: { name: 'Contra Costa', fips: '06013' }, geometry: { type: 'Polygon', coordinates: [[[-122.42, 38.15], [-121.87, 38.05], [-121.72, 37.82], [-121.72, 37.63], [-122.12, 37.9], [-122.37, 37.9], [-122.42, 38.15]]] } },
    { type: 'Feature', properties: { name: 'Marin', fips: '06041' }, geometry: { type: 'Polygon', coordinates: [[[-123.02, 38.32], [-122.73, 38.32], [-122.53, 38.0], [-122.5, 37.88], [-122.67, 37.82], [-122.87, 37.95], [-123.02, 38.15], [-123.02, 38.32]]] } },
    { type: 'Feature', properties: { name: 'San Francisco', fips: '06075' }, geometry: { type: 'Polygon', coordinates: [[[-122.52, 37.82], [-122.37, 37.82], [-122.37, 37.68], [-122.52, 37.71], [-122.52, 37.82]]] } },
    { type: 'Feature', properties: { name: 'San Mateo', fips: '06081' }, geometry: { type: 'Polygon', coordinates: [[[-122.52, 37.71], [-122.37, 37.68], [-122.15, 37.51], [-122.17, 37.28], [-122.47, 37.22], [-122.52, 37.35], [-122.52, 37.71]]] } },
    { type: 'Feature', properties: { name: 'Santa Clara', fips: '06085' }, geometry: { type: 'Polygon', coordinates: [[[-122.15, 37.51], [-121.87, 37.46], [-121.47, 37.18], [-121.4, 36.88], [-121.82, 36.95], [-122.17, 37.28], [-122.15, 37.51]]] } },
    { type: 'Feature', properties: { name: 'Sonoma', fips: '06097' }, geometry: { type: 'Polygon', coordinates: [[[-123.53, 38.77], [-122.9, 38.9], [-122.48, 38.37], [-122.53, 38.0], [-122.73, 38.32], [-123.02, 38.32], [-123.35, 38.52], [-123.53, 38.77]]] } },
    { type: 'Feature', properties: { name: 'Napa', fips: '06055' }, geometry: { type: 'Polygon', coordinates: [[[-122.9, 38.9], [-122.4, 38.86], [-122.35, 38.45], [-122.48, 38.37], [-122.9, 38.9]]] } },
    
    // Northern CA
    { type: 'Feature', properties: { name: 'Butte', fips: '06007' }, geometry: { type: 'Polygon', coordinates: [[[-122.1, 40.0], [-121.35, 40.15], [-121.08, 39.54], [-121.63, 39.36], [-122.03, 39.67], [-122.1, 40.0]]] } },
    { type: 'Feature', properties: { name: 'Shasta', fips: '06089' }, geometry: { type: 'Polygon', coordinates: [[[-123.07, 41.2], [-121.5, 41.18], [-121.35, 40.45], [-121.35, 40.15], [-122.1, 40.0], [-122.75, 40.35], [-123.07, 40.7], [-123.07, 41.2]]] } },
    { type: 'Feature', properties: { name: 'Humboldt', fips: '06023' }, geometry: { type: 'Polygon', coordinates: [[[-124.41, 41.47], [-123.77, 41.47], [-123.44, 40.95], [-123.54, 40.13], [-123.91, 40.0], [-124.35, 40.32], [-124.41, 41.47]]] } },
    { type: 'Feature', properties: { name: 'Mendocino', fips: '06045' }, geometry: { type: 'Polygon', coordinates: [[[-123.91, 40.0], [-123.07, 40.0], [-122.9, 39.58], [-122.9, 38.9], [-123.53, 38.77], [-123.82, 38.95], [-123.91, 40.0]]] } },
    { type: 'Feature', properties: { name: 'Del Norte', fips: '06015' }, geometry: { type: 'Polygon', coordinates: [[[-124.23, 42.0], [-123.52, 42.0], [-123.52, 41.47], [-124.23, 41.47], [-124.23, 42.0]]] } },
    
    // Central Valley
    { type: 'Feature', properties: { name: 'Sacramento', fips: '06067' }, geometry: { type: 'Polygon', coordinates: [[[-121.87, 38.93], [-121.14, 38.93], [-121.03, 38.5], [-121.14, 38.07], [-121.57, 38.05], [-121.87, 38.5], [-121.87, 38.93]]] } },
    { type: 'Feature', properties: { name: 'San Joaquin', fips: '06077' }, geometry: { type: 'Polygon', coordinates: [[[-121.57, 38.05], [-121.14, 38.07], [-120.93, 37.63], [-120.93, 37.32], [-121.47, 37.48], [-121.72, 37.63], [-121.57, 38.05]]] } },
    { type: 'Feature', properties: { name: 'Fresno', fips: '06019' }, geometry: { type: 'Polygon', coordinates: [[[-120.9, 37.32], [-119.3, 37.63], [-118.75, 36.45], [-119.57, 35.78], [-120.32, 36.14], [-120.9, 36.63], [-120.9, 37.32]]] } },
    { type: 'Feature', properties: { name: 'Stanislaus', fips: '06099' }, geometry: { type: 'Polygon', coordinates: [[[-121.47, 37.48], [-120.93, 37.32], [-120.65, 37.16], [-120.97, 37.16], [-121.22, 37.66], [-121.47, 37.48]]] } },
    { type: 'Feature', properties: { name: 'Kern', fips: '06029' }, geometry: { type: 'Polygon', coordinates: [[[-120.2, 35.78], [-118.0, 35.78], [-117.63, 34.82], [-118.88, 34.82], [-119.47, 35.1], [-120.2, 35.33], [-120.2, 35.78]]] } },
    
    // Central Coast
    { type: 'Feature', properties: { name: 'Monterey', fips: '06053' }, geometry: { type: 'Polygon', coordinates: [[[-122.15, 36.88], [-121.24, 36.97], [-120.2, 36.49], [-120.6, 35.78], [-121.42, 35.8], [-121.97, 36.24], [-122.15, 36.52], [-122.15, 36.88]]] } },
    { type: 'Feature', properties: { name: 'Santa Cruz', fips: '06087' }, geometry: { type: 'Polygon', coordinates: [[[-122.32, 37.18], [-121.82, 36.95], [-121.58, 36.88], [-121.97, 36.88], [-122.15, 36.88], [-122.32, 37.0], [-122.32, 37.18]]] } },
    { type: 'Feature', properties: { name: 'San Luis Obispo', fips: '06079' }, geometry: { type: 'Polygon', coordinates: [[[-121.42, 35.8], [-120.2, 35.78], [-119.67, 35.27], [-120.2, 34.97], [-121.0, 35.15], [-121.33, 35.53], [-121.42, 35.8]]] } },
    { type: 'Feature', properties: { name: 'Santa Barbara', fips: '06083' }, geometry: { type: 'Polygon', coordinates: [[[-121.0, 35.15], [-119.47, 35.1], [-119.22, 34.58], [-119.67, 34.42], [-120.65, 34.57], [-120.65, 35.0], [-121.0, 35.15]]] } },
    
    // Southern CA
    { type: 'Feature', properties: { name: 'Los Angeles', fips: '06037' }, geometry: { type: 'Polygon', coordinates: [[[-119.22, 34.58], [-117.67, 34.82], [-117.67, 33.87], [-118.05, 33.77], [-118.6, 33.77], [-118.95, 34.07], [-119.22, 34.58]]] } },
    { type: 'Feature', properties: { name: 'Orange', fips: '06059' }, geometry: { type: 'Polygon', coordinates: [[[-118.05, 33.77], [-117.51, 33.87], [-117.41, 33.43], [-117.78, 33.38], [-118.05, 33.47], [-118.05, 33.77]]] } },
    { type: 'Feature', properties: { name: 'San Diego', fips: '06073' }, geometry: { type: 'Polygon', coordinates: [[[-117.51, 33.87], [-116.1, 33.43], [-116.1, 32.62], [-117.12, 32.53], [-117.28, 32.8], [-117.41, 33.43], [-117.51, 33.87]]] } },
    { type: 'Feature', properties: { name: 'Riverside', fips: '06065' }, geometry: { type: 'Polygon', coordinates: [[[-117.67, 34.04], [-114.63, 34.87], [-114.63, 33.43], [-116.1, 33.43], [-117.51, 33.87], [-117.67, 34.04]]] } },
    { type: 'Feature', properties: { name: 'San Bernardino', fips: '06071' }, geometry: { type: 'Polygon', coordinates: [[[-117.67, 34.82], [-115.65, 35.8], [-114.63, 35.1], [-114.63, 34.87], [-117.67, 34.04], [-117.67, 34.82]]] } },
    { type: 'Feature', properties: { name: 'Ventura', fips: '06111' }, geometry: { type: 'Polygon', coordinates: [[[-119.67, 34.42], [-118.88, 34.82], [-118.95, 34.07], [-119.22, 34.07], [-119.48, 34.12], [-119.67, 34.42]]] } },
  ],
};

export type CaliforniaGeoJson = typeof CALIFORNIA_GEOJSON;
