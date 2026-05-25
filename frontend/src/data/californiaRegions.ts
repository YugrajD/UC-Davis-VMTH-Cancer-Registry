export const CALIFORNIA_REGION_BY_COUNTY: Record<string, string> = {
  Alameda: 'San Francisco Bay Area',
  Alpine: 'Sierra Nevada',
  Amador: 'Sierra Nevada',
  Butte: 'Sacramento Valley',
  Calaveras: 'Sierra Nevada',
  Colusa: 'Sacramento Valley',
  'Contra Costa': 'San Francisco Bay Area',
  'Del Norte': 'North Coast',
  'El Dorado': 'Sierra Nevada',
  Fresno: 'San Joaquin Valley',
  Glenn: 'Sacramento Valley',
  Humboldt: 'North Coast',
  Imperial: 'San Diego-Imperial',
  Inyo: 'Sierra Nevada',
  Kern: 'San Joaquin Valley',
  Kings: 'San Joaquin Valley',
  Lake: 'North Coast',
  Lassen: 'Shasta Cascade',
  'Los Angeles': 'Greater Los Angeles',
  Madera: 'San Joaquin Valley',
  Marin: 'San Francisco Bay Area',
  Mariposa: 'Sierra Nevada',
  Mendocino: 'North Coast',
  Merced: 'San Joaquin Valley',
  Modoc: 'Shasta Cascade',
  Mono: 'Sierra Nevada',
  Monterey: 'Central Coast',
  Napa: 'San Francisco Bay Area',
  Nevada: 'Sierra Nevada',
  Orange: 'Greater Los Angeles',
  Placer: 'Sierra Nevada',
  Plumas: 'Shasta Cascade',
  Riverside: 'Inland Empire',
  Sacramento: 'Sacramento Valley',
  'San Benito': 'Central Coast',
  'San Bernardino': 'Inland Empire',
  'San Diego': 'San Diego-Imperial',
  'San Francisco': 'San Francisco Bay Area',
  'San Joaquin': 'San Joaquin Valley',
  'San Luis Obispo': 'Central Coast',
  'San Mateo': 'San Francisco Bay Area',
  'Santa Barbara': 'Central Coast',
  'Santa Clara': 'San Francisco Bay Area',
  'Santa Cruz': 'Central Coast',
  Shasta: 'Shasta Cascade',
  Sierra: 'Sierra Nevada',
  Siskiyou: 'Shasta Cascade',
  Solano: 'San Francisco Bay Area',
  Sonoma: 'San Francisco Bay Area',
  Stanislaus: 'San Joaquin Valley',
  Sutter: 'Sacramento Valley',
  Tehama: 'Shasta Cascade',
  Trinity: 'Shasta Cascade',
  Tulare: 'San Joaquin Valley',
  Tuolumne: 'Sierra Nevada',
  Ventura: 'Central Coast',
  Yolo: 'Sacramento Valley',
  Yuba: 'Sacramento Valley',
};

export const UC_DAVIS_CATCHMENT_REGIONS = [
  'San Francisco Bay Area',
  'Sacramento Valley',
  'San Joaquin Valley',
  'Sierra Nevada',
  'North Coast',
  'Shasta Cascade',
];

export function normalizeCountyName(county: string): string {
  return county.trim().replace(/\s+County$/i, '');
}

export function regionForCounty(county: string): string {
  return CALIFORNIA_REGION_BY_COUNTY[normalizeCountyName(county)] ?? 'Other California';
}

export function isUcDavisCatchmentRegion(region: string): boolean {
  return UC_DAVIS_CATCHMENT_REGIONS.includes(region);
}
