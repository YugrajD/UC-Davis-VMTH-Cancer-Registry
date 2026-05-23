import { describe, expect, it } from 'vitest';

import {
  CALIFORNIA_REGION_BY_COUNTY,
  isUcDavisCatchmentRegion,
  regionForCounty,
} from '../data/californiaRegions';

describe('californiaRegions', () => {
  it('covers every California county', () => {
    expect(Object.keys(CALIFORNIA_REGION_BY_COUNTY).length).toBe(58);
  });

  it('maps common dashboard counties to the shared region taxonomy', () => {
    expect(regionForCounty('Alameda')).toBe('San Francisco Bay Area');
    expect(regionForCounty('Sacramento')).toBe('Sacramento Valley');
    expect(regionForCounty('San Joaquin')).toBe('San Joaquin Valley');
    expect(regionForCounty('Placer')).toBe('Sierra Nevada');
    expect(regionForCounty('Los Angeles')).toBe('Greater Los Angeles');
    expect(regionForCounty('San Diego')).toBe('San Diego-Imperial');
  });

  it('accepts county names with a County suffix', () => {
    expect(regionForCounty('Alameda County')).toBe('San Francisco Bay Area');
  });

  it('falls back safely for non-California or unknown county names', () => {
    expect(regionForCounty('Washoe')).toBe('Other California');
  });

  it('marks the Northern and Central California regions as UC Davis catchment regions', () => {
    expect(isUcDavisCatchmentRegion('San Francisco Bay Area')).toBe(true);
    expect(isUcDavisCatchmentRegion('Sacramento Valley')).toBe(true);
    expect(isUcDavisCatchmentRegion('San Joaquin Valley')).toBe(true);
    expect(isUcDavisCatchmentRegion('Greater Los Angeles')).toBe(false);
  });
});
