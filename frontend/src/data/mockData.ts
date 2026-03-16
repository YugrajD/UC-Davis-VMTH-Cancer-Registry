import type { CountyData, RegionSummary, CalEnviroScreenData } from '../types';
import type { IncidenceRecord, BreedDetail } from '../api/client';

// --- Mock breed list and breed detail data ---
// 10 dog breeds from 002_lookup_tables.sql — each gets ~325 cases (uniform pick)
export const MOCK_BREEDS = [
  'Golden Retriever', 'Labrador Retriever', 'Boxer', 'German Shepherd',
  'Rottweiler', 'Bernese Mountain Dog', 'Beagle', 'Bulldog',
  'Poodle', 'Mixed Breed Dog',
];

// Real distribution weights from seed data:
// Sex: Male 15%, Female 15%, Neutered Male 35%, Spayed Female 35%
// Counties: Sacramento 22%, San Joaquin 10%, Contra Costa 10%, Placer 9%,
//   Solano 8%, Alameda 8%, Yolo 6%, Stanislaus 6%, Butte 4%, El Dorado 5%,
//   Sutter 3%, Nevada 3%, Yuba 2%, Glenn 2%, Colusa 1%, Amador 1%
// Cancer types (dogs): Lymphoma 25%, Mast Cell Tumor 22%, Hemangiosarcoma 15%,
//   Osteosarcoma 12%, Melanoma 10%, TCC 6%, SCC 5%, Fibrosarcoma 5%

// Each dog breed gets ~325 cases (3250 dogs / 10 breeds, uniform pick)
const BREED_DETAILS: Record<string, BreedDetail> = {
  'Golden Retriever': {
    breed: 'Golden Retriever',
    total_cases: 328,
    sex_breakdown: [{ sex: 'Neutered Male', count: 115 }, { sex: 'Spayed Female', count: 115 }, { sex: 'Male', count: 49 }, { sex: 'Female', count: 49 }],
    cancer_types: [
      { cancer_type: 'Lymphoma', count: 82 }, { cancer_type: 'Hemangiosarcoma', count: 62 },
      { cancer_type: 'Mast Cell Tumor', count: 56 }, { cancer_type: 'Osteosarcoma', count: 39 },
      { cancer_type: 'Melanoma', count: 33 }, { cancer_type: 'Transitional Cell Carcinoma', count: 20 },
      { cancer_type: 'Squamous Cell Carcinoma', count: 18 }, { cancer_type: 'Fibrosarcoma', count: 18 },
    ],
    county_cases: [
      { county_name: 'Sacramento', fips_code: '06067', count: 72 }, { county_name: 'San Joaquin', fips_code: '06077', count: 33 },
      { county_name: 'Contra Costa', fips_code: '06013', count: 33 }, { county_name: 'Placer', fips_code: '06061', count: 30 },
      { county_name: 'Solano', fips_code: '06095', count: 26 }, { county_name: 'Alameda', fips_code: '06001', count: 26 },
      { county_name: 'Yolo', fips_code: '06113', count: 20 }, { county_name: 'Stanislaus', fips_code: '06099', count: 20 },
      { county_name: 'El Dorado', fips_code: '06017', count: 16 }, { county_name: 'Butte', fips_code: '06007', count: 13 },
      { county_name: 'Sutter', fips_code: '06101', count: 10 }, { county_name: 'Nevada', fips_code: '06057', count: 10 },
      { county_name: 'Yuba', fips_code: '06115', count: 7 }, { county_name: 'Glenn', fips_code: '06021', count: 7 },
      { county_name: 'Colusa', fips_code: '06011', count: 3 }, { county_name: 'Amador', fips_code: '06005', count: 2 },
    ],
  },
  'Labrador Retriever': {
    breed: 'Labrador Retriever',
    total_cases: 321,
    sex_breakdown: [{ sex: 'Neutered Male', count: 112 }, { sex: 'Spayed Female', count: 112 }, { sex: 'Male', count: 48 }, { sex: 'Female', count: 49 }],
    cancer_types: [
      { cancer_type: 'Mast Cell Tumor', count: 74 }, { cancer_type: 'Lymphoma', count: 71 },
      { cancer_type: 'Hemangiosarcoma', count: 48 }, { cancer_type: 'Osteosarcoma', count: 39 },
      { cancer_type: 'Melanoma', count: 32 }, { cancer_type: 'Transitional Cell Carcinoma', count: 19 },
      { cancer_type: 'Squamous Cell Carcinoma', count: 19 }, { cancer_type: 'Fibrosarcoma', count: 19 },
    ],
    county_cases: [
      { county_name: 'Sacramento', fips_code: '06067', count: 71 }, { county_name: 'San Joaquin', fips_code: '06077', count: 32 },
      { county_name: 'Contra Costa', fips_code: '06013', count: 32 }, { county_name: 'Placer', fips_code: '06061', count: 29 },
      { county_name: 'Solano', fips_code: '06095', count: 26 }, { county_name: 'Alameda', fips_code: '06001', count: 26 },
      { county_name: 'Yolo', fips_code: '06113', count: 19 }, { county_name: 'Stanislaus', fips_code: '06099', count: 19 },
      { county_name: 'El Dorado', fips_code: '06017', count: 16 }, { county_name: 'Butte', fips_code: '06007', count: 13 },
      { county_name: 'Sutter', fips_code: '06101', count: 10 }, { county_name: 'Nevada', fips_code: '06057', count: 10 },
      { county_name: 'Yuba', fips_code: '06115', count: 6 }, { county_name: 'Glenn', fips_code: '06021', count: 6 },
      { county_name: 'Colusa', fips_code: '06011', count: 3 }, { county_name: 'Amador', fips_code: '06005', count: 3 },
    ],
  },
  'Boxer': {
    breed: 'Boxer',
    total_cases: 318,
    sex_breakdown: [{ sex: 'Neutered Male', count: 111 }, { sex: 'Spayed Female', count: 111 }, { sex: 'Male', count: 48 }, { sex: 'Female', count: 48 }],
    cancer_types: [
      { cancer_type: 'Mast Cell Tumor', count: 89 }, { cancer_type: 'Lymphoma', count: 64 },
      { cancer_type: 'Hemangiosarcoma', count: 48 }, { cancer_type: 'Melanoma', count: 32 },
      { cancer_type: 'Osteosarcoma', count: 32 }, { cancer_type: 'Squamous Cell Carcinoma', count: 19 },
      { cancer_type: 'Transitional Cell Carcinoma', count: 18 }, { cancer_type: 'Fibrosarcoma', count: 16 },
    ],
    county_cases: [
      { county_name: 'Sacramento', fips_code: '06067', count: 70 }, { county_name: 'San Joaquin', fips_code: '06077', count: 32 },
      { county_name: 'Contra Costa', fips_code: '06013', count: 32 }, { county_name: 'Placer', fips_code: '06061', count: 29 },
      { county_name: 'Solano', fips_code: '06095', count: 25 }, { county_name: 'Alameda', fips_code: '06001', count: 25 },
      { county_name: 'Yolo', fips_code: '06113', count: 19 }, { county_name: 'Stanislaus', fips_code: '06099', count: 19 },
      { county_name: 'El Dorado', fips_code: '06017', count: 16 }, { county_name: 'Butte', fips_code: '06007', count: 13 },
      { county_name: 'Sutter', fips_code: '06101', count: 10 }, { county_name: 'Nevada', fips_code: '06057', count: 10 },
      { county_name: 'Yuba', fips_code: '06115', count: 6 }, { county_name: 'Glenn', fips_code: '06021', count: 6 },
      { county_name: 'Colusa', fips_code: '06011', count: 3 }, { county_name: 'Amador', fips_code: '06005', count: 3 },
    ],
  },
};

// Generate a plausible breed detail for breeds without explicit data
// Uses real distribution weights from seed_mock_data.py
function generateBreedDetail(breed: string): BreedDetail {
  // Seeded from breed name for stability
  let h = 0;
  for (let i = 0; i < breed.length; i++) h = Math.imul(31, h) + breed.charCodeAt(i) | 0;
  const r = () => { h = Math.imul(h ^ (h >>> 16), 0x45d9f3b); h = (h ^ (h >>> 16)) >>> 0; return (h % 100) / 100; };

  // ~325 cases per breed (3250 dogs / 10 breeds), with some natural variation
  const total = 280 + Math.round(r() * 90);
  return {
    breed,
    total_cases: total,
    // Real sex distribution: ~15% Male, ~15% Female, ~35% Neutered Male, ~35% Spayed Female
    sex_breakdown: [
      { sex: 'Neutered Male', count: Math.round(total * 0.35) },
      { sex: 'Spayed Female', count: Math.round(total * 0.35) },
      { sex: 'Male', count: Math.round(total * 0.15) },
      { sex: 'Female', count: Math.round(total * 0.15) },
    ],
    cancer_types: [
      { cancer_type: 'Lymphoma', count: Math.round(total * (0.20 + r() * 0.10)) },
      { cancer_type: 'Mast Cell Tumor', count: Math.round(total * (0.17 + r() * 0.10)) },
      { cancer_type: 'Hemangiosarcoma', count: Math.round(total * (0.10 + r() * 0.10)) },
      { cancer_type: 'Osteosarcoma', count: Math.round(total * (0.08 + r() * 0.08)) },
      { cancer_type: 'Melanoma', count: Math.round(total * (0.06 + r() * 0.08)) },
      { cancer_type: 'Transitional Cell Carcinoma', count: Math.round(total * (0.03 + r() * 0.06)) },
      { cancer_type: 'Squamous Cell Carcinoma', count: Math.round(total * (0.02 + r() * 0.06)) },
      { cancer_type: 'Fibrosarcoma', count: Math.round(total * (0.02 + r() * 0.06)) },
    ].sort((a, b) => b.count - a.count),
    // Real county weights: Sacramento 22%, San Joaquin 10%, Contra Costa 10%, Placer 9%, etc.
    county_cases: [
      { county_name: 'Sacramento', fips_code: '06067', count: Math.round(total * (0.18 + r() * 0.08)) },
      { county_name: 'San Joaquin', fips_code: '06077', count: Math.round(total * (0.07 + r() * 0.06)) },
      { county_name: 'Contra Costa', fips_code: '06013', count: Math.round(total * (0.07 + r() * 0.06)) },
      { county_name: 'Placer', fips_code: '06061', count: Math.round(total * (0.06 + r() * 0.06)) },
      { county_name: 'Solano', fips_code: '06095', count: Math.round(total * (0.05 + r() * 0.06)) },
      { county_name: 'Alameda', fips_code: '06001', count: Math.round(total * (0.05 + r() * 0.06)) },
      { county_name: 'Yolo', fips_code: '06113', count: Math.round(total * (0.04 + r() * 0.04)) },
      { county_name: 'Stanislaus', fips_code: '06099', count: Math.round(total * (0.04 + r() * 0.04)) },
      { county_name: 'El Dorado', fips_code: '06017', count: Math.round(total * (0.03 + r() * 0.04)) },
      { county_name: 'Butte', fips_code: '06007', count: Math.round(total * (0.02 + r() * 0.04)) },
    ].filter(c => c.count > 0),
  };
}

export function getMockBreedDetail(breed: string): BreedDetail {
  return BREED_DETAILS[breed] || generateBreedDetail(breed);
}

// --- Static mock data for presentation views ---
// These values are taken from the design screenshots so the
// dashboard matches what you see in the presentation.

// County‑level case counts used by:
// - Regional Summary table
// - County Details table
// - California County Map
// County weights from seed_mock_data.py (~5000 total cases):
// Sacramento 22%, San Joaquin 10%, Contra Costa 10%, Placer 9%,
// Solano 8%, Alameda 8%, Yolo 6%, Stanislaus 6%, Butte 4%,
// El Dorado 5%, Sutter 3%, Nevada 3%, Yuba 2%, Glenn 2%, Colusa 1%, Amador 1%
export const MOCK_COUNTY_DATA: CountyData[] = [
  // Central Valley
  { county: 'Sacramento', region: 'Central Valley', count: 1100, fips: '' },
  { county: 'San Joaquin', region: 'Central Valley', count: 500, fips: '' },
  { county: 'Placer', region: 'Central Valley', count: 450, fips: '' },
  { county: 'Yolo', region: 'Central Valley', count: 300, fips: '' },
  { county: 'Stanislaus', region: 'Central Valley', count: 300, fips: '' },
  { county: 'El Dorado', region: 'Central Valley', count: 250, fips: '' },

  // Bay Area
  { county: 'Contra Costa', region: 'Bay Area', count: 500, fips: '' },
  { county: 'Solano', region: 'Bay Area', count: 400, fips: '' },
  { county: 'Alameda', region: 'Bay Area', count: 400, fips: '' },

  // Northern CA
  { county: 'Butte', region: 'Northern CA', count: 200, fips: '' },
  { county: 'Sutter', region: 'Northern CA', count: 150, fips: '' },
  { county: 'Nevada', region: 'Northern CA', count: 150, fips: '' },
  { county: 'Yuba', region: 'Northern CA', count: 100, fips: '' },
  { county: 'Glenn', region: 'Northern CA', count: 100, fips: '' },
  { county: 'Colusa', region: 'Northern CA', count: 50, fips: '' },
  { county: 'Amador', region: 'Northern CA', count: 50, fips: '' },
];

// Pre‑computed helpers derived from MOCK_COUNTY_DATA
export const MOCK_REGION_SUMMARY: RegionSummary = generateRegionSummary(MOCK_COUNTY_DATA);

export const MOCK_COUNT_RANGE = (() => {
  const counts = MOCK_COUNTY_DATA.map((c) => c.count).filter((n) => n > 0);
  if (counts.length === 0) {
    return { min: 0, max: 1 };
  }
  return {
    min: Math.min(...counts),
    max: Math.max(...counts),
  };
})();

// Cancer‑type distribution used by the Cancer Types tab.
// Based on real veterinary oncology data proportions (~5000 total):
// Lymphoma 25%, Mast Cell Tumor 22%, Hemangiosarcoma 15%, Osteosarcoma 12%,
// Melanoma 10%, TCC 6%, SCC 5%, Fibrosarcoma 5%
export const MOCK_CANCER_TYPE_INCIDENTS: IncidenceRecord[] = [
  { cancer_type: 'Lymphoma', count: 1250 },
  { cancer_type: 'Mast Cell Tumor', count: 1100 },
  { cancer_type: 'Hemangiosarcoma', count: 750 },
  { cancer_type: 'Osteosarcoma', count: 600 },
  { cancer_type: 'Melanoma', count: 500 },
  { cancer_type: 'Transitional Cell Carcinoma', count: 300 },
  { cancer_type: 'Squamous Cell Carcinoma', count: 250 },
  { cancer_type: 'Fibrosarcoma', count: 250 },
];

// Mock CalEnviroScreen 4.0 data (county-level percentiles)
// Based on real CalEnviroScreen 4.0 approximate county averages
export const MOCK_CALENVIROSCREEN_DATA: CalEnviroScreenData[] = [
  { county_id: 1, county_name: 'Alameda', county_fips: '06001', ces_score: 38.2, pollution_burden: 42.1, ozone: 28.5, pm25: 45.3, diesel_pm: 55.2, pesticides: 12.1, toxic_releases: 48.3, traffic: 62.4, drinking_water: 35.8, lead: 52.1, cleanup_sites: 45.6, groundwater_threats: 38.9, hazardous_waste: 42.7, solid_waste: 35.2, impaired_water: 28.4, pop_characteristics: 34.8, asthma: 42.5, low_birth_weight: 38.9, cardiovascular: 35.2, poverty: 28.6, unemployment: 32.1, housing_burden: 55.8, education: 22.4, linguistic_isolation: 35.7 },
  { county_id: 2, county_name: 'Butte', county_fips: '06007', ces_score: 42.8, pollution_burden: 38.5, ozone: 52.1, pm25: 55.8, diesel_pm: 22.3, pesticides: 35.6, toxic_releases: 18.9, traffic: 15.2, drinking_water: 48.3, lead: 28.7, cleanup_sites: 22.1, groundwater_threats: 35.4, hazardous_waste: 15.8, solid_waste: 42.3, impaired_water: 38.5, pop_characteristics: 48.2, asthma: 52.8, low_birth_weight: 42.1, cardiovascular: 55.3, poverty: 58.9, unemployment: 52.4, housing_burden: 48.7, education: 45.2, linguistic_isolation: 18.5 },
  { county_id: 3, county_name: 'Contra Costa', county_fips: '06013', ces_score: 35.6, pollution_burden: 45.8, ozone: 32.4, pm25: 42.1, diesel_pm: 58.9, pesticides: 8.5, toxic_releases: 62.3, traffic: 55.8, drinking_water: 28.9, lead: 48.5, cleanup_sites: 52.3, groundwater_threats: 45.2, hazardous_waste: 55.8, solid_waste: 28.9, impaired_water: 22.1, pop_characteristics: 28.5, asthma: 35.2, low_birth_weight: 32.8, cardiovascular: 28.9, poverty: 22.4, unemployment: 28.5, housing_burden: 48.2, education: 18.9, linguistic_isolation: 28.4 },
  { county_id: 4, county_name: 'Del Norte', county_fips: '06015', ces_score: 28.5, pollution_burden: 18.2, ozone: 8.5, pm25: 22.3, diesel_pm: 12.8, pesticides: 15.2, toxic_releases: 8.5, traffic: 5.2, drinking_water: 55.8, lead: 15.2, cleanup_sites: 8.9, groundwater_threats: 12.5, hazardous_waste: 5.8, solid_waste: 18.9, impaired_water: 42.5, pop_characteristics: 42.8, asthma: 35.2, low_birth_weight: 38.5, cardiovascular: 48.2, poverty: 65.8, unemployment: 62.1, housing_burden: 52.3, education: 58.9, linguistic_isolation: 12.5 },
  { county_id: 5, county_name: 'El Dorado', county_fips: '06017', ces_score: 15.8, pollution_burden: 22.5, ozone: 58.2, pm25: 32.1, diesel_pm: 8.5, pesticides: 18.9, toxic_releases: 5.2, traffic: 12.8, drinking_water: 22.5, lead: 18.2, cleanup_sites: 12.5, groundwater_threats: 15.8, hazardous_waste: 8.2, solid_waste: 15.5, impaired_water: 18.9, pop_characteristics: 12.5, asthma: 18.2, low_birth_weight: 15.8, cardiovascular: 18.5, poverty: 15.2, unemployment: 18.9, housing_burden: 28.5, education: 12.8, linguistic_isolation: 8.5 },
  { county_id: 6, county_name: 'Fresno', county_fips: '06019', ces_score: 72.5, pollution_burden: 75.8, ozone: 82.5, pm25: 85.2, diesel_pm: 62.8, pesticides: 88.5, toxic_releases: 55.2, traffic: 48.9, drinking_water: 78.5, lead: 42.8, cleanup_sites: 48.5, groundwater_threats: 65.2, hazardous_waste: 52.8, solid_waste: 58.5, impaired_water: 72.1, pop_characteristics: 68.9, asthma: 72.5, low_birth_weight: 65.8, cardiovascular: 62.5, poverty: 78.2, unemployment: 72.8, housing_burden: 68.5, education: 75.2, linguistic_isolation: 72.8 },
  { county_id: 7, county_name: 'Humboldt', county_fips: '06023', ces_score: 25.2, pollution_burden: 15.8, ozone: 5.2, pm25: 28.5, diesel_pm: 15.2, pesticides: 22.8, toxic_releases: 12.5, traffic: 8.2, drinking_water: 42.8, lead: 22.5, cleanup_sites: 15.8, groundwater_threats: 18.2, hazardous_waste: 12.5, solid_waste: 22.8, impaired_water: 35.2, pop_characteristics: 38.5, asthma: 32.8, low_birth_weight: 35.2, cardiovascular: 42.5, poverty: 55.2, unemployment: 48.5, housing_burden: 45.8, education: 42.1, linguistic_isolation: 15.2 },
  { county_id: 8, county_name: 'Kern', county_fips: '06029', ces_score: 75.8, pollution_burden: 78.2, ozone: 88.5, pm25: 82.8, diesel_pm: 68.5, pesticides: 85.2, toxic_releases: 72.8, traffic: 52.5, drinking_water: 82.1, lead: 38.5, cleanup_sites: 55.2, groundwater_threats: 72.8, hazardous_waste: 62.5, solid_waste: 65.8, impaired_water: 78.5, pop_characteristics: 72.1, asthma: 68.5, low_birth_weight: 62.8, cardiovascular: 58.5, poverty: 75.2, unemployment: 68.9, housing_burden: 62.5, education: 72.8, linguistic_isolation: 65.2 },
  { county_id: 9, county_name: 'Los Angeles', county_fips: '06037', ces_score: 55.8, pollution_burden: 62.5, ozone: 72.8, pm25: 68.5, diesel_pm: 75.2, pesticides: 15.8, toxic_releases: 65.2, traffic: 82.5, drinking_water: 52.8, lead: 62.1, cleanup_sites: 68.5, groundwater_threats: 58.2, hazardous_waste: 65.8, solid_waste: 55.2, impaired_water: 48.5, pop_characteristics: 52.8, asthma: 55.2, low_birth_weight: 48.5, cardiovascular: 45.2, poverty: 52.8, unemployment: 48.5, housing_burden: 72.8, education: 48.2, linguistic_isolation: 58.5 },
  { county_id: 10, county_name: 'Marin', county_fips: '06041', ces_score: 8.5, pollution_burden: 12.8, ozone: 18.5, pm25: 25.2, diesel_pm: 22.8, pesticides: 5.2, toxic_releases: 8.5, traffic: 28.9, drinking_water: 12.5, lead: 15.8, cleanup_sites: 18.2, groundwater_threats: 8.5, hazardous_waste: 12.8, solid_waste: 8.2, impaired_water: 15.5, pop_characteristics: 8.2, asthma: 12.5, low_birth_weight: 8.8, cardiovascular: 8.5, poverty: 8.2, unemployment: 8.5, housing_burden: 35.2, education: 5.8, linguistic_isolation: 15.2 },
  { county_id: 11, county_name: 'Mendocino', county_fips: '06045', ces_score: 28.9, pollution_burden: 18.5, ozone: 12.8, pm25: 32.5, diesel_pm: 12.2, pesticides: 35.8, toxic_releases: 8.5, traffic: 5.8, drinking_water: 48.2, lead: 18.5, cleanup_sites: 12.8, groundwater_threats: 15.2, hazardous_waste: 8.5, solid_waste: 22.8, impaired_water: 35.8, pop_characteristics: 42.5, asthma: 35.8, low_birth_weight: 38.2, cardiovascular: 45.8, poverty: 55.8, unemployment: 52.5, housing_burden: 48.2, education: 48.5, linguistic_isolation: 25.8 },
  { county_id: 12, county_name: 'Monterey', county_fips: '06053', ces_score: 52.8, pollution_burden: 55.2, ozone: 42.5, pm25: 38.2, diesel_pm: 42.8, pesticides: 82.5, toxic_releases: 35.8, traffic: 32.5, drinking_water: 62.8, lead: 28.5, cleanup_sites: 35.2, groundwater_threats: 48.5, hazardous_waste: 35.8, solid_waste: 42.5, impaired_water: 55.8, pop_characteristics: 52.1, asthma: 42.5, low_birth_weight: 45.8, cardiovascular: 38.5, poverty: 58.2, unemployment: 52.8, housing_burden: 55.2, education: 62.8, linguistic_isolation: 65.2 },
  { county_id: 13, county_name: 'Napa', county_fips: '06055', ces_score: 22.5, pollution_burden: 28.8, ozone: 35.2, pm25: 38.5, diesel_pm: 18.2, pesticides: 42.8, toxic_releases: 15.2, traffic: 22.5, drinking_water: 25.8, lead: 22.1, cleanup_sites: 18.5, groundwater_threats: 22.8, hazardous_waste: 15.2, solid_waste: 18.5, impaired_water: 22.8, pop_characteristics: 22.5, asthma: 18.2, low_birth_weight: 15.8, cardiovascular: 18.5, poverty: 22.8, unemployment: 18.5, housing_burden: 42.8, education: 28.5, linguistic_isolation: 32.8 },
  { county_id: 14, county_name: 'Nevada', county_fips: '06057', ces_score: 12.5, pollution_burden: 18.2, ozone: 52.8, pm25: 28.5, diesel_pm: 5.8, pesticides: 12.5, toxic_releases: 8.2, traffic: 8.5, drinking_water: 18.2, lead: 12.5, cleanup_sites: 8.8, groundwater_threats: 12.2, hazardous_waste: 5.5, solid_waste: 12.8, impaired_water: 15.2, pop_characteristics: 15.8, asthma: 18.5, low_birth_weight: 12.2, cardiovascular: 22.5, poverty: 22.8, unemployment: 18.2, housing_burden: 32.5, education: 15.8, linguistic_isolation: 5.8 },
  { county_id: 15, county_name: 'Orange', county_fips: '06059', ces_score: 32.5, pollution_burden: 42.8, ozone: 58.5, pm25: 52.2, diesel_pm: 48.5, pesticides: 8.2, toxic_releases: 38.5, traffic: 72.8, drinking_water: 32.5, lead: 45.8, cleanup_sites: 42.2, groundwater_threats: 38.5, hazardous_waste: 42.8, solid_waste: 32.5, impaired_water: 28.2, pop_characteristics: 28.5, asthma: 25.8, low_birth_weight: 22.5, cardiovascular: 22.8, poverty: 25.2, unemployment: 22.8, housing_burden: 58.5, education: 28.2, linguistic_isolation: 42.5 },
  { county_id: 16, county_name: 'Placer', county_fips: '06061', ces_score: 12.2, pollution_burden: 18.5, ozone: 55.2, pm25: 28.8, diesel_pm: 8.2, pesticides: 15.5, toxic_releases: 5.8, traffic: 15.2, drinking_water: 15.8, lead: 12.2, cleanup_sites: 8.5, groundwater_threats: 12.8, hazardous_waste: 5.2, solid_waste: 12.5, impaired_water: 15.8, pop_characteristics: 8.5, asthma: 12.8, low_birth_weight: 8.5, cardiovascular: 12.2, poverty: 8.8, unemployment: 8.5, housing_burden: 28.2, education: 8.5, linguistic_isolation: 8.2 },
  { county_id: 17, county_name: 'Riverside', county_fips: '06065', ces_score: 58.2, pollution_burden: 65.8, ozone: 82.1, pm25: 72.5, diesel_pm: 62.8, pesticides: 45.2, toxic_releases: 48.5, traffic: 58.2, drinking_water: 62.5, lead: 38.2, cleanup_sites: 42.8, groundwater_threats: 55.2, hazardous_waste: 48.5, solid_waste: 52.8, impaired_water: 58.5, pop_characteristics: 52.8, asthma: 48.5, low_birth_weight: 42.8, cardiovascular: 42.5, poverty: 52.2, unemployment: 48.8, housing_burden: 58.5, education: 55.2, linguistic_isolation: 48.5 },
  { county_id: 18, county_name: 'Sacramento', county_fips: '06067', ces_score: 45.2, pollution_burden: 48.5, ozone: 52.8, pm25: 55.2, diesel_pm: 52.5, pesticides: 22.8, toxic_releases: 42.5, traffic: 52.8, drinking_water: 42.5, lead: 48.2, cleanup_sites: 45.8, groundwater_threats: 42.1, hazardous_waste: 38.5, solid_waste: 42.8, impaired_water: 35.2, pop_characteristics: 42.8, asthma: 48.5, low_birth_weight: 42.2, cardiovascular: 45.8, poverty: 42.5, unemployment: 42.8, housing_burden: 52.5, education: 38.2, linguistic_isolation: 35.8 },
  { county_id: 19, county_name: 'San Bernardino', county_fips: '06071', ces_score: 62.5, pollution_burden: 68.2, ozone: 85.8, pm25: 78.5, diesel_pm: 72.2, pesticides: 28.5, toxic_releases: 58.2, traffic: 62.5, drinking_water: 65.8, lead: 42.5, cleanup_sites: 52.8, groundwater_threats: 62.5, hazardous_waste: 58.2, solid_waste: 55.8, impaired_water: 62.5, pop_characteristics: 58.2, asthma: 55.8, low_birth_weight: 48.5, cardiovascular: 48.2, poverty: 58.5, unemployment: 55.2, housing_burden: 58.8, education: 58.5, linguistic_isolation: 48.2 },
  { county_id: 20, county_name: 'San Diego', county_fips: '06073', ces_score: 35.8, pollution_burden: 42.5, ozone: 48.2, pm25: 42.8, diesel_pm: 45.2, pesticides: 18.5, toxic_releases: 35.8, traffic: 58.2, drinking_water: 38.5, lead: 35.2, cleanup_sites: 38.8, groundwater_threats: 32.5, hazardous_waste: 35.8, solid_waste: 32.2, impaired_water: 28.5, pop_characteristics: 32.8, asthma: 28.5, low_birth_weight: 25.8, cardiovascular: 25.2, poverty: 32.5, unemployment: 28.8, housing_burden: 55.2, education: 32.5, linguistic_isolation: 38.2 },
  { county_id: 21, county_name: 'San Francisco', county_fips: '06075', ces_score: 28.5, pollution_burden: 38.2, ozone: 15.8, pm25: 42.5, diesel_pm: 62.8, pesticides: 2.5, toxic_releases: 42.5, traffic: 75.2, drinking_water: 22.8, lead: 55.8, cleanup_sites: 58.2, groundwater_threats: 42.5, hazardous_waste: 48.2, solid_waste: 25.8, impaired_water: 18.5, pop_characteristics: 22.8, asthma: 28.5, low_birth_weight: 22.2, cardiovascular: 18.5, poverty: 18.2, unemployment: 18.8, housing_burden: 62.5, education: 12.8, linguistic_isolation: 32.5 },
  { county_id: 22, county_name: 'San Joaquin', county_fips: '06077', ces_score: 65.2, pollution_burden: 68.5, ozone: 72.8, pm25: 75.2, diesel_pm: 65.8, pesticides: 72.5, toxic_releases: 52.8, traffic: 48.5, drinking_water: 72.2, lead: 42.8, cleanup_sites: 48.5, groundwater_threats: 58.2, hazardous_waste: 48.5, solid_waste: 55.8, impaired_water: 68.5, pop_characteristics: 62.5, asthma: 58.2, low_birth_weight: 55.8, cardiovascular: 52.5, poverty: 65.2, unemployment: 62.8, housing_burden: 58.5, education: 62.2, linguistic_isolation: 55.8 },
  { county_id: 23, county_name: 'San Luis Obispo', county_fips: '06079', ces_score: 18.5, pollution_burden: 22.8, ozone: 32.5, pm25: 25.2, diesel_pm: 15.8, pesticides: 28.5, toxic_releases: 12.2, traffic: 15.8, drinking_water: 28.5, lead: 15.2, cleanup_sites: 18.8, groundwater_threats: 22.5, hazardous_waste: 15.8, solid_waste: 18.2, impaired_water: 22.5, pop_characteristics: 18.2, asthma: 15.8, low_birth_weight: 12.5, cardiovascular: 15.8, poverty: 22.5, unemployment: 18.2, housing_burden: 42.5, education: 18.8, linguistic_isolation: 12.5 },
  { county_id: 24, county_name: 'San Mateo', county_fips: '06081', ces_score: 18.2, pollution_burden: 25.8, ozone: 18.5, pm25: 32.2, diesel_pm: 35.8, pesticides: 5.2, toxic_releases: 22.5, traffic: 52.8, drinking_water: 18.2, lead: 28.5, cleanup_sites: 32.8, groundwater_threats: 22.5, hazardous_waste: 28.2, solid_waste: 15.8, impaired_water: 12.5, pop_characteristics: 15.2, asthma: 15.8, low_birth_weight: 12.5, cardiovascular: 12.2, poverty: 12.8, unemployment: 12.5, housing_burden: 48.2, education: 12.5, linguistic_isolation: 22.8 },
  { county_id: 25, county_name: 'Santa Barbara', county_fips: '06083', ces_score: 35.2, pollution_burden: 38.5, ozone: 42.8, pm25: 32.5, diesel_pm: 28.2, pesticides: 55.8, toxic_releases: 22.5, traffic: 28.8, drinking_water: 42.5, lead: 22.8, cleanup_sites: 25.2, groundwater_threats: 32.5, hazardous_waste: 22.8, solid_waste: 28.5, impaired_water: 35.2, pop_characteristics: 35.8, asthma: 28.5, low_birth_weight: 32.2, cardiovascular: 28.5, poverty: 38.2, unemployment: 32.8, housing_burden: 52.5, education: 42.8, linguistic_isolation: 45.2 },
  { county_id: 26, county_name: 'Santa Clara', county_fips: '06085', ces_score: 28.8, pollution_burden: 35.2, ozone: 32.5, pm25: 42.8, diesel_pm: 48.5, pesticides: 12.8, toxic_releases: 42.2, traffic: 65.8, drinking_water: 28.5, lead: 42.2, cleanup_sites: 52.5, groundwater_threats: 42.8, hazardous_waste: 48.5, solid_waste: 28.2, impaired_water: 22.8, pop_characteristics: 25.2, asthma: 22.8, low_birth_weight: 18.5, cardiovascular: 18.2, poverty: 18.5, unemployment: 15.8, housing_burden: 58.2, education: 15.2, linguistic_isolation: 32.2 },
  { county_id: 27, county_name: 'Santa Cruz', county_fips: '06087', ces_score: 22.8, pollution_burden: 25.2, ozone: 22.5, pm25: 22.8, diesel_pm: 18.5, pesticides: 35.2, toxic_releases: 12.8, traffic: 22.5, drinking_water: 32.8, lead: 18.5, cleanup_sites: 15.2, groundwater_threats: 22.8, hazardous_waste: 12.5, solid_waste: 18.8, impaired_water: 28.5, pop_characteristics: 22.5, asthma: 18.8, low_birth_weight: 15.2, cardiovascular: 15.8, poverty: 25.2, unemployment: 22.5, housing_burden: 52.8, education: 22.5, linguistic_isolation: 22.8 },
  { county_id: 28, county_name: 'Shasta', county_fips: '06089', ces_score: 35.5, pollution_burden: 28.2, ozone: 45.8, pm25: 48.5, diesel_pm: 15.2, pesticides: 18.5, toxic_releases: 12.8, traffic: 8.5, drinking_water: 42.2, lead: 22.8, cleanup_sites: 18.5, groundwater_threats: 22.2, hazardous_waste: 12.5, solid_waste: 28.8, impaired_water: 42.2, pop_characteristics: 42.5, asthma: 48.2, low_birth_weight: 42.5, cardiovascular: 52.8, poverty: 52.5, unemployment: 48.2, housing_burden: 42.5, education: 42.8, linguistic_isolation: 8.5 },
  { county_id: 29, county_name: 'Solano', county_fips: '06095', ces_score: 38.5, pollution_burden: 42.2, ozone: 38.5, pm25: 45.8, diesel_pm: 48.2, pesticides: 22.5, toxic_releases: 42.8, traffic: 42.5, drinking_water: 35.2, lead: 42.5, cleanup_sites: 38.8, groundwater_threats: 35.2, hazardous_waste: 38.5, solid_waste: 32.8, impaired_water: 28.5, pop_characteristics: 35.8, asthma: 38.5, low_birth_weight: 35.2, cardiovascular: 38.8, poverty: 32.5, unemployment: 32.8, housing_burden: 45.2, education: 32.5, linguistic_isolation: 28.2 },
  { county_id: 30, county_name: 'Sonoma', county_fips: '06097', ces_score: 18.8, pollution_burden: 22.5, ozone: 25.8, pm25: 28.2, diesel_pm: 15.5, pesticides: 35.2, toxic_releases: 12.5, traffic: 18.8, drinking_water: 22.5, lead: 18.2, cleanup_sites: 15.5, groundwater_threats: 18.8, hazardous_waste: 12.2, solid_waste: 15.5, impaired_water: 18.2, pop_characteristics: 18.5, asthma: 15.2, low_birth_weight: 12.8, cardiovascular: 15.5, poverty: 18.8, unemployment: 15.2, housing_burden: 42.5, education: 18.2, linguistic_isolation: 22.5 },
  { county_id: 31, county_name: 'Stanislaus', county_fips: '06099', ces_score: 62.8, pollution_burden: 65.5, ozone: 72.2, pm25: 72.8, diesel_pm: 55.2, pesticides: 68.5, toxic_releases: 42.8, traffic: 38.5, drinking_water: 68.2, lead: 38.5, cleanup_sites: 42.2, groundwater_threats: 55.8, hazardous_waste: 42.5, solid_waste: 52.8, impaired_water: 62.2, pop_characteristics: 58.5, asthma: 55.2, low_birth_weight: 52.8, cardiovascular: 52.5, poverty: 62.8, unemployment: 58.5, housing_burden: 55.2, education: 62.5, linguistic_isolation: 52.8 },
  { county_id: 32, county_name: 'Sutter', county_fips: '06101', ces_score: 52.5, pollution_burden: 55.8, ozone: 58.2, pm25: 62.5, diesel_pm: 42.8, pesticides: 72.5, toxic_releases: 28.5, traffic: 22.8, drinking_water: 58.5, lead: 32.8, cleanup_sites: 28.5, groundwater_threats: 42.2, hazardous_waste: 28.8, solid_waste: 42.5, impaired_water: 52.8, pop_characteristics: 52.2, asthma: 48.5, low_birth_weight: 48.2, cardiovascular: 48.5, poverty: 55.2, unemployment: 52.8, housing_burden: 48.5, education: 55.8, linguistic_isolation: 48.5 },
  { county_id: 33, county_name: 'Ventura', county_fips: '06111', ces_score: 32.8, pollution_burden: 38.5, ozone: 55.2, pm25: 42.5, diesel_pm: 35.8, pesticides: 48.2, toxic_releases: 28.5, traffic: 42.2, drinking_water: 32.5, lead: 28.8, cleanup_sites: 32.5, groundwater_threats: 28.2, hazardous_waste: 32.8, solid_waste: 25.5, impaired_water: 28.2, pop_characteristics: 28.5, asthma: 22.8, low_birth_weight: 22.5, cardiovascular: 22.2, poverty: 28.5, unemployment: 22.8, housing_burden: 52.2, education: 32.8, linguistic_isolation: 38.5 },
  { county_id: 34, county_name: 'Yolo', county_fips: '06113', ces_score: 35.8, pollution_burden: 42.2, ozone: 48.5, pm25: 52.8, diesel_pm: 35.2, pesticides: 55.8, toxic_releases: 22.5, traffic: 28.8, drinking_water: 38.5, lead: 28.2, cleanup_sites: 25.8, groundwater_threats: 32.5, hazardous_waste: 22.8, solid_waste: 32.5, impaired_water: 38.2, pop_characteristics: 32.8, asthma: 28.5, low_birth_weight: 28.2, cardiovascular: 28.8, poverty: 35.2, unemployment: 32.5, housing_burden: 48.8, education: 28.5, linguistic_isolation: 28.8 },
  { county_id: 35, county_name: 'Yuba', county_fips: '06115', ces_score: 55.2, pollution_burden: 52.8, ozone: 55.5, pm25: 58.2, diesel_pm: 35.8, pesticides: 62.5, toxic_releases: 22.8, traffic: 18.5, drinking_water: 58.2, lead: 32.5, cleanup_sites: 28.2, groundwater_threats: 42.5, hazardous_waste: 28.2, solid_waste: 42.5, impaired_water: 52.2, pop_characteristics: 58.5, asthma: 55.2, low_birth_weight: 52.5, cardiovascular: 55.8, poverty: 62.5, unemployment: 58.2, housing_burden: 52.8, education: 58.2, linguistic_isolation: 35.2 },
  { county_id: 36, county_name: 'Colusa', county_fips: '06011', ces_score: 55.8, pollution_burden: 58.2, ozone: 52.5, pm25: 58.8, diesel_pm: 28.5, pesticides: 85.2, toxic_releases: 18.5, traffic: 12.8, drinking_water: 65.2, lead: 22.5, cleanup_sites: 18.2, groundwater_threats: 35.8, hazardous_waste: 18.5, solid_waste: 35.2, impaired_water: 55.8, pop_characteristics: 55.2, asthma: 42.5, low_birth_weight: 48.2, cardiovascular: 48.5, poverty: 62.8, unemployment: 55.2, housing_burden: 48.5, education: 65.2, linguistic_isolation: 62.5 },
  { county_id: 37, county_name: 'Glenn', county_fips: '06021', ces_score: 52.2, pollution_burden: 55.5, ozone: 55.8, pm25: 58.5, diesel_pm: 25.2, pesticides: 82.8, toxic_releases: 15.8, traffic: 8.2, drinking_water: 62.5, lead: 18.8, cleanup_sites: 15.5, groundwater_threats: 32.8, hazardous_waste: 15.2, solid_waste: 32.5, impaired_water: 52.5, pop_characteristics: 52.8, asthma: 42.2, low_birth_weight: 45.5, cardiovascular: 48.8, poverty: 62.2, unemployment: 55.8, housing_burden: 45.2, education: 62.8, linguistic_isolation: 58.2 },
  { county_id: 38, county_name: 'Alpine', county_fips: '06003', ces_score: 5.2, pollution_burden: 8.5, ozone: 42.8, pm25: 15.2, diesel_pm: 2.5, pesticides: 2.8, toxic_releases: 2.5, traffic: 2.2, drinking_water: 12.5, lead: 5.8, cleanup_sites: 2.5, groundwater_threats: 5.2, hazardous_waste: 2.8, solid_waste: 5.5, impaired_water: 8.2, pop_characteristics: 18.5, asthma: 12.8, low_birth_weight: 15.2, cardiovascular: 18.5, poverty: 28.2, unemployment: 22.5, housing_burden: 25.8, education: 22.5, linguistic_isolation: 5.2 },
  { county_id: 39, county_name: 'Amador', county_fips: '06005', ces_score: 18.5, pollution_burden: 22.2, ozone: 55.8, pm25: 32.5, diesel_pm: 8.8, pesticides: 15.2, toxic_releases: 8.5, traffic: 8.2, drinking_water: 28.5, lead: 18.2, cleanup_sites: 15.5, groundwater_threats: 18.8, hazardous_waste: 12.2, solid_waste: 18.5, impaired_water: 22.8, pop_characteristics: 18.2, asthma: 22.5, low_birth_weight: 18.8, cardiovascular: 25.2, poverty: 22.5, unemployment: 22.8, housing_burden: 32.5, education: 18.2, linguistic_isolation: 5.8 },
  { county_id: 40, county_name: 'Calaveras', county_fips: '06009', ces_score: 15.2, pollution_burden: 18.8, ozone: 52.5, pm25: 28.2, diesel_pm: 5.5, pesticides: 12.8, toxic_releases: 5.2, traffic: 5.8, drinking_water: 25.2, lead: 15.5, cleanup_sites: 12.8, groundwater_threats: 15.5, hazardous_waste: 8.5, solid_waste: 15.2, impaired_water: 18.5, pop_characteristics: 15.8, asthma: 18.2, low_birth_weight: 15.5, cardiovascular: 22.8, poverty: 22.2, unemployment: 18.5, housing_burden: 28.8, education: 15.5, linguistic_isolation: 5.2 },
  { county_id: 41, county_name: 'Imperial', county_fips: '06025', ces_score: 82.5, pollution_burden: 78.8, ozone: 72.5, pm25: 88.2, diesel_pm: 58.5, pesticides: 92.8, toxic_releases: 42.5, traffic: 35.8, drinking_water: 85.2, lead: 32.5, cleanup_sites: 38.8, groundwater_threats: 72.5, hazardous_waste: 45.2, solid_waste: 62.8, impaired_water: 88.5, pop_characteristics: 82.2, asthma: 78.5, low_birth_weight: 72.8, cardiovascular: 68.5, poverty: 88.2, unemployment: 85.8, housing_burden: 72.5, education: 85.8, linguistic_isolation: 82.5 },
  { county_id: 42, county_name: 'Inyo', county_fips: '06027', ces_score: 22.8, pollution_burden: 25.5, ozone: 48.2, pm25: 28.8, diesel_pm: 8.5, pesticides: 8.2, toxic_releases: 12.5, traffic: 5.8, drinking_water: 35.2, lead: 12.8, cleanup_sites: 8.5, groundwater_threats: 15.2, hazardous_waste: 8.8, solid_waste: 12.5, impaired_water: 22.8, pop_characteristics: 25.5, asthma: 22.2, low_birth_weight: 18.5, cardiovascular: 28.8, poverty: 32.5, unemployment: 28.2, housing_burden: 28.5, education: 25.8, linguistic_isolation: 8.2 },
  { county_id: 43, county_name: 'Kings', county_fips: '06031', ces_score: 72.8, pollution_burden: 75.2, ozone: 78.5, pm25: 82.8, diesel_pm: 55.2, pesticides: 88.5, toxic_releases: 42.8, traffic: 32.5, drinking_water: 78.2, lead: 35.8, cleanup_sites: 42.5, groundwater_threats: 62.8, hazardous_waste: 45.5, solid_waste: 58.2, impaired_water: 72.5, pop_characteristics: 68.8, asthma: 65.2, low_birth_weight: 62.5, cardiovascular: 58.8, poverty: 75.2, unemployment: 72.5, housing_burden: 62.8, education: 72.5, linguistic_isolation: 68.2 },
  { county_id: 44, county_name: 'Lake', county_fips: '06033', ces_score: 48.5, pollution_burden: 35.2, ozone: 38.8, pm25: 42.5, diesel_pm: 12.8, pesticides: 28.5, toxic_releases: 15.2, traffic: 8.5, drinking_water: 55.8, lead: 25.2, cleanup_sites: 35.8, groundwater_threats: 42.5, hazardous_waste: 28.2, solid_waste: 35.5, impaired_water: 52.8, pop_characteristics: 58.5, asthma: 55.2, low_birth_weight: 52.8, cardiovascular: 58.5, poverty: 72.8, unemployment: 68.5, housing_burden: 55.2, education: 62.5, linguistic_isolation: 18.5 },
  { county_id: 45, county_name: 'Lassen', county_fips: '06035', ces_score: 32.5, pollution_burden: 28.8, ozone: 45.2, pm25: 38.5, diesel_pm: 12.2, pesticides: 18.5, toxic_releases: 8.8, traffic: 5.5, drinking_water: 42.8, lead: 15.2, cleanup_sites: 12.5, groundwater_threats: 18.8, hazardous_waste: 8.5, solid_waste: 22.2, impaired_water: 35.5, pop_characteristics: 42.8, asthma: 38.5, low_birth_weight: 35.2, cardiovascular: 42.8, poverty: 48.5, unemployment: 42.2, housing_burden: 38.5, education: 45.8, linguistic_isolation: 8.5 },
  { county_id: 46, county_name: 'Madera', county_fips: '06039', ces_score: 68.2, pollution_burden: 72.5, ozone: 78.8, pm25: 78.5, diesel_pm: 48.2, pesticides: 82.5, toxic_releases: 35.8, traffic: 28.5, drinking_water: 72.8, lead: 32.5, cleanup_sites: 38.2, groundwater_threats: 55.8, hazardous_waste: 38.5, solid_waste: 52.2, impaired_water: 65.8, pop_characteristics: 65.5, asthma: 62.8, low_birth_weight: 58.5, cardiovascular: 55.2, poverty: 72.5, unemployment: 68.8, housing_burden: 62.5, education: 72.2, linguistic_isolation: 68.5 },
  { county_id: 47, county_name: 'Mariposa', county_fips: '06043', ces_score: 12.8, pollution_burden: 15.5, ozone: 48.2, pm25: 25.8, diesel_pm: 5.2, pesticides: 8.5, toxic_releases: 5.8, traffic: 5.2, drinking_water: 22.5, lead: 12.8, cleanup_sites: 8.2, groundwater_threats: 12.5, hazardous_waste: 5.2, solid_waste: 12.8, impaired_water: 18.5, pop_characteristics: 22.8, asthma: 18.5, low_birth_weight: 18.2, cardiovascular: 25.5, poverty: 32.8, unemployment: 28.5, housing_burden: 32.2, education: 28.5, linguistic_isolation: 5.5 },
  { county_id: 48, county_name: 'Merced', county_fips: '06047', ces_score: 72.2, pollution_burden: 72.8, ozone: 75.5, pm25: 78.2, diesel_pm: 58.5, pesticides: 85.8, toxic_releases: 45.2, traffic: 35.8, drinking_water: 75.5, lead: 38.2, cleanup_sites: 42.8, groundwater_threats: 62.5, hazardous_waste: 48.2, solid_waste: 55.5, impaired_water: 72.8, pop_characteristics: 68.2, asthma: 65.5, low_birth_weight: 62.2, cardiovascular: 58.8, poverty: 78.5, unemployment: 75.2, housing_burden: 62.8, education: 75.5, linguistic_isolation: 68.8 },
  { county_id: 49, county_name: 'Modoc', county_fips: '06049', ces_score: 18.2, pollution_burden: 12.5, ozone: 35.8, pm25: 28.2, diesel_pm: 5.5, pesticides: 22.8, toxic_releases: 5.2, traffic: 2.8, drinking_water: 38.5, lead: 8.2, cleanup_sites: 5.5, groundwater_threats: 8.8, hazardous_waste: 5.2, solid_waste: 15.5, impaired_water: 28.2, pop_characteristics: 35.8, asthma: 28.5, low_birth_weight: 28.2, cardiovascular: 38.5, poverty: 52.2, unemployment: 45.8, housing_burden: 35.2, education: 48.5, linguistic_isolation: 5.8 },
  { county_id: 50, county_name: 'Mono', county_fips: '06051', ces_score: 8.8, pollution_burden: 12.2, ozone: 45.5, pm25: 18.8, diesel_pm: 5.2, pesticides: 5.5, toxic_releases: 2.8, traffic: 5.5, drinking_water: 15.8, lead: 8.5, cleanup_sites: 5.2, groundwater_threats: 8.5, hazardous_waste: 2.5, solid_waste: 8.8, impaired_water: 12.5, pop_characteristics: 15.2, asthma: 12.5, low_birth_weight: 12.8, cardiovascular: 15.5, poverty: 18.8, unemployment: 15.2, housing_burden: 32.5, education: 15.8, linguistic_isolation: 12.8 },
  { county_id: 51, county_name: 'Plumas', county_fips: '06063', ces_score: 15.5, pollution_burden: 18.8, ozone: 48.5, pm25: 32.8, diesel_pm: 5.8, pesticides: 12.2, toxic_releases: 5.5, traffic: 5.2, drinking_water: 28.8, lead: 12.5, cleanup_sites: 8.8, groundwater_threats: 12.2, hazardous_waste: 5.5, solid_waste: 15.8, impaired_water: 22.5, pop_characteristics: 22.2, asthma: 22.8, low_birth_weight: 18.5, cardiovascular: 28.2, poverty: 32.5, unemployment: 28.8, housing_burden: 32.2, education: 28.8, linguistic_isolation: 5.5 },
  { county_id: 52, county_name: 'San Benito', county_fips: '06069', ces_score: 42.5, pollution_burden: 45.8, ozone: 42.2, pm25: 35.5, diesel_pm: 28.8, pesticides: 68.5, toxic_releases: 22.8, traffic: 22.5, drinking_water: 48.2, lead: 22.5, cleanup_sites: 25.8, groundwater_threats: 35.2, hazardous_waste: 22.5, solid_waste: 32.8, impaired_water: 42.5, pop_characteristics: 42.2, asthma: 35.5, low_birth_weight: 38.8, cardiovascular: 32.5, poverty: 42.8, unemployment: 38.5, housing_burden: 48.2, education: 45.5, linguistic_isolation: 48.8 },
  { county_id: 53, county_name: 'Sierra', county_fips: '06091', ces_score: 5.8, pollution_burden: 8.2, ozone: 42.5, pm25: 22.8, diesel_pm: 2.8, pesticides: 5.5, toxic_releases: 2.2, traffic: 2.5, drinking_water: 18.5, lead: 8.8, cleanup_sites: 5.5, groundwater_threats: 8.2, hazardous_waste: 2.5, solid_waste: 8.5, impaired_water: 12.8, pop_characteristics: 18.2, asthma: 15.5, low_birth_weight: 15.8, cardiovascular: 22.2, poverty: 28.5, unemployment: 25.2, housing_burden: 25.5, education: 22.8, linguistic_isolation: 2.8 },
  { county_id: 54, county_name: 'Siskiyou', county_fips: '06093', ces_score: 25.5, pollution_burden: 18.8, ozone: 28.2, pm25: 35.5, diesel_pm: 8.8, pesticides: 18.2, toxic_releases: 12.5, traffic: 5.8, drinking_water: 42.5, lead: 15.8, cleanup_sites: 18.2, groundwater_threats: 22.5, hazardous_waste: 12.8, solid_waste: 22.5, impaired_water: 38.2, pop_characteristics: 42.5, asthma: 38.8, low_birth_weight: 35.5, cardiovascular: 48.2, poverty: 58.5, unemployment: 52.8, housing_burden: 42.5, education: 48.8, linguistic_isolation: 8.2 },
  { county_id: 55, county_name: 'Tehama', county_fips: '06103', ces_score: 48.8, pollution_burden: 42.5, ozone: 52.2, pm25: 55.5, diesel_pm: 22.8, pesticides: 45.2, toxic_releases: 15.5, traffic: 8.8, drinking_water: 52.5, lead: 22.2, cleanup_sites: 18.8, groundwater_threats: 28.5, hazardous_waste: 15.5, solid_waste: 32.8, impaired_water: 45.2, pop_characteristics: 55.8, asthma: 52.5, low_birth_weight: 48.8, cardiovascular: 55.5, poverty: 65.2, unemployment: 58.8, housing_burden: 48.5, education: 58.2, linguistic_isolation: 22.8 },
  { county_id: 56, county_name: 'Trinity', county_fips: '06105', ces_score: 18.8, pollution_burden: 12.2, ozone: 15.5, pm25: 32.8, diesel_pm: 5.5, pesticides: 12.8, toxic_releases: 5.8, traffic: 2.5, drinking_water: 38.2, lead: 12.5, cleanup_sites: 8.8, groundwater_threats: 12.2, hazardous_waste: 5.5, solid_waste: 18.2, impaired_water: 32.5, pop_characteristics: 38.8, asthma: 32.5, low_birth_weight: 32.2, cardiovascular: 42.5, poverty: 55.5, unemployment: 48.8, housing_burden: 38.5, education: 48.2, linguistic_isolation: 5.5 },
  { county_id: 57, county_name: 'Tulare', county_fips: '06107', ces_score: 75.5, pollution_burden: 78.8, ozone: 85.2, pm25: 85.5, diesel_pm: 62.2, pesticides: 90.5, toxic_releases: 48.8, traffic: 38.5, drinking_water: 82.8, lead: 35.5, cleanup_sites: 45.2, groundwater_threats: 68.8, hazardous_waste: 52.5, solid_waste: 58.8, impaired_water: 75.2, pop_characteristics: 72.5, asthma: 68.8, low_birth_weight: 65.5, cardiovascular: 62.2, poverty: 82.5, unemployment: 78.8, housing_burden: 65.5, education: 78.2, linguistic_isolation: 75.5 },
  { county_id: 58, county_name: 'Tuolumne', county_fips: '06109', ces_score: 18.2, pollution_burden: 22.5, ozone: 52.8, pm25: 32.2, diesel_pm: 8.5, pesticides: 12.8, toxic_releases: 8.2, traffic: 5.5, drinking_water: 28.2, lead: 15.5, cleanup_sites: 12.2, groundwater_threats: 15.5, hazardous_waste: 8.8, solid_waste: 15.2, impaired_water: 22.8, pop_characteristics: 22.5, asthma: 22.2, low_birth_weight: 18.8, cardiovascular: 25.5, poverty: 28.8, unemployment: 25.5, housing_burden: 32.8, education: 25.2, linguistic_isolation: 5.8 },
];

// Generate hierarchical summary for the summary table
export function generateRegionSummary(countyData: CountyData[]): RegionSummary {
  const regionMap = new Map<string, CountyData[]>();
  
  countyData.forEach(county => {
    const existing = regionMap.get(county.region);
    if (existing) {
      existing.push(county);
    } else {
      regionMap.set(county.region, [county]);
    }
  });
  
  // Calculate totals
  const totalCount = countyData.reduce((sum, c) => sum + c.count, 0);
  const totalPop = countyData.reduce((sum, c) => sum + (c.population ?? 0), 0);
  
  // Catchment area (Northern CA + Bay Area for this example)
  const catchmentRegions = ['Bay Area', 'Northern CA', 'Central Valley'];
  const catchmentCounties = countyData.filter(c => catchmentRegions.includes(c.region));
  const catchmentCount = catchmentCounties.reduce((sum, c) => sum + c.count, 0);
  const catchmentPop = catchmentCounties.reduce((sum, c) => sum + (c.population ?? 0), 0);
  
  const regions: RegionSummary[] = Array.from(regionMap.entries()).map(([regionName, counties]) => {
    const regionCount = counties.reduce((sum, c) => sum + c.count, 0);
    const regionPop = counties.reduce((sum, c) => sum + (c.population ?? 0), 0);
    
    return {
      name: regionName,
      type: 'region' as const,
      count: regionCount,
      population: regionPop,
      rate: regionPop > 0 ? Math.round((regionCount / regionPop) * 10000 * 10) / 10 : 0,
      children: counties.map(c => ({
        name: c.county,
        type: 'county' as const,
        count: c.count,
        population: c.population,
        rate: c.rate,
      })),
    };
  });
  
  return {
    name: 'California',
    type: 'state',
    count: totalCount,
    population: totalPop,
    rate: totalPop > 0 ? Math.round((totalCount / totalPop) * 10000 * 10) / 10 : 0,
    children: [
      {
        name: 'UC Davis Catchment Area',
        type: 'catchment',
        count: catchmentCount,
        population: catchmentPop,
        rate: catchmentPop > 0 ? Math.round((catchmentCount / catchmentPop) * 10000 * 10) / 10 : 0,
        children: regions.filter(r => catchmentRegions.includes(r.name)),
      },
      ...regions.filter(r => !catchmentRegions.includes(r.name)),
    ],
  };
}
