import type { CancerRecord, CountyData, RegionSummary, CalEnviroScreenData } from '../types';
import type { IncidenceRecord } from '../api/client';

// California counties with their regions and FIPS codes
export const CALIFORNIA_COUNTIES: { name: string; region: string; fips: string; population: number }[] = [
  // Bay Area
  { name: 'Alameda', region: 'Bay Area', fips: '06001', population: 145000 },
  { name: 'Contra Costa', region: 'Bay Area', fips: '06013', population: 98000 },
  { name: 'Marin', region: 'Bay Area', fips: '06041', population: 32000 },
  { name: 'San Francisco', region: 'Bay Area', fips: '06075', population: 78000 },
  { name: 'San Mateo', region: 'Bay Area', fips: '06081', population: 68000 },
  { name: 'Santa Clara', region: 'Bay Area', fips: '06085', population: 165000 },
  { name: 'Sonoma', region: 'Bay Area', fips: '06097', population: 52000 },
  { name: 'Napa', region: 'Bay Area', fips: '06055', population: 18000 },
  
  // Northern CA
  { name: 'Butte', region: 'Northern CA', fips: '06007', population: 22000 },
  { name: 'Shasta', region: 'Northern CA', fips: '06089', population: 19000 },
  { name: 'Humboldt', region: 'Northern CA', fips: '06023', population: 14000 },
  { name: 'Mendocino', region: 'Northern CA', fips: '06045', population: 9500 },
  { name: 'Del Norte', region: 'Northern CA', fips: '06015', population: 3200 },
  
  // Central Valley
  { name: 'Sacramento', region: 'Central Valley', fips: '06067', population: 135000 },
  { name: 'San Joaquin', region: 'Central Valley', fips: '06077', population: 62000 },
  { name: 'Fresno', region: 'Central Valley', fips: '06019', population: 85000 },
  { name: 'Stanislaus', region: 'Central Valley', fips: '06099', population: 48000 },
  { name: 'Kern', region: 'Central Valley', fips: '06029', population: 72000 },
  
  // Central Coast
  { name: 'Monterey', region: 'Central Coast', fips: '06053', population: 38000 },
  { name: 'Santa Cruz', region: 'Central Coast', fips: '06087', population: 28000 },
  { name: 'San Luis Obispo', region: 'Central Coast', fips: '06079', population: 31000 },
  { name: 'Santa Barbara', region: 'Central Coast', fips: '06083', population: 42000 },
  
  // Southern CA
  { name: 'Los Angeles', region: 'Southern CA', fips: '06037', population: 890000 },
  { name: 'Orange', region: 'Southern CA', fips: '06059', population: 285000 },
  { name: 'San Diego', region: 'Southern CA', fips: '06073', population: 295000 },
  { name: 'Riverside', region: 'Southern CA', fips: '06065', population: 195000 },
  { name: 'San Bernardino', region: 'Southern CA', fips: '06071', population: 175000 },
  { name: 'Ventura', region: 'Southern CA', fips: '06111', population: 82000 },
];

// --- Static mock data for presentation views ---
// These values are taken from the design screenshots so the
// dashboard matches what you see in the presentation.

// County‑level case counts used by:
// - Regional Summary table
// - County Details table
// - California County Map
export const MOCK_COUNTY_DATA: CountyData[] = [
  // Central Valley
  { county: 'Yolo', region: 'Central Valley', count: 56, fips: '' },
  { county: 'Sacramento', region: 'Central Valley', count: 52, fips: '' },
  { county: 'Placer', region: 'Central Valley', count: 24, fips: '' },
  { county: 'El Dorado', region: 'Central Valley', count: 12, fips: '' },
  { county: 'San Joaquin', region: 'Central Valley', count: 11, fips: '' },

  // Bay Area
  { county: 'Solano', region: 'Bay Area', count: 28, fips: '' },
  { county: 'Contra Costa', region: 'Bay Area', count: 25, fips: '' },
  { county: 'Alameda', region: 'Bay Area', count: 19, fips: '' },

  // Northern CA
  { county: 'Butte', region: 'Northern CA', count: 17, fips: '' },
  { county: 'Sutter', region: 'Northern CA', count: 10, fips: '' },

  // Additional low‑count counties so totals match the summary
  { county: 'Shasta', region: 'Northern CA', count: 4, fips: '' },
  { county: 'Nevada', region: 'Northern CA', count: 3, fips: '' },
  { county: 'Yuba', region: 'Northern CA', count: 3, fips: '' },
  { county: 'Glenn', region: 'Northern CA', count: 2, fips: '' },
  { county: 'Colusa', region: 'Northern CA', count: 2, fips: '' },
  { county: 'Marin', region: 'Bay Area', count: 1, fips: '' },
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
// Matches the values shown in the design screenshot.
export const MOCK_CANCER_TYPE_INCIDENTS: IncidenceRecord[] = [
  { cancer_type: 'Osseous and chondromatous neoplasms', count: 414 },
  { cancer_type: 'Myelodysplastic syndromes', count: 246 },
  { cancer_type: 'Adenomas and adenocarcinomas', count: 223 },
  { cancer_type: 'Blood vessel tumors', count: 171 },
  { cancer_type: 'Gliomas', count: 157 },
  { cancer_type: 'Complex mixed and stromal neoplasms', count: 135 },
  { cancer_type: 'Adnexal and skin appendage neoplasms', count: 102 },
  { cancer_type: 'Ductal and lobular neoplasms', count: 86 },
  { cancer_type: 'Neoplasms, NOS', count: 68 },
  { cancer_type: 'Mature T- and NK-cell lymphomas', count: 67 },
];

// Generate mock cancer records
function generateMockRecords(): CancerRecord[] {
  const records: CancerRecord[] = [];
  const cancerTypes = ['Lymphoma', 'Osteosarcoma', 'Mast Cell Tumor', 'Hemangiosarcoma', 'Melanoma', 'Transitional Cell Carcinoma', 'Soft Tissue Sarcoma', 'Mammary Carcinoma'];
  const breeds = ['Golden Retriever', 'Labrador Retriever', 'Boxer', 'German Shepherd', 'Rottweiler', 'Bernese Mountain Dog', 'Beagle', 'French Bulldog', 'Poodle', 'Mixed Breed'];
  const sexes: ('male_intact' | 'male_neutered' | 'female_intact' | 'female_spayed')[] = ['male_intact', 'male_neutered', 'female_intact', 'female_spayed'];
  
  // Cancer type base rates (per 10,000) - vary by type
  const cancerBaseRates: Record<string, number> = {
    'Lymphoma': 45,
    'Osteosarcoma': 28,
    'Mast Cell Tumor': 52,
    'Hemangiosarcoma': 35,
    'Melanoma': 22,
    'Transitional Cell Carcinoma': 15,
    'Soft Tissue Sarcoma': 32,
    'Mammary Carcinoma': 48,
  };
  
  // Breed risk multipliers
  const breedRiskMultipliers: Record<string, Record<string, number>> = {
    'Golden Retriever': { 'Lymphoma': 1.8, 'Hemangiosarcoma': 2.2, default: 1.3 },
    'Labrador Retriever': { 'Lymphoma': 1.5, 'Mast Cell Tumor': 1.4, default: 1.1 },
    'Boxer': { 'Mast Cell Tumor': 2.5, 'Lymphoma': 1.6, default: 1.4 },
    'German Shepherd': { 'Hemangiosarcoma': 1.8, default: 1.2 },
    'Rottweiler': { 'Osteosarcoma': 2.8, default: 1.3 },
    'Bernese Mountain Dog': { 'Histiocytic Sarcoma': 3.0, 'Lymphoma': 1.9, default: 1.5 },
    'Beagle': { default: 0.8 },
    'French Bulldog': { default: 0.9 },
    'Poodle': { default: 0.85 },
    'Mixed Breed': { default: 0.7 },
  };
  
  CALIFORNIA_COUNTIES.forEach(county => {
    cancerTypes.forEach(cancerType => {
      breeds.forEach(breed => {
        sexes.forEach(sex => {
          // Calculate rate with some randomness
          const baseRate = cancerBaseRates[cancerType] || 30;
          const breedMultiplier = breedRiskMultipliers[breed]?.[cancerType] || breedRiskMultipliers[breed]?.default || 1;
          
          // Regional variation
          const regionMultiplier = county.region === 'Bay Area' ? 1.1 : 
                                   county.region === 'Southern CA' ? 1.05 : 
                                   county.region === 'Northern CA' ? 0.95 : 1;
          
          // Sex-based variation
          const sexMultiplier = cancerType === 'Mammary Carcinoma' 
            ? (sex.startsWith('female') ? (sex === 'female_intact' ? 7 : 0.5) : 0.01)
            : (sex.includes('neutered') || sex.includes('spayed') ? 1.15 : 1);
          
          const rate = baseRate * breedMultiplier * regionMultiplier * sexMultiplier * (0.8 + Math.random() * 0.4);
          
          // Population for this specific demographic
          const breedPop = Math.round(county.population * (breed === 'Mixed Breed' ? 0.35 : 0.065) / 4);
          const count = Math.round(breedPop * rate / 10000);
          
          if (count > 0) {
            records.push({
              county: county.name,
              region: county.region,
              cancerType,
              breed,
              sex,
              count,
              population: breedPop,
              rate: Math.round(rate * 10) / 10,
              year: 2024,
            });
          }
        });
      });
    });
  });
  
  return records;
}

export const MOCK_RECORDS = generateMockRecords();

// Aggregate data for county-level display
export function aggregateByCounty(
  records: CancerRecord[],
  filters: { cancerType?: string; breed?: string; sex?: string }
): CountyData[] {
  let filtered = records;
  
  if (filters.cancerType && filters.cancerType !== 'All Types') {
    filtered = filtered.filter(r => r.cancerType === filters.cancerType);
  }
  if (filters.breed && filters.breed !== 'All Breeds') {
    filtered = filtered.filter(r => r.breed === filters.breed);
  }
  if (filters.sex && filters.sex !== 'all') {
    filtered = filtered.filter(r => r.sex === filters.sex);
  }
  
  const countyMap = new Map<string, { count: number; population: number; region: string; fips: string }>();
  
  filtered.forEach(record => {
    const existing = countyMap.get(record.county);
    const countyInfo = CALIFORNIA_COUNTIES.find(c => c.name === record.county);
    
    if (existing) {
      existing.count += record.count;
      existing.population += record.population ?? 0;
    } else {
      countyMap.set(record.county, {
        count: record.count,
        population: record.population ?? 0,
        region: record.region,
        fips: countyInfo?.fips || '',
      });
    }
  });
  
  return Array.from(countyMap.entries()).map(([county, data]) => ({
    county,
    region: data.region,
    count: data.count,
    population: data.population,
    rate: data.population > 0 ? Math.round((data.count / data.population) * 10000 * 10) / 10 : 0,
    fips: data.fips,
  }));
}

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
