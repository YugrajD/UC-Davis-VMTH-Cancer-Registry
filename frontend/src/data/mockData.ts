import type { CancerRecord, CountyData, RegionSummary } from '../types';

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
      existing.population += record.population;
    } else {
      countyMap.set(record.county, {
        count: record.count,
        population: record.population,
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
  const totalPop = countyData.reduce((sum, c) => sum + c.population, 0);
  
  // Catchment area (Northern CA + Bay Area for this example)
  const catchmentRegions = ['Bay Area', 'Northern CA', 'Central Valley'];
  const catchmentCounties = countyData.filter(c => catchmentRegions.includes(c.region));
  const catchmentCount = catchmentCounties.reduce((sum, c) => sum + c.count, 0);
  const catchmentPop = catchmentCounties.reduce((sum, c) => sum + c.population, 0);
  
  const regions: RegionSummary[] = Array.from(regionMap.entries()).map(([regionName, counties]) => {
    const regionCount = counties.reduce((sum, c) => sum + c.count, 0);
    const regionPop = counties.reduce((sum, c) => sum + c.population, 0);
    
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
