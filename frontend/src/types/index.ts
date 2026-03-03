// Type definitions
export type Sex = 'all' | 'male_intact' | 'male_neutered' | 'female_intact' | 'female_spayed';

export type RateType = 'incidence' | 'mortality';

export type TabType = 'overview' | 'breed-disparities' | 'cancer-types' | 'analysis' | 'data-upload';

export interface CancerRecord {
  county: string;
  region: string;
  cancerType: string;
  breed: string;
  sex: Sex;
  count: number;
  year: number;
}

export interface CountyData {
  county: string;
  region: string;
  count: number;
  fips: string;
}

export interface RegionSummary {
  name: string;
  type: 'state' | 'catchment' | 'region' | 'county';
  count: number;
  children?: RegionSummary[];
}

export interface FilterState {
  rateType: RateType;
  sex: Sex;
  cancerType: string;
  breed: string;
}

export interface Tab {
  id: TabType;
  label: string;
}

// Constants
export const TABS: Tab[] = [
  { id: 'overview', label: 'Overview' },
  { id: 'breed-disparities', label: 'Breed Disparities' },
  { id: 'cancer-types', label: 'Cancer Types' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'data-upload', label: 'Data Upload' },
] as const;

export interface CalEnviroScreenData {
  county_id: number;
  county_name: string;
  county_fips: string;
  ces_score: number | null;
  pollution_burden: number | null;
  ozone: number | null;
  pm25: number | null;
  diesel_pm: number | null;
  pesticides: number | null;
  toxic_releases: number | null;
  traffic: number | null;
  drinking_water: number | null;
  lead: number | null;
  cleanup_sites: number | null;
  groundwater_threats: number | null;
  hazardous_waste: number | null;
  solid_waste: number | null;
  impaired_water: number | null;
  pop_characteristics: number | null;
  asthma: number | null;
  low_birth_weight: number | null;
  cardiovascular: number | null;
  poverty: number | null;
  unemployment: number | null;
  housing_burden: number | null;
  education: number | null;
  linguistic_isolation: number | null;
}

export type CESIndicator = keyof Omit<CalEnviroScreenData, 'county_id' | 'county_name' | 'county_fips'>;

export const CES_INDICATORS: { value: CESIndicator; label: string }[] = [
  { value: 'ces_score', label: 'Overall CES Score' },
  { value: 'pollution_burden', label: 'Pollution Burden' },
  { value: 'ozone', label: 'Ozone' },
  { value: 'pm25', label: 'PM 2.5' },
  { value: 'diesel_pm', label: 'Diesel PM' },
  { value: 'pesticides', label: 'Pesticides' },
  { value: 'toxic_releases', label: 'Toxic Releases' },
  { value: 'traffic', label: 'Traffic' },
  { value: 'drinking_water', label: 'Drinking Water' },
  { value: 'lead', label: 'Lead' },
  { value: 'cleanup_sites', label: 'Cleanup Sites' },
  { value: 'groundwater_threats', label: 'Groundwater Threats' },
  { value: 'hazardous_waste', label: 'Hazardous Waste' },
  { value: 'solid_waste', label: 'Solid Waste' },
  { value: 'impaired_water', label: 'Impaired Water Bodies' },
  { value: 'pop_characteristics', label: 'Pop. Characteristics' },
  { value: 'asthma', label: 'Asthma' },
  { value: 'low_birth_weight', label: 'Low Birth Weight' },
  { value: 'cardiovascular', label: 'Cardiovascular Disease' },
  { value: 'poverty', label: 'Poverty' },
  { value: 'unemployment', label: 'Unemployment' },
  { value: 'housing_burden', label: 'Housing Burden' },
  { value: 'education', label: 'Education' },
  { value: 'linguistic_isolation', label: 'Linguistic Isolation' },
];

export const CANCER_TYPES: string[] = [
  'All Types',
  'Lymphoma',
  'Osteosarcoma',
  'Mast Cell Tumor',
  'Hemangiosarcoma',
  'Melanoma',
  'Transitional Cell Carcinoma',
  'Soft Tissue Sarcoma',
  'Mammary Carcinoma',
];

export const BREEDS: string[] = [
  'All Breeds',
  'Golden Retriever',
  'Labrador Retriever',
  'Boxer',
  'German Shepherd',
  'Rottweiler',
  'Bernese Mountain Dog',
  'Beagle',
  'French Bulldog',
  'Poodle',
  'Mixed Breed',
];

export const SEX_OPTIONS: { value: Sex; label: string }[] = [
  { value: 'all', label: 'All' },
  { value: 'male_intact', label: 'Male Intact' },
  { value: 'male_neutered', label: 'Male Neutered' },
  { value: 'female_intact', label: 'Female Intact' },
  { value: 'female_spayed', label: 'Female Spayed' },
];

export const RATE_OPTIONS: { value: RateType; label: string }[] = [
  { value: 'incidence', label: 'Incidence' },
  { value: 'mortality', label: 'Mortality' },
];
