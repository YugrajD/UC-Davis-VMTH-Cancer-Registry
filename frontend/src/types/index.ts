// Type definitions
export type Sex = 'all' | 'male_intact' | 'male_neutered' | 'female_intact' | 'female_spayed';

export type RateType = 'incidence' | 'mortality';

export type TabType = 'overview' | 'breed-disparities' | 'cancer-types' | 'regional-comparison';

export interface CancerRecord {
  county: string;
  region: string;
  cancerType: string;
  breed: string;
  sex: Sex;
  count: number;
  population: number;
  rate: number; // per 10,000
  year: number;
}

export interface CountyData {
  county: string;
  region: string;
  count: number;
  population: number;
  rate: number;
  fips: string;
}

export interface RegionSummary {
  name: string;
  type: 'state' | 'catchment' | 'region' | 'county';
  count: number;
  population: number;
  rate: number;
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
  { id: 'regional-comparison', label: 'Regional Comparison' },
] as const;

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
