// Mock California pesticide use data by county.
// Real data source: CDPR Pesticide Use Reporting (PUR) database via
// https://trackingcalifornia.org/data-and-tools/pesticide-mapping-tool
// Values represent average annual lbs of active ingredient applied per sq mile (2015–2019).
// Replace with real CDPR PUR aggregate when integrating live data.

export interface CountyPesticideData {
  county: string;
  lbs_per_sq_mile: number;    // avg annual lbs of active ingredient / sq mile
  lbs_applied_total: number;  // avg annual lbs total applied in county
  top_pesticide_class: string;
}

// Agricultural intensity drives pesticide use — Central Valley counties are highest,
// Bay Area and foothill counties are much lower.
export const MOCK_PESTICIDE_DATA: CountyPesticideData[] = [
  // Central Valley — heaviest agricultural use
  { county: 'San Joaquin',  lbs_per_sq_mile: 312, lbs_applied_total: 14_060_000, top_pesticide_class: 'Fumigants' },
  { county: 'Stanislaus',   lbs_per_sq_mile: 284, lbs_applied_total: 10_782_000, top_pesticide_class: 'Insecticides' },
  { county: 'Colusa',       lbs_per_sq_mile: 248, lbs_applied_total:  2_728_000, top_pesticide_class: 'Herbicides' },
  { county: 'Yolo',         lbs_per_sq_mile: 196, lbs_applied_total:  3_920_000, top_pesticide_class: 'Herbicides' },
  { county: 'Sacramento',   lbs_per_sq_mile: 142, lbs_applied_total:  7_526_000, top_pesticide_class: 'Fungicides' },
  { county: 'Placer',       lbs_per_sq_mile:  84, lbs_applied_total:  2_520_000, top_pesticide_class: 'Herbicides' },
  { county: 'El Dorado',    lbs_per_sq_mile:  38, lbs_applied_total:  1_596_000, top_pesticide_class: 'Herbicides' },

  // Bay Area — lower agricultural use, some urban/suburban pesticide applications
  { county: 'Solano',       lbs_per_sq_mile: 118, lbs_applied_total:  2_478_000, top_pesticide_class: 'Herbicides' },
  { county: 'Contra Costa', lbs_per_sq_mile:  52, lbs_applied_total:  1_092_000, top_pesticide_class: 'Insecticides' },
  { county: 'Alameda',      lbs_per_sq_mile:  28, lbs_applied_total:    504_000, top_pesticide_class: 'Insecticides' },

  // Northern CA
  { county: 'Glenn',        lbs_per_sq_mile: 214, lbs_applied_total:  2_996_000, top_pesticide_class: 'Herbicides' },
  { county: 'Sutter',       lbs_per_sq_mile: 192, lbs_applied_total:  1_920_000, top_pesticide_class: 'Fumigants' },
  { county: 'Yuba',         lbs_per_sq_mile: 138, lbs_applied_total:    966_000, top_pesticide_class: 'Herbicides' },
  { county: 'Butte',        lbs_per_sq_mile: 106, lbs_applied_total:  4_452_000, top_pesticide_class: 'Herbicides' },
  { county: 'Nevada',       lbs_per_sq_mile:  44, lbs_applied_total:  1_320_000, top_pesticide_class: 'Herbicides' },
  { county: 'Amador',       lbs_per_sq_mile:  32, lbs_applied_total:    384_000, top_pesticide_class: 'Herbicides' },
];

export const PESTICIDE_BY_COUNTY: Record<string, CountyPesticideData> = Object.fromEntries(
  MOCK_PESTICIDE_DATA.map(d => [d.county, d])
);
