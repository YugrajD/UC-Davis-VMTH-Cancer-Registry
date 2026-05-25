// Mock California pesticide use data by county.
// Real data source: CDPR Pesticide Use Reporting (PUR) database via
// https://trackingcalifornia.org/data-and-tools/pesticide-mapping-tool
// Values represent average annual lbs of active ingredient applied per sq mile (2015–2019).
// Replace with real CDPR PUR aggregate when integrating live data.

export type PesticideClass = 'fumigants' | 'herbicides' | 'insecticides' | 'fungicides';

export const PESTICIDE_CLASSES: { value: PesticideClass; label: string }[] = [
  { value: 'fumigants', label: 'Fumigants' },
  { value: 'herbicides', label: 'Herbicides' },
  { value: 'insecticides', label: 'Insecticides' },
  { value: 'fungicides', label: 'Fungicides' },
];

export interface ActiveIngredient {
  name: string;
  lbs_applied: number;
  category: PesticideClass;
}

export interface CountyPesticideData {
  county: string;
  lbs_per_sq_mile: number;    // avg annual lbs of active ingredient / sq mile
  lbs_applied_total: number;  // avg annual lbs total applied in county
  top_pesticide_class: string;
  by_class: Record<PesticideClass, number>; // lbs/sq mi per class
  top_ingredients: ActiveIngredient[];      // top 5, sorted by lbs_applied desc
  by_year: Record<number, number>;          // { 2015: N, 2016: N, ..., 2019: N } in lbs/sq mi
}

// Helper: distribute aggregate lbs/sq mi across 4 classes based on which is the
// top class for that county. The top class gets ~45%, next two ~20% each, last ~15%.
function distributeByClass(total: number, top: string): Record<PesticideClass, number> {
  const classes: PesticideClass[] = ['fumigants', 'herbicides', 'insecticides', 'fungicides'];
  const topKey = top.toLowerCase() as PesticideClass;
  const result = {} as Record<PesticideClass, number>;
  for (const cls of classes) {
    if (cls === topKey) {
      result[cls] = Math.round(total * 0.45);
    } else {
      // Remaining 55% split among other 3 — slight variation per class
      const share = cls === classes[(classes.indexOf(topKey) + 1) % 4] ? 0.22
        : cls === classes[(classes.indexOf(topKey) + 2) % 4] ? 0.18
        : 0.15;
      result[cls] = Math.round(total * share);
    }
  }
  return result;
}

// Ingredient pools by top pesticide class
const INGREDIENT_POOLS: Record<PesticideClass, { name: string; category: PesticideClass }[]> = {
  fumigants: [
    { name: '1,3-Dichloropropene', category: 'fumigants' },
    { name: 'Metam-sodium', category: 'fumigants' },
    { name: 'Chloropicrin', category: 'fumigants' },
    { name: 'Glyphosate', category: 'herbicides' },
    { name: 'Chlorpyrifos', category: 'insecticides' },
  ],
  herbicides: [
    { name: 'Glyphosate', category: 'herbicides' },
    { name: '2,4-D', category: 'herbicides' },
    { name: 'Paraquat dichloride', category: 'herbicides' },
    { name: 'Pendimethalin', category: 'herbicides' },
    { name: 'Sulfur', category: 'fungicides' },
  ],
  insecticides: [
    { name: 'Chlorpyrifos', category: 'insecticides' },
    { name: 'Permethrin', category: 'insecticides' },
    { name: 'Malathion', category: 'insecticides' },
    { name: 'Glyphosate', category: 'herbicides' },
    { name: 'Sulfur', category: 'fungicides' },
  ],
  fungicides: [
    { name: 'Sulfur', category: 'fungicides' },
    { name: 'Copper hydroxide', category: 'fungicides' },
    { name: 'Mancozeb', category: 'fungicides' },
    { name: 'Chlorpyrifos', category: 'insecticides' },
    { name: 'Glyphosate', category: 'herbicides' },
  ],
};

function generateIngredients(totalLbs: number, topClass: string): ActiveIngredient[] {
  const topKey = topClass.toLowerCase() as PesticideClass;
  const pool = INGREDIENT_POOLS[topKey];
  // Distribute total lbs across 5 ingredients: ~35%, ~22%, ~18%, ~14%, ~11%
  const shares = [0.35, 0.22, 0.18, 0.14, 0.11];
  return pool.map((ing, i) => ({
    name: ing.name,
    lbs_applied: Math.round(totalLbs * shares[i]),
    category: ing.category,
  }));
}

// Central Valley counties get a slight upward trend; others are roughly flat
const CENTRAL_VALLEY = new Set([
  'San Joaquin', 'Stanislaus', 'Colusa', 'Yolo', 'Sacramento',
  'Glenn', 'Sutter', 'Yuba', 'Butte',
]);

function generateByYear(avgLbsSqMi: number, county: string): Record<number, number> {
  const isCV = CENTRAL_VALLEY.has(county);
  // For CV counties: slight upward trend (~-6% to +8% relative to avg)
  // For others: roughly flat with small variation
  const offsets = isCV
    ? { 2015: -0.06, 2016: -0.03, 2017: 0.0, 2018: 0.04, 2019: 0.08 }
    : { 2015: -0.02, 2016: 0.01, 2017: 0.0, 2018: -0.01, 2019: 0.02 };
  const result: Record<number, number> = {};
  for (const [yr, pct] of Object.entries(offsets)) {
    result[Number(yr)] = Math.round(avgLbsSqMi * (1 + pct));
  }
  return result;
}

// Agricultural intensity drives pesticide use — Central Valley counties are highest,
// Bay Area and foothill counties are much lower.
export const MOCK_PESTICIDE_DATA: CountyPesticideData[] = [
  // Central Valley — heaviest agricultural use
  { county: 'San Joaquin',  lbs_per_sq_mile: 312, lbs_applied_total: 14_060_000, top_pesticide_class: 'Fumigants',    by_class: distributeByClass(312, 'Fumigants'),    top_ingredients: generateIngredients(14_060_000, 'Fumigants'),    by_year: generateByYear(312, 'San Joaquin') },
  { county: 'Stanislaus',   lbs_per_sq_mile: 284, lbs_applied_total: 10_782_000, top_pesticide_class: 'Insecticides', by_class: distributeByClass(284, 'Insecticides'), top_ingredients: generateIngredients(10_782_000, 'Insecticides'), by_year: generateByYear(284, 'Stanislaus') },
  { county: 'Colusa',       lbs_per_sq_mile: 248, lbs_applied_total:  2_728_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(248, 'Herbicides'),   top_ingredients: generateIngredients(2_728_000, 'Herbicides'),    by_year: generateByYear(248, 'Colusa') },
  { county: 'Yolo',         lbs_per_sq_mile: 196, lbs_applied_total:  3_920_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(196, 'Herbicides'),   top_ingredients: generateIngredients(3_920_000, 'Herbicides'),    by_year: generateByYear(196, 'Yolo') },
  { county: 'Sacramento',   lbs_per_sq_mile: 142, lbs_applied_total:  7_526_000, top_pesticide_class: 'Fungicides',   by_class: distributeByClass(142, 'Fungicides'),   top_ingredients: generateIngredients(7_526_000, 'Fungicides'),    by_year: generateByYear(142, 'Sacramento') },
  { county: 'Placer',       lbs_per_sq_mile:  84, lbs_applied_total:  2_520_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(84, 'Herbicides'),    top_ingredients: generateIngredients(2_520_000, 'Herbicides'),    by_year: generateByYear(84, 'Placer') },
  { county: 'El Dorado',    lbs_per_sq_mile:  38, lbs_applied_total:  1_596_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(38, 'Herbicides'),    top_ingredients: generateIngredients(1_596_000, 'Herbicides'),    by_year: generateByYear(38, 'El Dorado') },

  // Bay Area — lower agricultural use, some urban/suburban pesticide applications
  { county: 'Solano',       lbs_per_sq_mile: 118, lbs_applied_total:  2_478_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(118, 'Herbicides'),   top_ingredients: generateIngredients(2_478_000, 'Herbicides'),    by_year: generateByYear(118, 'Solano') },
  { county: 'Contra Costa', lbs_per_sq_mile:  52, lbs_applied_total:  1_092_000, top_pesticide_class: 'Insecticides', by_class: distributeByClass(52, 'Insecticides'),   top_ingredients: generateIngredients(1_092_000, 'Insecticides'),  by_year: generateByYear(52, 'Contra Costa') },
  { county: 'Alameda',      lbs_per_sq_mile:  28, lbs_applied_total:    504_000, top_pesticide_class: 'Insecticides', by_class: distributeByClass(28, 'Insecticides'),   top_ingredients: generateIngredients(504_000, 'Insecticides'),    by_year: generateByYear(28, 'Alameda') },

  // Northern CA
  { county: 'Glenn',        lbs_per_sq_mile: 214, lbs_applied_total:  2_996_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(214, 'Herbicides'),   top_ingredients: generateIngredients(2_996_000, 'Herbicides'),    by_year: generateByYear(214, 'Glenn') },
  { county: 'Sutter',       lbs_per_sq_mile: 192, lbs_applied_total:  1_920_000, top_pesticide_class: 'Fumigants',    by_class: distributeByClass(192, 'Fumigants'),    top_ingredients: generateIngredients(1_920_000, 'Fumigants'),     by_year: generateByYear(192, 'Sutter') },
  { county: 'Yuba',         lbs_per_sq_mile: 138, lbs_applied_total:    966_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(138, 'Herbicides'),   top_ingredients: generateIngredients(966_000, 'Herbicides'),      by_year: generateByYear(138, 'Yuba') },
  { county: 'Butte',        lbs_per_sq_mile: 106, lbs_applied_total:  4_452_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(106, 'Herbicides'),   top_ingredients: generateIngredients(4_452_000, 'Herbicides'),    by_year: generateByYear(106, 'Butte') },
  { county: 'Nevada',       lbs_per_sq_mile:  44, lbs_applied_total:  1_320_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(44, 'Herbicides'),    top_ingredients: generateIngredients(1_320_000, 'Herbicides'),    by_year: generateByYear(44, 'Nevada') },
  { county: 'Amador',       lbs_per_sq_mile:  32, lbs_applied_total:    384_000, top_pesticide_class: 'Herbicides',   by_class: distributeByClass(32, 'Herbicides'),    top_ingredients: generateIngredients(384_000, 'Herbicides'),      by_year: generateByYear(32, 'Amador') },
];

export const PESTICIDE_BY_COUNTY: Record<string, CountyPesticideData> = Object.fromEntries(
  MOCK_PESTICIDE_DATA.map(d => [d.county, d])
);
