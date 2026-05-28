// California pesticide use data by county.
// Source: CDPR Pesticide Use Reporting (PUR) database.
// Annual report text files: https://files.cdpr.ca.gov/pub/outgoing/pur/data/
// Values: lbs of active ingredient per square mile (5-year avg 2015–2019).
// Class breakdown uses 2019 proportions applied to the 5-year average.
// Last updated: May 2026

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
  lbs_per_sq_mile: number;
  lbs_applied_total: number;
  top_pesticide_class: string;
  by_class: Record<PesticideClass, number>;
  top_ingredients: ActiveIngredient[];
  by_year: Record<number, number>;
}

export const MOCK_PESTICIDE_DATA: CountyPesticideData[] = [
  {
    county: 'San Joaquin',
    lbs_per_sq_mile: 9319,
    lbs_applied_total: 13289254,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 1028, herbicides: 803, insecticides: 1134, fungicides: 6354 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 7646513, category: 'fungicides' },
      { name: "Mineral Oil", lbs_applied: 860575, category: 'insecticides' },
      { name: "1,3-Dichloropropene", lbs_applied: 750433, category: 'fumigants' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 377664, category: 'herbicides' },
      { name: "Glyphosate (K salt)", lbs_applied: 288388, category: 'herbicides' },
    ],
    by_year: { 2015: 8973, 2016: 8888, 2017: 9456, 2018: 9554, 2019: 9726 },
  },
  {
    county: 'Stanislaus',
    lbs_per_sq_mile: 5622,
    lbs_applied_total: 8405088,
    top_pesticide_class: 'Insecticides',
    by_class: { fumigants: 1660, herbicides: 967, insecticides: 1811, fungicides: 1184 },
    top_ingredients: [
      { name: "Mineral Oil", lbs_applied: 1776918, category: 'insecticides' },
      { name: "1,3-Dichloropropene", lbs_applied: 1346852, category: 'fumigants' },
      { name: "Sulfur", lbs_applied: 759770, category: 'fungicides' },
      { name: "Potassium N-methyldithiocarbamate", lbs_applied: 529033, category: 'fumigants' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 496753, category: 'herbicides' },
    ],
    by_year: { 2015: 5370, 2016: 5319, 2017: 5482, 2018: 5892, 2019: 6048 },
  },
  {
    county: 'Sutter',
    lbs_per_sq_mile: 5399,
    lbs_applied_total: 3282491,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 681, herbicides: 1598, insecticides: 1367, fungicides: 1753 },
    top_ingredients: [
      { name: "Mineral Oil", lbs_applied: 479930, category: 'insecticides' },
      { name: "Propanil", lbs_applied: 406682, category: 'herbicides' },
      { name: "1,3-Dichloropropene", lbs_applied: 274166, category: 'fumigants' },
      { name: "Sulfur", lbs_applied: 242521, category: 'fungicides' },
      { name: "Copper hydroxide", lbs_applied: 239970, category: 'fungicides' },
    ],
    by_year: { 2015: 5596, 2016: 5543, 2017: 5193, 2018: 5206, 2019: 5457 },
  },
  {
    county: 'Sacramento',
    lbs_per_sq_mile: 4567,
    lbs_applied_total: 4539137,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 27, herbicides: 313, insecticides: 833, fungicides: 3394 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 3028962, category: 'fungicides' },
      { name: "Mineral Oil", lbs_applied: 653225, category: 'insecticides' },
      { name: "Glyphosate (K salt)", lbs_applied: 86608, category: 'herbicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 82570, category: 'herbicides' },
      { name: "Lime-sulfur", lbs_applied: 46006, category: 'fungicides' },
    ],
    by_year: { 2015: 4407, 2016: 4365, 2017: 4392, 2018: 5111, 2019: 4558 },
  },
  {
    county: 'Yolo',
    lbs_per_sq_mile: 4256,
    lbs_applied_total: 4357847,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 182, herbicides: 670, insecticides: 487, fungicides: 2917 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 2415528, category: 'fungicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 169490, category: 'herbicides' },
      { name: "Mineral Oil", lbs_applied: 163620, category: 'insecticides' },
      { name: "Glyphosate (K salt)", lbs_applied: 134350, category: 'herbicides' },
      { name: "Kaolin", lbs_applied: 102322, category: 'insecticides' },
    ],
    by_year: { 2015: 4497, 2016: 4454, 2017: 3934, 2018: 4291, 2019: 4102 },
  },
  {
    county: 'Colusa',
    lbs_per_sq_mile: 2526,
    lbs_applied_total: 2907966,
    top_pesticide_class: 'Herbicides',
    by_class: { fumigants: 57, herbicides: 1298, insecticides: 503, fungicides: 668 },
    top_ingredients: [
      { name: "Propanil", lbs_applied: 468990, category: 'herbicides' },
      { name: "Thiobencarb", lbs_applied: 245734, category: 'herbicides' },
      { name: "Glyphosate (K salt)", lbs_applied: 183731, category: 'herbicides' },
      { name: "Sulfur", lbs_applied: 162780, category: 'fungicides' },
      { name: "Kaolin", lbs_applied: 124466, category: 'insecticides' },
    ],
    by_year: { 2015: 2753, 2016: 2727, 2017: 2449, 2018: 2523, 2019: 2181 },
  },
  {
    county: 'Yuba',
    lbs_per_sq_mile: 1961,
    lbs_applied_total: 1254884,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 438, herbicides: 472, insecticides: 437, fungicides: 614 },
    top_ingredients: [
      { name: "1,3-Dichloropropene", lbs_applied: 197231, category: 'fumigants' },
      { name: "Mineral Oil", lbs_applied: 151814, category: 'insecticides' },
      { name: "Propanil", lbs_applied: 133339, category: 'herbicides' },
      { name: "Copper hydroxide", lbs_applied: 110699, category: 'fungicides' },
      { name: "Sulfur", lbs_applied: 57907, category: 'fungicides' },
    ],
    by_year: { 2015: 2024, 2016: 2005, 2017: 2047, 2018: 1775, 2019: 1954 },
  },
  {
    county: 'Butte',
    lbs_per_sq_mile: 1932,
    lbs_applied_total: 3160569,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 202, herbicides: 595, insecticides: 400, fungicides: 735 },
    top_ingredients: [
      { name: "Copper hydroxide", lbs_applied: 339034, category: 'fungicides' },
      { name: "Propanil", lbs_applied: 337268, category: 'herbicides' },
      { name: "1,3-Dichloropropene", lbs_applied: 223069, category: 'fumigants' },
      { name: "Mineral Oil", lbs_applied: 214842, category: 'insecticides' },
      { name: "Copper sulfate", lbs_applied: 197153, category: 'fungicides' },
    ],
    by_year: { 2015: 2019, 2016: 1999, 2017: 2043, 2018: 1813, 2019: 1785 },
  },
  {
    county: 'Glenn',
    lbs_per_sq_mile: 1839,
    lbs_applied_total: 2420332,
    top_pesticide_class: 'Herbicides',
    by_class: { fumigants: 70, herbicides: 793, insecticides: 346, fungicides: 630 },
    top_ingredients: [
      { name: "Propanil", lbs_applied: 315287, category: 'herbicides' },
      { name: "Copper hydroxide", lbs_applied: 155756, category: 'fungicides' },
      { name: "Copper sulfate", lbs_applied: 124454, category: 'fungicides' },
      { name: "Glyphosate (K salt)", lbs_applied: 124313, category: 'herbicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 122587, category: 'herbicides' },
    ],
    by_year: { 2015: 1760, 2016: 1743, 2017: 1910, 2018: 1955, 2019: 1828 },
  },
  {
    county: 'Solano',
    lbs_per_sq_mile: 1550,
    lbs_applied_total: 1406260,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 46, herbicides: 408, insecticides: 293, fungicides: 802 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 534890, category: 'fungicides' },
      { name: "Mineral Oil", lbs_applied: 152998, category: 'insecticides' },
      { name: "Glyphosate (K salt)", lbs_applied: 107492, category: 'herbicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 96196, category: 'herbicides' },
      { name: "Pendimethalin", lbs_applied: 41648, category: 'herbicides' },
    ],
    by_year: { 2015: 1499, 2016: 1484, 2017: 1475, 2018: 1667, 2019: 1628 },
  },
  {
    county: 'Contra Costa',
    lbs_per_sq_mile: 743,
    lbs_applied_total: 532335,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 37, herbicides: 181, insecticides: 182, fungicides: 343 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 165956, category: 'fungicides' },
      { name: "Mineral Oil", lbs_applied: 36836, category: 'insecticides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 35025, category: 'herbicides' },
      { name: "Glyphosate (K salt)", lbs_applied: 31789, category: 'herbicides' },
      { name: "Disodium Octaborate Tetrahydra", lbs_applied: 24587, category: 'insecticides' },
    ],
    by_year: { 2015: 795, 2016: 787, 2017: 627, 2018: 721, 2019: 788 },
  },
  {
    county: 'Alameda',
    lbs_per_sq_mile: 453,
    lbs_applied_total: 334224,
    top_pesticide_class: 'Fumigants',
    by_class: { fumigants: 152, herbicides: 104, insecticides: 105, fungicides: 92 },
    top_ingredients: [
      { name: "Sulfuryl Fluoride", lbs_applied: 63137, category: 'fumigants' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 22592, category: 'herbicides' },
      { name: "Sulfur", lbs_applied: 21272, category: 'fungicides' },
      { name: "Mineral Oil", lbs_applied: 19955, category: 'insecticides' },
      { name: "Disodium Octaborate Tetrahydra", lbs_applied: 13405, category: 'insecticides' },
    ],
    by_year: { 2015: 493, 2016: 489, 2017: 510, 2018: 387, 2019: 385 },
  },
  {
    county: 'Placer',
    lbs_per_sq_mile: 273,
    lbs_applied_total: 410517,
    top_pesticide_class: 'Herbicides',
    by_class: { fumigants: 37, herbicides: 105, insecticides: 48, fungicides: 82 },
    top_ingredients: [
      { name: "Copper sulfate", lbs_applied: 59172, category: 'fungicides' },
      { name: "Propanil", lbs_applied: 40245, category: 'herbicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 38905, category: 'herbicides' },
      { name: "1,3-Dichloropropene", lbs_applied: 23302, category: 'fumigants' },
      { name: "Diquat Dibromide", lbs_applied: 19952, category: 'herbicides' },
    ],
    by_year: { 2015: 305, 2016: 302, 2017: 200, 2018: 289, 2019: 270 },
  },
  {
    county: 'Amador',
    lbs_per_sq_mile: 166,
    lbs_applied_total: 98506,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 1, herbicides: 23, insecticides: 33, fungicides: 110 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 79949, category: 'fungicides' },
      { name: "Mineral Oil", lbs_applied: 12016, category: 'insecticides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 9673, category: 'herbicides' },
      { name: "Methylated Soybean Oil", lbs_applied: 8231, category: 'insecticides' },
      { name: "Kaolin", lbs_applied: 2671, category: 'insecticides' },
    ],
    by_year: { 2015: 133, 2016: 131, 2017: 164, 2018: 172, 2019: 228 },
  },
  {
    county: 'El Dorado',
    lbs_per_sq_mile: 107,
    lbs_applied_total: 191731,
    top_pesticide_class: 'Insecticides',
    by_class: { fumigants: 4, herbicides: 34, insecticides: 43, fungicides: 26 },
    top_ingredients: [
      { name: "Mineral Oil", lbs_applied: 40726, category: 'insecticides' },
      { name: "Sulfur", lbs_applied: 30315, category: 'fungicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 21688, category: 'herbicides' },
      { name: "Glyphosate (DMA salt)", lbs_applied: 19397, category: 'herbicides' },
      { name: "Methylated Soybean Oil", lbs_applied: 6446, category: 'insecticides' },
    ],
    by_year: { 2015: 130, 2016: 129, 2017: 79, 2018: 90, 2019: 109 },
  },
  {
    county: 'Nevada',
    lbs_per_sq_mile: 77,
    lbs_applied_total: 75263,
    top_pesticide_class: 'Fungicides',
    by_class: { fumigants: 1, herbicides: 23, insecticides: 9, fungicides: 44 },
    top_ingredients: [
      { name: "Sulfur", lbs_applied: 15940, category: 'fungicides' },
      { name: "Glyphosate (DMA salt)", lbs_applied: 5361, category: 'herbicides' },
      { name: "Copper Ethanolamine Complexes,", lbs_applied: 4872, category: 'fungicides' },
      { name: "Glyphosate (IPA salt)", lbs_applied: 4824, category: 'herbicides' },
      { name: "Mineral Oil", lbs_applied: 2321, category: 'insecticides' },
    ],
    by_year: { 2015: 101, 2016: 100, 2017: 73, 2018: 58, 2019: 54 },
  },
];

export const PESTICIDE_BY_COUNTY: Record<string, CountyPesticideData> = Object.fromEntries(
  MOCK_PESTICIDE_DATA.map(d => [d.county, d])
);
