/**
 * California county-level age-adjusted all-cancer incidence rates per 100,000 population.
 * Source: California Cancer Registry via California Health Maps (californiahealthmaps.org)
 * Period: 2017–2021 (5-year average)
 * Both sexes, all races/ethnicities, all ages.
 * Age-adjusted to the 2000 U.S. Standard Population.
 */
export interface HumanCancerRate {
  county: string;
  /** Age-adjusted incidence rate per 100,000; null if suppressed (<16 cases) */
  rate: number | null;
  /** Estimated average annual case count; null if suppressed */
  cases: number | null;
}

export const HUMAN_CANCER_RATES: HumanCancerRate[] = [
  { county: 'Alameda', rate: 372.4, cases: 6227 },
  { county: 'Alpine', rate: null, cases: null },
  { county: 'Amador', rate: 427.1, cases: 171 },
  { county: 'Butte', rate: 484.8, cases: 1054 },
  { county: 'Calaveras', rate: 416.1, cases: 189 },
  { county: 'Colusa', rate: 373.7, cases: 81 },
  { county: 'Contra Costa', rate: 410.1, cases: 4765 },
  { county: 'Del Norte', rate: 343.4, cases: 95 },
  { county: 'El Dorado', rate: 425.0, cases: 810 },
  { county: 'Fresno', rate: 390.7, cases: 3941 },
  { county: 'Glenn', rate: 487.2, cases: 140 },
  { county: 'Humboldt', rate: 441.4, cases: 603 },
  { county: 'Imperial', rate: 367.4, cases: 661 },
  { county: 'Inyo', rate: 345.3, cases: 65 },
  { county: 'Kern', rate: 403.3, cases: 3644 },
  { county: 'Kings', rate: 365.0, cases: 553 },
  { county: 'Lake', rate: 421.1, cases: 285 },
  { county: 'Lassen', rate: 331.2, cases: 108 },
  { county: 'Los Angeles', rate: 369.8, cases: 37039 },
  { county: 'Madera', rate: 394.4, cases: 615 },
  { county: 'Marin', rate: 443.7, cases: 1163 },
  { county: 'Mariposa', rate: 403.4, cases: 69 },
  { county: 'Mendocino', rate: 405.7, cases: 371 },
  { county: 'Merced', rate: 379.6, cases: 1058 },
  { county: 'Modoc', rate: 293.0, cases: 26 },
  { county: 'Mono', rate: 291.7, cases: 39 },
  { county: 'Monterey', rate: 374.0, cases: 1641 },
  { county: 'Napa', rate: 416.6, cases: 578 },
  { county: 'Nevada', rate: 401.7, cases: 410 },
  { county: 'Orange', rate: 408.8, cases: 13005 },
  { county: 'Placer', rate: 437.5, cases: 1752 },
  { county: 'Plumas', rate: 361.2, cases: 71 },
  { county: 'Riverside', rate: 398.7, cases: 9596 },
  { county: 'Sacramento', rate: 407.8, cases: 6409 },
  { county: 'San Benito', rate: 397.6, cases: 252 },
  { county: 'San Bernardino', rate: 399.1, cases: 8663 },
  { county: 'San Diego', rate: 428.9, cases: 14122 },
  { county: 'San Francisco', rate: 385.0, cases: 3331 },
  { county: 'San Joaquin', rate: 397.1, cases: 3062 },
  { county: 'San Luis Obispo', rate: 456.3, cases: 1286 },
  { county: 'San Mateo', rate: 399.2, cases: 3044 },
  { county: 'Santa Barbara', rate: 453.1, cases: 2020 },
  { county: 'Santa Clara', rate: 384.6, cases: 7431 },
  { county: 'Santa Cruz', rate: 464.5, cases: 1257 },
  { county: 'Shasta', rate: 463.0, cases: 843 },
  { county: 'Sierra', rate: 259.6, cases: 8 },
  { county: 'Siskiyou', rate: 362.7, cases: 160 },
  { county: 'Solano', rate: 418.1, cases: 1886 },
  { county: 'Sonoma', rate: 436.3, cases: 2146 },
  { county: 'Stanislaus', rate: 424.7, cases: 2339 },
  { county: 'Sutter', rate: 415.8, cases: 412 },
  { county: 'Tehama', rate: 449.7, cases: 293 },
  { county: 'Trinity', rate: 303.4, cases: 48 },
  { county: 'Tulare', rate: 362.7, cases: 1708 },
  { county: 'Tuolumne', rate: 421.5, cases: 232 },
  { county: 'Ventura', rate: 435.0, cases: 3671 },
  { county: 'Yolo', rate: 414.2, cases: 894 },
  { county: 'Yuba', rate: 449.2, cases: 361 },
];
