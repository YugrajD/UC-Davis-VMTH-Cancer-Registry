// California EPA Superfund National Priorities List (NPL) site data.
// Sources: EPA SEMS/CUMULIS site profiles, ATSDR health assessments, latitude.to verified coordinates,
// Wikipedia "List of Superfund sites in California", EPA Region 9 records.
// Coordinates: [longitude, latitude]
// Status: "active" = currently on Final NPL | "remediated" = deleted from NPL | "proposed" = proposed NPL
// Last verified: May 2026  (California has ~97 active, ~17 deleted, ~2 proposed NPL sites)

export interface SuperfundSite {
  name: string;
  county: string;
  coordinates: [number, number]; // [longitude, latitude]
  status: 'active' | 'remediated' | 'proposed';
  contaminants: string[];
}

export const SUPERFUND_SITES: SuperfundSite[] = [
  // === RIVERSIDE COUNTY ===
  {
    name: 'Stringfellow Acid Pits',
    county: 'Riverside',
    // Coordinates from latitude.to verified: 34.0234, -117.4607
    coordinates: [-117.4607, 34.0234],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Sulfuric acid', 'Heavy metals'],
  },

  // === LOS ANGELES COUNTY ===
  {
    name: 'Operating Industries Inc., Landfill',
    county: 'Los Angeles',
    // Coordinates from latitude.to verified: 34.0328, -118.1133 (900 N Potrero Grande Dr, Monterey Park)
    coordinates: [-118.1133, 34.0328],
    status: 'remediated',
    contaminants: ['Volatile organic compounds', 'Methane', 'Liquid industrial chemicals'],
  },
  {
    name: 'Montrose Chemical Corp.',
    county: 'Los Angeles',
    // 20201 Normandie Ave, Los Angeles (near Torrance) — address geocoded
    coordinates: [-118.3057, 33.8683],
    status: 'active',
    contaminants: ['DDT', 'Chlorobenzene', 'PCBs'],
  },
  {
    name: 'Del Amo',
    county: 'Los Angeles',
    // 1105 W Del Amo Blvd, Torrance — address geocoded
    coordinates: [-118.3012, 33.8450],
    status: 'active',
    contaminants: ['Benzene', 'Styrene', 'Butadiene'],
  },
  {
    name: 'Pemaco Maywood',
    county: 'Los Angeles',
    // 5050 Slauson Blvd, Maywood, CA — address geocoded
    coordinates: [-118.1836, 33.9876],
    status: 'active',
    contaminants: ['Volatile organic compounds', 'Solvents', 'Heavy metals'],
  },
  {
    name: 'San Gabriel Valley (Area 1)',
    county: 'Los Angeles',
    // El Monte / South El Monte / Rosemead area centroid
    coordinates: [-118.0705, 34.0522],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Perchloroethylene', 'Volatile organic compounds'],
  },
  {
    name: 'San Gabriel Valley (Area 2)',
    county: 'Los Angeles',
    // Baldwin Park / Azusa / West Covina area centroid
    coordinates: [-117.9611, 34.0753],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Perchloroethylene', 'Volatile organic compounds'],
  },
  {
    name: 'San Gabriel Valley (Area 3)',
    county: 'Los Angeles',
    // 19 sq-mile area, eastern Los Angeles County centroid
    coordinates: [-117.9150, 34.0300],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Tetrachloroethylene', 'Volatile organic compounds'],
  },
  {
    name: 'San Gabriel Valley (Area 4)',
    county: 'Los Angeles',
    // 5 sq-mile area, eastern Los Angeles County centroid
    coordinates: [-117.8700, 34.0700],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Tetrachloroethylene', 'Volatile organic compounds'],
  },

  // === ORANGE COUNTY ===
  {
    name: 'McColl',
    county: 'Orange',
    // Rosecrans & Sunny Ridge, Fullerton, CA — address geocoded
    coordinates: [-117.9247, 33.8886],
    status: 'active',
    contaminants: ['Benzene', 'Thiophane', 'Petroleum refinery waste'],
  },

  // === SANTA BARBARA COUNTY ===
  {
    name: 'Casmalia Resources',
    county: 'Santa Barbara',
    // 1.2 miles N of Town of Casmalia, 10 mi SW of Santa Maria — location estimated
    coordinates: [-120.5427, 34.8455],
    status: 'active',
    contaminants: ['Solvents', 'Pesticides', 'PCBs', 'Heavy metals'],
  },

  // === SACRAMENTO COUNTY ===
  {
    name: 'Aerojet General Corp.',
    county: 'Sacramento',
    // 5,900-acre site, Rancho Cordova, 15 mi east of Sacramento — location estimated
    coordinates: [-121.2750, 38.5833],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Perchlorate', 'NDMA'],
  },
  {
    name: 'McClellan Air Force Base',
    county: 'Sacramento',
    // Coordinates from latitude.to verified: 38.6673, -121.4003
    coordinates: [-121.4003, 38.6673],
    status: 'active',
    contaminants: ['Trichloroethylene', 'PCBs', 'Heavy metals'],
  },
  {
    name: 'Mather Air Force Base',
    county: 'Sacramento',
    // Coordinates from latitude.to verified: 38.5523, -121.2918
    coordinates: [-121.2918, 38.5523],
    status: 'remediated',
    contaminants: ['Trichloroethylene', 'Petroleum hydrocarbons', 'Solvents'],
  },
  {
    name: 'Sacramento Army Depot',
    county: 'Sacramento',
    // Coordinates from latitude.to verified: 38.5180, -121.3913
    coordinates: [-121.3913, 38.5180],
    status: 'remediated',
    contaminants: ['Trichloroethylene', 'PCBs', 'Solvents'],
  },

  // === MONTEREY COUNTY ===
  {
    name: 'Fort Ord',
    county: 'Monterey',
    // Coordinates from latitude.to verified: 36.6392, -121.7353
    coordinates: [-121.7353, 36.6392],
    status: 'active',
    contaminants: ['Trichloroethylene', 'PFAS', 'Volatile organic compounds'],
  },
  {
    name: 'Firestone Tire & Rubber Co. (Salinas Plant)',
    county: 'Monterey',
    // 43-acre former tire plant, Salinas — address geocoded
    coordinates: [-121.6555, 36.6777],
    status: 'remediated',
    contaminants: ['Trichloroethylene', 'Petroleum hydrocarbons', 'Volatile organic compounds'],
  },

  // === SANTA CLARA COUNTY (23 NPL sites — most of any US county) ===
  {
    name: 'Fairchild Semiconductor Corp. (Mountain View Plant)',
    county: 'Santa Clara',
    // MEW Study Area, Mountain View — location estimated from city centroid
    coordinates: [-122.0631, 37.3980],
    status: 'active',
    contaminants: ['Trichloroethylene', '1,1-Dichloroethylene', 'Volatile organic compounds'],
  },
  {
    name: 'Intel Corp. (Mountain View Plant)',
    county: 'Santa Clara',
    // 365 E Middlefield Rd, Mountain View — address geocoded
    coordinates: [-122.0580, 37.3994],
    status: 'remediated',
    contaminants: ['Trichloroethylene', 'Volatile organic compounds', 'Solvents'],
  },
  {
    name: 'Fairchild Semiconductor Corp. (South San Jose Plant)',
    county: 'Santa Clara',
    // 101 Bernal Rd, San Jose — address geocoded
    coordinates: [-121.8090, 37.2670],
    status: 'active',
    contaminants: ['Trichloroethylene', '1,1-Dichloroethylene', 'Freon'],
  },
  {
    name: 'National Semiconductor Corp.',
    county: 'Santa Clara',
    // 2900 Semiconductor Dr, Santa Clara / Sunnyvale — address geocoded
    coordinates: [-122.0047, 37.3685],
    status: 'active',
    contaminants: ['Volatile organic compounds', 'Trichloroethylene', 'Perchloroethylene'],
  },
  {
    name: 'Advanced Micro Devices (901/902 Thompson Place)',
    county: 'Santa Clara',
    // 901-902 Thompson Pl, Sunnyvale — address geocoded
    coordinates: [-122.0233, 37.3947],
    status: 'active',
    contaminants: ['Volatile organic compounds', 'Trichloroethylene'],
  },
  {
    name: 'Advanced Micro Devices (Building 915 DeGuigne Drive)',
    county: 'Santa Clara',
    // 915 DeGuigne Dr, Sunnyvale — address geocoded
    coordinates: [-122.0356, 37.3842],
    status: 'active',
    contaminants: ['Volatile organic compounds', 'Trichloroethylene', 'Perchloroethylene'],
  },
  {
    name: 'Lorentz Barrel & Drum Co.',
    county: 'Santa Clara',
    // 7-acre site, San Jose — location estimated
    coordinates: [-121.9013, 37.3770],
    status: 'active',
    contaminants: ['Volatile organic compounds', 'Solvents', 'Heavy metals'],
  },
  {
    name: 'South Bay Asbestos Area',
    county: 'Santa Clara',
    // Alviso District, San Jose — location estimated
    coordinates: [-121.9808, 37.4058],
    status: 'active',
    contaminants: ['Asbestos'],
  },
  {
    name: 'Intel Corp. (Santa Clara III)',
    county: 'Santa Clara',
    // 2880 Northwestern Pkwy, Santa Clara — address geocoded; deleted from NPL 2019
    coordinates: [-121.9875, 37.3882],
    status: 'remediated',
    contaminants: ['Trichloroethylene', 'Volatile organic compounds'],
  },

  // === SAN MATEO COUNTY ===
  {
    name: 'Romic Chemical Corp. (East Palo Alto)',
    county: 'San Mateo',
    // 2081 Bay Rd, East Palo Alto — address geocoded
    coordinates: [-122.1292, 37.4680],
    status: 'active',
    contaminants: ['Volatile organic compounds', 'Solvents', 'Heavy metals'],
  },

  // === SAN FRANCISCO COUNTY ===
  {
    name: 'Hunters Point Naval Shipyard',
    county: 'San Francisco',
    // Coordinates from latitude.to verified: 37.7264, -122.3639
    coordinates: [-122.3639, 37.7264],
    status: 'active',
    contaminants: ['PCBs', 'Heavy metals', 'Radioactive materials', 'Solvents'],
  },

  // === SOLANO COUNTY ===
  {
    name: 'Mare Island Naval Shipyard',
    county: 'Solano',
    // Coordinates from latitude.to verified: 38.0953, -122.2780
    coordinates: [-122.2780, 38.0953],
    status: 'active',
    contaminants: ['PCBs', 'Heavy metals', 'Solvents', 'Petroleum hydrocarbons'],
  },
  {
    name: 'Travis Air Force Base',
    county: 'Solano',
    // Fairfield, Solano County — location estimated
    coordinates: [-121.9270, 38.2670],
    status: 'active',
    contaminants: ['PFAS', 'Solvents', 'Petroleum hydrocarbons'],
  },

  // === FRESNO COUNTY ===
  {
    name: 'Fresno Municipal Sanitary Landfill',
    county: 'Fresno',
    // 4 miles from Fresno city center — location estimated
    coordinates: [-119.8011, 36.7380],
    status: 'remediated',
    contaminants: ['Methane', 'Vinyl chloride', 'Volatile organic compounds'],
  },
  {
    name: 'Atlas Asbestos Mine',
    county: 'Fresno',
    // 18 miles NW of Coalinga, western Fresno County — location estimated
    coordinates: [-120.3543, 36.2187],
    status: 'remediated',
    contaminants: ['Asbestos'],
  },

  // === SAN DIEGO COUNTY ===
  {
    name: 'Camp Pendleton Marine Corps Base',
    county: 'San Diego',
    // 125,000-acre base, San Diego County — location estimated
    coordinates: [-117.4310, 33.2750],
    status: 'active',
    contaminants: ['PCBs', 'Pesticides', 'Petroleum hydrocarbons', 'Volatile organic compounds'],
  },
  {
    name: 'Marine Corps Recruit Depot San Diego',
    county: 'San Diego',
    // San Diego city — location estimated
    coordinates: [-117.1973, 32.7270],
    status: 'active',
    contaminants: ['PCBs', 'Volatile organic compounds', 'Solvents'],
  },
  {
    name: 'Miramar Naval Air Station',
    county: 'San Diego',
    // Marine Corps Air Station Miramar — location estimated
    coordinates: [-117.1427, 32.8688],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Petroleum hydrocarbons', 'Heavy metals'],
  },

  // === TULARE COUNTY ===
  {
    name: 'Beckman Instruments (Porterville Plant)',
    county: 'Tulare',
    // Porterville, Tulare County — address geocoded
    coordinates: [-119.0165, 36.0657],
    status: 'active',
    contaminants: ['1,1-Dichloroethylene', 'Freon 113', 'Trichloroethylene'],
  },

  // === CONTRA COSTA COUNTY ===
  {
    name: 'Liquid Gold Oil Corp.',
    county: 'Contra Costa',
    // Richmond, Contra Costa County — location estimated
    coordinates: [-122.3477, 37.9477],
    status: 'active',
    contaminants: ['Petroleum hydrocarbons', 'Heavy metals', 'Solvents'],
  },

  // === SAN BERNARDINO COUNTY ===
  {
    name: 'Newmark Ground Water Contamination',
    county: 'San Bernardino',
    // 8 sq-mile plume, San Bernardino city — location estimated
    coordinates: [-117.2898, 34.1083],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Perchloroethylene', 'Volatile organic compounds'],
  },
  {
    name: 'Barstow Marine Corps Logistics Base',
    county: 'San Bernardino',
    // Barstow, San Bernardino County — location estimated from city: 34.8986, -117.0228
    coordinates: [-116.9450, 34.9000],
    status: 'active',
    contaminants: ['Trichloroethylene', 'Petroleum hydrocarbons', 'Solvents'],
  },

  // === VENTURA COUNTY ===
  {
    name: 'Halaco Engineering Co.',
    county: 'Ventura',
    // 6200 Perkins Rd, Oxnard — address geocoded; near Ormond Beach
    coordinates: [-119.2398, 34.2250],
    status: 'active',
    contaminants: ['Heavy metals', 'Zinc', 'Lead', 'Cadmium'],
  },

  // === ALAMEDA COUNTY ===
  {
    name: 'Alameda Naval Air Station',
    county: 'Alameda',
    // Former Naval Air Station, Alameda Island — location estimated
    coordinates: [-122.3150, 37.7870],
    status: 'active',
    contaminants: ['PFAS', 'Solvents', 'Lead'],
  },
];

// Backward-compatible alias (components still import MOCK_SUPERFUND_SITES)
export const MOCK_SUPERFUND_SITES = SUPERFUND_SITES;

// Aggregate site counts per county for choropleth / scatter plot
export const SUPERFUND_BY_COUNTY: Record<string, { total: number; active: number; remediated: number }> = (() => {
  const result: Record<string, { total: number; active: number; remediated: number }> = {};
  for (const site of SUPERFUND_SITES) {
    if (!result[site.county]) result[site.county] = { total: 0, active: 0, remediated: 0 };
    result[site.county].total++;
    if (site.status === 'active') result[site.county].active++;
    if (site.status === 'remediated') result[site.county].remediated++;
  }
  return result;
})();
