// Mock California EPA Superfund (NPL) site data.
// Real data: https://www.epa.gov/superfund/search-superfund-sites-where-you-live
// Replace with live EPA ECHO API results when integrating real data.

export interface SuperfundSite {
  name: string;
  county: string;
  coordinates: [number, number]; // [longitude, latitude]
  status: 'active' | 'remediated' | 'proposed';
  contaminants: string[];
}

export const MOCK_SUPERFUND_SITES: SuperfundSite[] = [
  // Sacramento County
  { name: 'McClellan Air Force Base', county: 'Sacramento', coordinates: [-121.401, 38.668], status: 'active', contaminants: ['TCE', 'PCE', 'Heavy metals'] },
  { name: 'Aerojet General Corp', county: 'Sacramento', coordinates: [-121.220, 38.605], status: 'active', contaminants: ['Perchlorate', 'TCE', 'NDMA'] },
  { name: 'Sacramento Army Depot', county: 'Sacramento', coordinates: [-121.432, 38.530], status: 'remediated', contaminants: ['Solvents', 'Lead'] },

  // San Joaquin County
  { name: 'Sharpe Army Depot', county: 'San Joaquin', coordinates: [-121.288, 37.892], status: 'active', contaminants: ['Pesticides', 'Solvents', 'Heavy metals'] },
  { name: 'GBF Inc Landfill', county: 'San Joaquin', coordinates: [-121.176, 37.943], status: 'remediated', contaminants: ['VOCs', 'Heavy metals'] },

  // Stanislaus County
  { name: 'Modesto Groundwater Contamination', county: 'Stanislaus', coordinates: [-120.998, 37.639], status: 'active', contaminants: ['DBCP', 'EDB'] },
  { name: 'Del Amo Facility', county: 'Stanislaus', coordinates: [-120.850, 37.560], status: 'proposed', contaminants: ['Pesticides', 'Solvents'] },

  // Yolo County
  { name: 'Cal Compact Landfill', county: 'Yolo', coordinates: [-121.870, 38.752], status: 'remediated', contaminants: ['Leachate', 'VOCs'] },

  // Placer County
  { name: 'Rocklin Rodeo Area Groundwater', county: 'Placer', coordinates: [-121.240, 38.792], status: 'active', contaminants: ['PCE', 'TCE'] },

  // El Dorado County
  { name: 'El Dorado Mine', county: 'El Dorado', coordinates: [-120.480, 38.701], status: 'remediated', contaminants: ['Mercury', 'Arsenic'] },

  // Contra Costa County
  { name: 'Chemwest Systems Inc', county: 'Contra Costa', coordinates: [-121.780, 38.015], status: 'remediated', contaminants: ['PCBs', 'Heavy metals'] },
  { name: 'Acme Fill Corp', county: 'Contra Costa', coordinates: [-122.030, 37.940], status: 'active', contaminants: ['VOCs', 'Leachate'] },
  { name: 'IT Corp Field', county: 'Contra Costa', coordinates: [-121.900, 38.050], status: 'remediated', contaminants: ['Pesticides', 'Solvents'] },

  // Solano County
  { name: 'Travis Air Force Base', county: 'Solano', coordinates: [-121.927, 38.267], status: 'active', contaminants: ['PFAS', 'Solvents', 'Petroleum'] },
  { name: 'Cordelia Landfill', county: 'Solano', coordinates: [-122.065, 38.174], status: 'remediated', contaminants: ['Heavy metals', 'VOCs'] },

  // Alameda County
  { name: 'Davis Street Transfer Station', county: 'Alameda', coordinates: [-122.145, 37.700], status: 'remediated', contaminants: ['Heavy metals', 'Solvents'] },
  { name: 'Alameda Naval Air Station', county: 'Alameda', coordinates: [-122.315, 37.787], status: 'active', contaminants: ['PFAS', 'Solvents', 'Lead'] },

  // Butte County
  { name: 'Diamond National Corp', county: 'Butte', coordinates: [-121.810, 39.728], status: 'remediated', contaminants: ['Wood preservatives', 'PCP'] },

  // Sutter County
  { name: 'Pemaco Maywood Superfund', county: 'Sutter', coordinates: [-121.700, 39.020], status: 'proposed', contaminants: ['Petroleum', 'Solvents'] },

  // Yuba County
  { name: 'Yuba Goldfields', county: 'Yuba', coordinates: [-121.512, 39.152], status: 'active', contaminants: ['Mercury', 'Arsenic', 'Lead'] },

  // Colusa County
  { name: 'Colusa County Agricultural Land', county: 'Colusa', coordinates: [-122.228, 39.178], status: 'proposed', contaminants: ['DBCP', 'Pesticides'] },
];

// Aggregate site counts per county for choropleth / scatter plot
export const SUPERFUND_BY_COUNTY: Record<string, { total: number; active: number; remediated: number }> = (() => {
  const result: Record<string, { total: number; active: number; remediated: number }> = {};
  for (const site of MOCK_SUPERFUND_SITES) {
    if (!result[site.county]) result[site.county] = { total: 0, active: 0, remediated: 0 };
    result[site.county].total++;
    if (site.status === 'active') result[site.county].active++;
    if (site.status === 'remediated') result[site.county].remediated++;
  }
  return result;
})();
