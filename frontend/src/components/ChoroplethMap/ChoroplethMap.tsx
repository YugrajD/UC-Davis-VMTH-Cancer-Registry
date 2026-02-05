import { useMemo, useState } from 'react';
import { ComposableMap, Geographies, Geography } from 'react-simple-maps';
import { scaleLinear } from 'd3-scale';
import type { CountyData } from '../../types';

interface ChoroplethMapProps {
  data: CountyData[];
  rateRange: { min: number; max: number };
  hoveredCounty?: string | null;
  onCountyHover?: (county: string | null) => void;
  onCountyClick?: (county: string) => void;
}

// Use the reliable California counties GeoJSON from Code for America
const GEO_URL = 'https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/california-counties.geojson';

export function ChoroplethMap({ 
  data, 
  rateRange, 
  hoveredCounty, 
  onCountyHover,
  onCountyClick 
}: ChoroplethMapProps) {
  const [tooltipContent, setTooltipContent] = useState<{
    county: string;
    data: CountyData | undefined;
    x: number;
    y: number;
  } | null>(null);

  // Create a map of county name (lowercase) -> CountyData for quick lookup
  const countyDataMap = useMemo(() => {
    const map = new Map<string, CountyData>();
    data.forEach(county => {
      // Store with lowercase key for case-insensitive matching
      map.set(county.county.toLowerCase(), county);
    });
    return map;
  }, [data]);

  // Color scale for the choropleth
  const colorScale = useMemo(() => {
    return scaleLinear<string>()
      .domain([rateRange.min, (rateRange.min + rateRange.max) / 2, rateRange.max])
      .range(['#E6F3F5', '#6BB5BF', '#1A6B77']);
  }, [rateRange]);

  const handleMouseEnter = (
    countyName: string,
    event: React.MouseEvent
  ) => {
    const countyInfo = countyDataMap.get(countyName.toLowerCase());
    
    setTooltipContent({
      county: countyName,
      data: countyInfo,
      x: event.clientX,
      y: event.clientY,
    });
    
    onCountyHover?.(countyName);
  };

  const handleMouseLeave = () => {
    setTooltipContent(null);
    onCountyHover?.(null);
  };

  const handleClick = (countyName: string) => {
    onCountyClick?.(countyName);
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden relative">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
          California County Map
        </h3>
        <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
          Age-adjusted incidence rate per 10,000
        </p>
      </div>
      
      <div className="relative" style={{ minHeight: '450px', backgroundColor: '#f8fafc' }}>
        <ComposableMap
          projection="geoMercator"
          projectionConfig={{
            scale: 2400,
            center: [-119.5, 37.5],
          }}
          width={400}
          height={450}
          style={{
            width: '100%',
            height: '100%',
          }}
        >
          <Geographies geography={GEO_URL}>
            {({ geographies }) => {
              return geographies.map((geo) => {
                // The GeoJSON uses "name" property for county names
                const countyName = (geo.properties.name || '') as string;
                const countyInfo = countyDataMap.get(countyName.toLowerCase());
                const rate = countyInfo?.rate ?? 0;
                const isHovered = hoveredCounty?.toLowerCase() === countyName.toLowerCase();
                
                // Determine fill color based on rate
                const fillColor = rate > 0 ? colorScale(rate) : '#E5E7EB';
                
                return (
                  <Geography
                    key={geo.rsmKey}
                    geography={geo}
                    fill={isHovered ? '#F5A623' : fillColor}
                    stroke={isHovered ? '#E87722' : '#FFFFFF'}
                    strokeWidth={isHovered ? 1.5 : 0.5}
                    style={{
                      default: {
                        outline: 'none',
                      },
                      hover: {
                        fill: '#F5A623',
                        stroke: '#E87722',
                        strokeWidth: 1.5,
                        outline: 'none',
                        cursor: 'pointer',
                      },
                      pressed: {
                        fill: '#E87722',
                        outline: 'none',
                      },
                    }}
                    onMouseEnter={(event) => handleMouseEnter(countyName, event as unknown as React.MouseEvent)}
                    onMouseLeave={handleMouseLeave}
                    onClick={() => handleClick(countyName)}
                  />
                );
              });
            }}
          </Geographies>
        </ComposableMap>

        {/* Legend */}
        <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm rounded-lg p-3 border border-gray-200 shadow-sm">
          <p className="text-xs font-medium text-[var(--color-text-primary)] mb-2">
            Rate per 10,000
          </p>
          <div className="flex items-center gap-2">
            <div 
              className="w-32 h-3 rounded"
              style={{
                background: 'linear-gradient(to right, #E6F3F5, #6BB5BF, #1A6B77)',
              }}
            />
          </div>
          <div className="flex justify-between mt-1">
            <span className="text-[10px] text-[var(--color-text-secondary)]">
              {rateRange.min.toFixed(1)}
            </span>
            <span className="text-[10px] text-[var(--color-text-secondary)]">
              {rateRange.max.toFixed(1)}
            </span>
          </div>
          <div className="mt-2 pt-2 border-t border-gray-100">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-[#E5E7EB]" />
              <span className="text-[10px] text-[var(--color-text-secondary)]">No data</span>
            </div>
          </div>
        </div>
      </div>

      {/* Tooltip */}
      {tooltipContent && (
        <div
          className="fixed z-50 pointer-events-none"
          style={{
            left: tooltipContent.x + 12,
            top: tooltipContent.y - 12,
            transform: 'translateY(-100%)',
          }}
        >
          <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 min-w-[180px]">
            <p className="font-semibold text-sm text-[var(--color-text-primary)]">
              {tooltipContent.county}
            </p>
            {tooltipContent.data ? (
              <div className="mt-2 space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Region:</span>
                  <span className="font-medium">{tooltipContent.data.region}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Cases:</span>
                  <span className="font-medium">{tooltipContent.data.count.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Population:</span>
                  <span className="font-medium">{tooltipContent.data.population.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-[var(--color-text-secondary)]">Rate:</span>
                  <span className="font-semibold text-[var(--color-teal-dark)]">
                    {tooltipContent.data.rate.toFixed(1)}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-xs text-[var(--color-text-secondary)] mt-1 italic">
                No data available for this county
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
