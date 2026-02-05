import { useState } from 'react';
import type { RegionSummary } from '../../types';

interface SummaryTableProps {
  data: RegionSummary;
}

interface RowProps {
  item: RegionSummary;
  depth: number;
  isExpanded: boolean;
  onToggle: () => void;
  expandedItems: Set<string>;
  onItemToggle: (name: string) => void;
}

function SummaryRow({ item, depth, isExpanded, onToggle, expandedItems, onItemToggle }: RowProps) {
  const hasChildren = item.children && item.children.length > 0;
  const indent = depth * 20;
  
  const getTypeStyles = () => {
    switch (item.type) {
      case 'state':
        return 'font-semibold text-[var(--color-text-primary)] bg-gray-50';
      case 'catchment':
        return 'font-medium text-[var(--color-teal-dark)] bg-[#E6F3F5]';
      case 'region':
        return 'font-medium text-[var(--color-text-primary)]';
      case 'county':
        return 'text-[var(--color-text-secondary)]';
      default:
        return '';
    }
  };

  return (
    <>
      <tr className={`border-b border-gray-100 hover:bg-gray-50 transition-colors ${getTypeStyles()}`}>
        <td className="py-2 px-3">
          <div className="flex items-center" style={{ paddingLeft: `${indent}px` }}>
            {hasChildren && (
              <button
                onClick={onToggle}
                className="w-5 h-5 flex items-center justify-center text-gray-400 hover:text-[var(--color-teal)] mr-1 transition-colors"
              >
                <svg
                  className={`w-3 h-3 transition-transform duration-200 ${isExpanded ? 'rotate-90' : ''}`}
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            )}
            {!hasChildren && <span className="w-6" />}
            <span className="text-sm">{item.name}</span>
          </div>
        </td>
        <td className="py-2 px-3 text-right text-sm tabular-nums">
          {item.count.toLocaleString()}
        </td>
        <td className="py-2 px-3 text-right text-sm tabular-nums">
          {item.population.toLocaleString()}
        </td>
        <td className="py-2 px-3 text-right text-sm tabular-nums font-medium">
          {item.rate.toFixed(1)}
        </td>
      </tr>
      
      {isExpanded && hasChildren && item.children!.map((child) => (
        <SummaryRow
          key={child.name}
          item={child}
          depth={depth + 1}
          isExpanded={expandedItems.has(child.name)}
          onToggle={() => onItemToggle(child.name)}
          expandedItems={expandedItems}
          onItemToggle={onItemToggle}
        />
      ))}
    </>
  );
}

export function SummaryTable({ data }: SummaryTableProps) {
  const [expandedItems, setExpandedItems] = useState<Set<string>>(
    new Set(['California', 'UC Davis Catchment Area'])
  );

  const toggleItem = (name: string) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(name)) {
      newExpanded.delete(name);
    } else {
      newExpanded.add(name);
    }
    setExpandedItems(newExpanded);
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 shadow-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50">
        <h3 className="text-sm font-semibold text-[var(--color-text-primary)] uppercase tracking-wider">
          Regional Summary
        </h3>
        <p className="text-xs text-[var(--color-text-secondary)] mt-0.5">
          Age-adjusted rates per 10,000 dog population
        </p>
      </div>
      
      <div className="overflow-x-auto custom-scrollbar">
        <table className="w-full text-left">
          <thead>
            <tr className="bg-[var(--color-teal)] text-white">
              <th className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider">
                Region / County
              </th>
              <th className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider text-right">
                Count
              </th>
              <th className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider text-right">
                Population
              </th>
              <th className="py-2.5 px-3 text-xs font-semibold uppercase tracking-wider text-right">
                Rate
              </th>
            </tr>
          </thead>
          <tbody>
            <SummaryRow
              item={data}
              depth={0}
              isExpanded={expandedItems.has(data.name)}
              onToggle={() => toggleItem(data.name)}
              expandedItems={expandedItems}
              onItemToggle={toggleItem}
            />
          </tbody>
        </table>
      </div>
    </div>
  );
}
