# Workstream 5: Trend Line Visualization

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #5 (US #6)

## 5.1 Install Charting Library

Add `recharts` to `frontend/package.json`:

```bash
npm install recharts
```

`recharts` is chosen because:
- Pure React components (no D3 DOM manipulation conflicts)
- Built-in responsive container
- Lightweight (~40KB gzipped)
- Good TypeScript support

## 5.2 Frontend — TrendChart Component

**File:** `frontend/src/components/TrendChart/TrendChart.tsx`

**Component design:**

```
┌──────────────────────────────────────────────────┐
│  Cancer Case Trends Over Time                     │
│                                                    │
│  [All Cases]  [By Cancer Type]  (toggle buttons)   │
│                                                    │
│  Count                                             │
│  ▲                                                 │
│  │     ╱╲                                          │
│  │    ╱  ╲    ╱╲                                   │
│  │   ╱    ╲  ╱  ╲  ╱──╲                           │
│  │  ╱      ╲╱    ╲╱    ╲                           │
│  │ ╱                     ╲                          │
│  └────────────────────────────────▶ Year           │
│   1995  2000  2005  2010  2015  2020  2025         │
│                                                    │
│  Legend: ── All Cases  ── Lymphoma  ── MCT          │
│                                                    │
│  Tooltip: Year 2020: 342 cases                     │
└──────────────────────────────────────────────────┘
```

**Props:**

```typescript
interface TrendChartProps {
  filters: FilterState;  // Current filter state to pass to API
  mode?: 'overview' | 'detailed';  // overview shows single line, detailed shows by cancer type
}
```

**Data fetching:**

```typescript
// Uses existing backend endpoints:
// GET /api/v1/trends/yearly         → single "All Cases" line
// GET /api/v1/trends/by-cancer-type → one line per cancer type

// These endpoints already support filter params:
//   species[], cancer_type[], county[], sex
```

**Recharts implementation outline:**

```tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Transform API response (TrendsResponse) into recharts format:
// [{ year: 2020, "All Cases": 342, "Lymphoma": 45, "MCT": 38, ... }]

// Color palette matching UC Davis branding:
const CANCER_COLORS: Record<string, string> = {
  'All Cases': '#022851',        // UC Davis blue
  'Lymphoma': '#B4D3B2',
  'Mast Cell Tumor': '#F2A900',  // UC Davis gold
  'Osteosarcoma': '#6B7280',
  'Hemangiosarcoma': '#DC2626',
  'Melanoma': '#7C3AED',
  'Squamous Cell Carcinoma': '#059669',
  'Fibrosarcoma': '#D97706',
  'Transitional Cell Carcinoma': '#2563EB',
};
```

## 5.3 API Client Additions

Add to `frontend/src/api/client.ts`:

```typescript
export interface TrendPoint {
  year: number;
  count: number;
  deceased?: number;
  alive?: number;
}

export interface TrendSeries {
  name: string;
  data: TrendPoint[];
}

export interface TrendsResponse {
  series: TrendSeries[];
}

export async function fetchYearlyTrends(filters: FilterParams = {}): Promise<TrendsResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/trends/yearly?${params}` : '/api/v1/trends/yearly';
  return fetchJson(url);
}

export async function fetchTrendsByCancerType(filters: FilterParams = {}): Promise<TrendsResponse> {
  const params = filtersToParams(filters);
  const url = params.toString() ? `/api/v1/trends/by-cancer-type?${params}` : '/api/v1/trends/by-cancer-type';
  return fetchJson(url);
}
```

## 5.4 Integration into App

1. **Overview tab:** Add `<TrendChart mode="overview" />` below the map/tables grid.
2. **New "Trends" tab:** Add dedicated `TabType = 'trends'` with `<TrendChart mode="detailed" />` showing by-cancer-type multi-series view.

## 5.5 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `frontend/src/components/TrendChart/TrendChart.tsx` | Recharts line chart |

**Files to modify:**
| File | Change |
|------|--------|
| `frontend/package.json` | Add `recharts` |
| `frontend/src/api/client.ts` | Add `fetchYearlyTrends`, `fetchTrendsByCancerType`, trend types |
| `frontend/src/types/index.ts` | Add `TrendPoint`, `TrendSeries`, `TrendsResponse`, extend TabType with 'trends' |
| `frontend/src/App.tsx` | Add TrendChart to overview + trends tab |
| `frontend/src/components/index.ts` | Export TrendChart |
