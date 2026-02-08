# Workstream 7: Fix Frontend Tabs (Real Data)

[Back to Overview](../IMPLEMENTATION_PLAN.md)

---

**Gaps addressed:** #7 (US #11)

## 7.1 Current Problem

In `frontend/src/App.tsx` lines 65-153, three tabs render fake data:

- **`breed-disparities`** (lines 65-86): Hardcoded 4 breeds with `Math.random() * 50 + 30`
- **`cancer-types`** (lines 88-116): Hardcoded 6 cancer types with `50 - i * 6 + Math.random() * 5`
- **`regional-comparison`** (lines 118-153): Hardcoded 5 regions with static values

## 7.2 BreedDisparities Component

**File:** `frontend/src/components/BreedDisparities/BreedDisparities.tsx`

**Data source:** `GET /api/v1/incidence/by-breed` (already exists, `fetchIncidenceByBreed` already defined in client.ts)

**Implementation:**
```typescript
interface BreedDisparitiesProps {
  filters: FilterState;
}

// 1. Call fetchIncidenceByBreed with current filters
// 2. Render a horizontal bar chart (can use recharts BarChart or the existing
//    CSS bar style from the current cancer-types tab)
// 3. Show: breed name, case count, rate per 10k (if population data available)
// 4. Sort by count descending
```

## 7.3 CancerTypesChart Component

**File:** `frontend/src/components/CancerTypesChart/CancerTypesChart.tsx`

**Data source:** `GET /api/v1/incidence/by-cancer-type` (already exists, `fetchIncidenceByCancerType` already defined in client.ts)

**Implementation:**
```typescript
interface CancerTypesChartProps {
  filters: FilterState;
}

// 1. Call fetchIncidenceByCancerType with current filters
// 2. Render horizontal bars (keep existing visual style)
// 3. Show ICD-O code next to cancer type name (after Workstream 3)
// 4. Replace the Math.random() values with actual counts
```

## 7.4 RegionalComparison Component

**File:** `frontend/src/components/RegionalComparison/RegionalComparison.tsx`

**Data source:** The `regionSummary` object already computed in `useFilteredData.ts` contains real region-level aggregations.

**Implementation:**
```typescript
interface RegionalComparisonProps {
  regionSummary: RegionSummary;
  filters: FilterState;
}

// 1. Use the regionSummary data (already contains real counts + rates per region)
// 2. Fetch trend data (GET /api/v1/trends/yearly with county filter per region)
//    to determine actual trend direction (up/down/stable)
// 3. Replace hardcoded regions with actual regions from the data
// 4. Calculate trend: compare last 2 years of data, if increasing → "up", etc.
```

**Trend direction calculation:**
```typescript
function getTrendDirection(trendData: TrendPoint[]): 'up' | 'down' | 'stable' {
  if (trendData.length < 2) return 'stable';
  const recent = trendData.slice(-3);  // Last 3 years
  const earlier = trendData.slice(-6, -3);  // 3 years before that
  const recentAvg = recent.reduce((s, p) => s + p.count, 0) / recent.length;
  const earlierAvg = earlier.reduce((s, p) => s + p.count, 0) / earlier.length;
  const change = (recentAvg - earlierAvg) / earlierAvg;
  if (change > 0.05) return 'up';
  if (change < -0.05) return 'down';
  return 'stable';
}
```

## 7.5 Refactor App.tsx

Remove the inline hardcoded tab content from `App.tsx` and replace with component imports:

```tsx
// Before (App.tsx lines 65-153): inline JSX with Math.random()
// After:
{activeTab === 'breed-disparities' && <BreedDisparities filters={filters} />}
{activeTab === 'cancer-types' && <CancerTypesChart filters={filters} />}
{activeTab === 'regional-comparison' && (
  <RegionalComparison regionSummary={regionSummary} filters={filters} />
)}
```

## 7.6 Files Summary

**Files to create:**
| File | Purpose |
|------|---------|
| `frontend/src/components/BreedDisparities/BreedDisparities.tsx` | Real breed data |
| `frontend/src/components/CancerTypesChart/CancerTypesChart.tsx` | Real cancer type data |
| `frontend/src/components/RegionalComparison/RegionalComparison.tsx` | Real regional data |

**Files to modify:**
| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Replace inline hardcoded tabs with component imports |
| `frontend/src/api/client.ts` | Wire up existing `fetchIncidenceByBreed`, `fetchIncidenceByCancerType` |
| `frontend/src/components/index.ts` | Export new components |
