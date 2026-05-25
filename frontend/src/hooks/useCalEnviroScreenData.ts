import type { CalEnviroScreenData } from '../types';
import { MOCK_CALENVIROSCREEN_DATA } from '../data/mockData';

export function useCalEnviroScreenData(): { data: CalEnviroScreenData[]; loading: boolean; error: string | null } {
  return { data: MOCK_CALENVIROSCREEN_DATA, loading: false, error: null };
}
