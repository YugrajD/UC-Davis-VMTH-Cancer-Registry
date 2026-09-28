import { useEffect, useState } from 'react';

/**
 * useState backed by sessionStorage: value survives switching tabs within
 * the app and refreshing the page, but clears when the browser tab closes.
 * Falls back to defaultValue when storage is empty, unavailable (e.g.
 * private browsing), or holds unparseable JSON.
 */
export function useSessionStorageState<T>(key: string, defaultValue: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = sessionStorage.getItem(key);
      return stored === null ? defaultValue : (JSON.parse(stored) as T);
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      sessionStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Storage unavailable or full — persistence is a nice-to-have, not required.
    }
  }, [key, value]);

  return [value, setValue] as const;
}
