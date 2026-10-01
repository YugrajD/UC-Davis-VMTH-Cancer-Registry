import { useEffect, useState } from 'react';

/**
 * useState backed by localStorage: value persists across browser sessions
 * (closing the tab, restarting the browser) until explicitly changed.
 * Falls back to defaultValue when storage is empty, unavailable (e.g.
 * private browsing), or holds unparseable JSON.
 */
export function useLocalStorageState<T>(key: string, defaultValue: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored === null ? defaultValue : (JSON.parse(stored) as T);
    } catch {
      return defaultValue;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Storage unavailable or full — persistence is a nice-to-have, not required.
    }
  }, [key, value]);

  return [value, setValue] as const;
}
