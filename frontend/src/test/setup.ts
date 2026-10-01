import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';

// Node's native localStorage global (unlike sessionStorage, which it keeps
// in-memory) requires a --localstorage-file backing path to actually work;
// without one, setItem/getItem/clear are all missing. Stub it with a real
// in-memory Storage so tests don't depend on that flag or touch disk.
class MemoryStorage {
  private store = new Map<string, string>();
  getItem(key: string) {
    return this.store.has(key) ? this.store.get(key)! : null;
  }
  setItem(key: string, value: string) {
    this.store.set(key, String(value));
  }
  removeItem(key: string) {
    this.store.delete(key);
  }
  clear() {
    this.store.clear();
  }
}

class MockFileReader {
  result: string | ArrayBuffer | null = null;
  error: DOMException | null = null;
  onload: ((event: ProgressEvent<FileReader>) => void) | null = null;
  onerror: ((event: ProgressEvent<FileReader>) => void) | null = null;

  readAsText(file: Blob) {
    file.text()
      .then((text) => {
        this.result = text;
        this.onload?.({ target: this } as unknown as ProgressEvent<FileReader>);
      })
      .catch((error: unknown) => {
        this.error = error instanceof DOMException ? error : new DOMException('Read failed');
        this.onerror?.({ target: this } as unknown as ProgressEvent<FileReader>);
      });
  }
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn());
  vi.stubGlobal('confirm', vi.fn(() => true));
  vi.stubGlobal('alert', vi.fn());
  vi.stubGlobal('FileReader', MockFileReader);
  vi.stubGlobal('localStorage', new MemoryStorage());
});

afterEach(() => {
  cleanup();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
