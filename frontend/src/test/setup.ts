import '@testing-library/jest-dom/vitest';
import { cleanup } from '@testing-library/react';
import { afterEach, beforeEach, vi } from 'vitest';

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
});

afterEach(() => {
  cleanup();
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
