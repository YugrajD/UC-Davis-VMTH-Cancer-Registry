import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useLocalStorageState } from './useLocalStorageState';

describe('useLocalStorageState', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('returns the default value when nothing is stored', () => {
    const { result } = renderHook(() => useLocalStorageState('test.key', 'default'));
    expect(result.current[0]).toBe('default');
  });

  it('reads an existing stored value on init', () => {
    localStorage.setItem('test.key', JSON.stringify(['a', 'b']));
    const { result } = renderHook(() => useLocalStorageState<string[]>('test.key', []));
    expect(result.current[0]).toEqual(['a', 'b']);
  });

  it('persists updates to localStorage', () => {
    const { result } = renderHook(() => useLocalStorageState<string[]>('test.key', []));

    act(() => {
      result.current[1](['x', 'y']);
    });

    expect(JSON.parse(localStorage.getItem('test.key') ?? 'null')).toEqual(['x', 'y']);
  });

  it('falls back to the default when stored JSON is corrupted', () => {
    localStorage.setItem('test.key', '{not valid json');
    const { result } = renderHook(() => useLocalStorageState('test.key', 'fallback'));
    expect(result.current[0]).toBe('fallback');
  });

  it('keeps separate keys independent', () => {
    const { result: a } = renderHook(() => useLocalStorageState('key.a', 'a-default'));
    const { result: b } = renderHook(() => useLocalStorageState('key.b', 'b-default'));

    act(() => {
      a.current[1]('a-updated');
    });

    expect(a.current[0]).toBe('a-updated');
    expect(b.current[0]).toBe('b-default');
  });

  it('supports the functional updater form', () => {
    const { result } = renderHook(() => useLocalStorageState<string[]>('test.key', ['a']));

    act(() => {
      result.current[1]((prev) => [...prev, 'b']);
    });

    expect(result.current[0]).toEqual(['a', 'b']);
  });
});
