import { act, renderHook } from '@testing-library/react';
import { beforeEach, describe, expect, it } from 'vitest';
import { useSessionStorageState } from './useSessionStorageState';

describe('useSessionStorageState', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  it('returns the default value when nothing is stored', () => {
    const { result } = renderHook(() => useSessionStorageState('test.key', 'default'));
    expect(result.current[0]).toBe('default');
  });

  it('reads an existing stored value on init', () => {
    sessionStorage.setItem('test.key', JSON.stringify(['a', 'b']));
    const { result } = renderHook(() => useSessionStorageState<string[]>('test.key', []));
    expect(result.current[0]).toEqual(['a', 'b']);
  });

  it('persists updates to sessionStorage', () => {
    const { result } = renderHook(() => useSessionStorageState<string[]>('test.key', []));

    act(() => {
      result.current[1](['x', 'y']);
    });

    expect(JSON.parse(sessionStorage.getItem('test.key') ?? 'null')).toEqual(['x', 'y']);
  });

  it('persists null explicitly, distinct from an unset key', () => {
    const { result } = renderHook(() => useSessionStorageState<string[] | null>('test.key', null));

    act(() => {
      result.current[1](['x']);
    });
    act(() => {
      result.current[1](null);
    });

    expect(sessionStorage.getItem('test.key')).toBe('null');
  });

  it('falls back to the default when stored JSON is corrupted', () => {
    sessionStorage.setItem('test.key', '{not valid json');
    const { result } = renderHook(() => useSessionStorageState('test.key', 'fallback'));
    expect(result.current[0]).toBe('fallback');
  });

  it('keeps separate keys independent', () => {
    const { result: a } = renderHook(() => useSessionStorageState('key.a', 'a-default'));
    const { result: b } = renderHook(() => useSessionStorageState('key.b', 'b-default'));

    act(() => {
      a.current[1]('a-updated');
    });

    expect(a.current[0]).toBe('a-updated');
    expect(b.current[0]).toBe('b-default');
  });

  it('supports the functional updater form', () => {
    const { result } = renderHook(() => useSessionStorageState<string[]>('test.key', ['a']));

    act(() => {
      result.current[1]((prev) => [...prev, 'b']);
    });

    expect(result.current[0]).toEqual(['a', 'b']);
  });
});
