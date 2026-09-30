import { useId, useMemo, useRef, useState } from 'react';
import type { TaxonomyTermOut } from '../../api/client';
import {
  buildTaxonomyIndex,
  footerFor,
  LEVEL_LABELS,
  normalize,
  RESULT_CAP,
  searchCodes,
  searchTerms,
  type CodeEntry,
  type TaxonomyEntry,
  type TaxonomyIndex,
} from './taxonomyMatch';

type Box = 'code' | 'term';

interface PickOption {
  key: string;
  node: React.ReactNode;
  pick: () => void;
}

interface ListState {
  open: boolean;
  head?: string;
  footer?: string;
  empty?: string;
  options: PickOption[];
  active: number;
}

const CLOSED: ListState = { open: false, options: [], active: -1 };

function CodeRow({ c }: { c: CodeEntry }) {
  const more = c.terms.length > 1 ? ` +${c.terms.length - 1} term${c.terms.length > 2 ? 's' : ''}` : '';
  return (
    <>
      <span className="flex flex-wrap gap-x-2 gap-y-0.5 items-baseline">
        <span className="font-mono text-blue-700">{c.code}</span>
        <span>{c.preferred.term}</span>
        {more && <span className="text-[11px] border border-gray-300 rounded px-1 text-gray-500">{more.trim()}</span>}
      </span>
      <span className="text-xs text-gray-500">{c.group}</span>
    </>
  );
}

function TermRow({ e, fuzzy }: { e: TaxonomyEntry; fuzzy: boolean }) {
  return (
    <>
      <span className="flex flex-wrap gap-x-2 gap-y-0.5 items-baseline">
        <span>{e.term}</span>
        {e.level !== 0 && (
          <span className="text-[11px] border border-gray-300 rounded px-1 text-gray-500">{LEVEL_LABELS[e.level]}</span>
        )}
        <span className="font-mono text-blue-700">{e.code}</span>
        {fuzzy && (
          <span className="text-[11px] border border-amber-400 rounded px-1 text-amber-700">close match</span>
        )}
      </span>
      <span className="text-xs text-gray-500">{e.group}</span>
    </>
  );
}

export interface CodePickerCode {
  taxonomy_group: string;
  taxonomy_term: string;
}

export function CodePicker({
  terms,
  codes,
  onAdd,
}: {
  terms: TaxonomyTermOut[];
  codes: CodePickerCode[];
  onAdd: (group: string, term: string) => void;
}) {
  const index: TaxonomyIndex = useMemo(() => buildTaxonomyIndex(terms), [terms]);

  const [codeText, setCodeText] = useState('');
  const [termText, setTermText] = useState('');
  const [current, setCurrent] = useState<TaxonomyEntry | null>(null);
  const [lastPair, setLastPair] = useState<TaxonomyEntry | null>(null);
  const [autofilled, setAutofilled] = useState(false);
  const [codeError, setCodeError] = useState('');
  const [termError, setTermError] = useState('');
  const [addedMessage, setAddedMessage] = useState('');
  const [codeList, setCodeList] = useState<ListState>(CLOSED);
  const [termList, setTermList] = useState<ListState>(CLOSED);

  const codeInputRef = useRef<HTMLInputElement>(null);
  const idPrefix = useId();

  const listFor = (box: Box) => (box === 'code' ? codeList : termList);
  const setListFor = (box: Box) => (box === 'code' ? setCodeList : setTermList);

  function closeLists() {
    setCodeList(CLOSED);
    setTermList(CLOSED);
  }

  function clearErrors() {
    setCodeError('');
    setTermError('');
  }

  function setPair(e: TaxonomyEntry, isAutofill: boolean) {
    setCurrent(e);
    setLastPair(e);
    setAutofilled(isAutofill);
    setCodeText(e.code);
    setTermText(e.term);
    setAddedMessage('');
    clearErrors();
    closeLists();
  }

  function breakPair(other: Box) {
    if (current) {
      if (other === 'code') setCodeText('');
      else setTermText('');
    }
    setCurrent(null);
    setAutofilled(false);
    setAddedMessage('');
    clearErrors();
  }

  function pickCode(c: CodeEntry) {
    setPair(c.preferred, c.terms.length > 1);
  }

  function showCodeResults(text: string) {
    if (/[a-z]/i.test(text)) {
      setCodeList({
        open: true,
        options: [],
        empty: 'Codes are numbers like 8010/3. To search by name, use the Term box.',
        active: -1,
      });
      return;
    }
    if (!text.trim()) {
      setCodeList(CLOSED);
      return;
    }
    const { hits, total } = searchCodes(index, text);
    const shown = hits.slice(0, RESULT_CAP);
    setCodeList({
      open: true,
      options: shown.map((c) => ({ key: c.code, node: <CodeRow c={c} />, pick: () => pickCode(c) })),
      footer: footerFor(total),
      empty: 'No matches',
      active: shown.length ? 0 : -1,
    });
  }

  function showTermResults(text: string) {
    if (!text.trim()) {
      setTermList(CLOSED);
      return;
    }
    const { hits, total } = searchTerms(index, text);
    const shown = hits.slice(0, RESULT_CAP);
    setTermList({
      open: true,
      options: shown.map((h) => ({
        key: `${h.entry.code}::${h.entry.term}`,
        node: <TermRow e={h.entry} fuzzy={h.fuzzy} />,
        pick: () => setPair(h.entry, false),
      })),
      footer: footerFor(total),
      empty: 'No matches',
      active: shown.length ? 0 : -1,
    });
  }

  function showSiblings(pair: TaxonomyEntry) {
    const siblings = index.byCode.get(pair.code) ?? [pair];
    setTermList({
      open: true,
      head: `Terms under ${pair.code}`,
      options: siblings.map((e) => ({
        key: `${e.code}::${e.term}`,
        node: <TermRow e={e} fuzzy={false} />,
        pick: () => setPair(e, false),
      })),
      active: siblings.findIndex((e) => e.term === pair.term && e.code === pair.code),
    });
  }

  function commitOnLeave(box: Box) {
    if (current) return;
    const raw = (box === 'code' ? codeText : termText).trim();
    if (!raw) return;
    if (box === 'code') {
      const qd = raw.replace(/\D/g, '');
      const exact = index.codes.filter((c) => c.digits === qd);
      if (!/[a-z]/i.test(raw) && exact.length === 1) {
        pickCode(exact[0]);
        return;
      }
      setCodeError(`No code matches ‘${raw}’. Pick one from the list.`);
    } else {
      const qn = normalize(raw);
      const exact = index.entries.filter((e) => e.norm === qn);
      if (exact.length === 1) {
        setPair(exact[0], false);
        return;
      }
      if (exact.length > 1) {
        setTermError(
          `${exact[0].term} is listed under ${exact.length} codes (${exact.map((e) => e.code).join(', ')}). Pick one from the list.`,
        );
        return;
      }
      setTermError(`No term matches ‘${raw}’. Pick one from the list.`);
    }
  }

  function handleCodeChange(value: string) {
    setCodeText(value);
    breakPair('term');
    showCodeResults(value);
  }

  function handleTermChange(value: string) {
    setTermText(value);
    breakPair('code');
    showTermResults(value);
  }

  function handleCodeFocus() {
    if (!current && codeText) showCodeResults(codeText);
  }

  function handleTermFocus() {
    const siblings = current ? index.byCode.get(current.code) : null;
    if (current && termText === current.term && siblings && siblings.length > 1) {
      showSiblings(current);
    } else if (!current && termText) {
      showTermResults(termText);
    }
  }

  function handleKeyDown(box: Box, ev: React.KeyboardEvent<HTMLInputElement>) {
    const list = listFor(box);
    const setList = setListFor(box);
    const isOpen = list.open && list.options.length > 0;
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      if (!isOpen) return;
      ev.preventDefault();
      setList((prev) => ({
        ...prev,
        active: (prev.active + (ev.key === 'ArrowDown' ? 1 : -1) + prev.options.length) % prev.options.length,
      }));
    } else if (ev.key === 'Enter') {
      ev.preventDefault();
      if (isOpen && list.active >= 0) list.options[list.active].pick();
      else commitOnLeave(box);
    } else if (ev.key === 'Tab') {
      if (isOpen && list.options.length === 1) list.options[0].pick();
    } else if (ev.key === 'Escape') {
      closeLists();
      if (!current && lastPair) setPair(lastPair, false);
    }
  }

  function handleBlur(box: Box) {
    closeLists();
    commitOnLeave(box);
  }

  function handleAdd() {
    if (!current) return;
    const dup = codes.some((c) => c.taxonomy_group === current.group && c.taxonomy_term === current.term);
    if (dup) {
      setAddedMessage(`Already added: ${current.code} ${current.term}`);
      return;
    }
    onAdd(current.group, current.term);
    setCurrent(null);
    setLastPair(null);
    setAutofilled(false);
    setCodeText('');
    setTermText('');
    setAddedMessage('');
    closeLists();
    codeInputRef.current?.focus();
  }

  const others = current ? (index.byCode.get(current.code)?.length ?? 1) - 1 : 0;
  const loading = terms.length === 0;

  function renderListbox(box: Box) {
    const list = listFor(box);
    if (!list.open) return null;
    const hasContent = list.options.length > 0 || !!list.empty || !!list.head;
    if (!hasContent) return null;
    return (
      <ul
        id={`${idPrefix}-${box}-list`}
        role="listbox"
        className="absolute z-10 mt-1 w-full min-w-[320px] max-h-72 overflow-auto bg-white border border-gray-200 rounded shadow-lg"
        onMouseDown={(ev) => {
          // Keep focus in the input on a click anywhere in the list — the
          // scrollbar, the footer/head note, or a row — rather than only
          // preventing default on rows, which let a scrollbar drag blur
          // the input and run the leave-box validation mid-browse.
          ev.preventDefault();
          const li = (ev.target as HTMLElement).closest('[role="option"]');
          if (!li) return;
          const i = list.options.findIndex((opt) => opt.key === li.getAttribute('data-key'));
          if (i >= 0) list.options[i].pick();
        }}
      >
        {list.head && (
          <li role="presentation" className="px-2 pt-1.5 pb-0.5 text-[11px] uppercase tracking-wide text-gray-400">
            {list.head}
          </li>
        )}
        {list.options.map((opt, i) => (
          <li
            key={opt.key}
            id={`${idPrefix}-${box}-opt-${i}`}
            data-key={opt.key}
            role="option"
            aria-selected={i === list.active}
            ref={(el) => {
              if (i === list.active) el?.scrollIntoView?.({ block: 'nearest' });
            }}
            className={`px-2 py-1.5 text-sm cursor-pointer grid gap-0.5 ${i === list.active ? 'bg-blue-50' : ''}`}
          >
            {opt.node}
          </li>
        ))}
        {list.options.length === 0 && list.empty && (
          <li role="presentation" className="px-2 py-1.5 text-sm text-gray-500">
            {list.empty}
          </li>
        )}
        {list.footer && (
          <li role="presentation" className="px-2 py-1.5 text-xs text-gray-500">
            {list.footer}
          </li>
        )}
      </ul>
    );
  }

  return (
    <div className="space-y-1">
      <div className="grid grid-cols-[minmax(0,140px)_minmax(0,1fr)_auto] gap-2 items-start">
        <div className="relative">
          <input
            ref={codeInputRef}
            type="text"
            role="combobox"
            aria-expanded={codeList.open}
            aria-controls={`${idPrefix}-code-list`}
            aria-autocomplete="list"
            aria-activedescendant={codeList.active >= 0 ? `${idPrefix}-code-opt-${codeList.active}` : undefined}
            placeholder={loading ? 'Loading taxonomy…' : 'ICD-O code'}
            aria-label="ICD-O code"
            disabled={loading}
            autoComplete="off"
            spellCheck={false}
            value={codeText}
            onChange={(e) => handleCodeChange(e.target.value)}
            onFocus={handleCodeFocus}
            onBlur={() => handleBlur('code')}
            onKeyDown={(e) => handleKeyDown('code', e)}
            className={`w-full px-2 py-1.5 border rounded text-sm font-mono disabled:opacity-50 ${
              codeError ? 'border-red-400 bg-red-50' : 'border-gray-300'
            }`}
          />
          {renderListbox('code')}
          {codeError && (
            <p role="alert" className="text-xs text-red-600 mt-0.5">
              {codeError}
            </p>
          )}
        </div>

        <div className="relative">
          <input
            type="text"
            role="combobox"
            aria-expanded={termList.open}
            aria-controls={`${idPrefix}-term-list`}
            aria-autocomplete="list"
            aria-activedescendant={termList.active >= 0 ? `${idPrefix}-term-opt-${termList.active}` : undefined}
            placeholder={loading ? 'Loading taxonomy…' : 'Term'}
            aria-label="Term"
            disabled={loading}
            autoComplete="off"
            spellCheck={false}
            value={termText}
            onChange={(e) => handleTermChange(e.target.value)}
            onFocus={handleTermFocus}
            onBlur={() => handleBlur('term')}
            onKeyDown={(e) => handleKeyDown('term', e)}
            className={`w-full px-2 py-1.5 border rounded text-sm disabled:opacity-50 ${
              termError ? 'border-red-400 bg-red-50' : 'border-gray-300'
            }`}
          />
          {renderListbox('term')}
          {termError && (
            <p role="alert" className="text-xs text-red-600 mt-0.5">
              {termError}
            </p>
          )}
        </div>

        <button
          type="button"
          onClick={handleAdd}
          disabled={!current}
          className="px-3 py-1.5 text-sm font-medium bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-40 self-start"
        >
          Add
        </button>
      </div>

      <div className="text-xs text-gray-500 min-h-[1.4em]" aria-live="polite">
        {addedMessage ? (
          <span className="text-amber-700">{addedMessage}</span>
        ) : current ? (
          <>
            Group: <b className="text-gray-700">{current.group}</b> · {LEVEL_LABELS[current.level]} term
            {autofilled && others > 0 && (
              <div className="text-amber-700">
                {others} other term{others > 1 ? 's use' : ' uses'} {current.code}. Click the Term box to switch.
              </div>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
