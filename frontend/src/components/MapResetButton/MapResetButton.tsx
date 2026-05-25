/**
 * Small overlay button anchored to the bottom-right of a DeckGL map.
 * Clicking it snaps the camera back to its default view.
 *
 * Wrapped in a `group` so the hover tooltip shows even when the button is
 * disabled (most browsers suppress native `title` tooltips on disabled
 * elements).  `pointer-events-none` on the tooltip prevents it from
 * blocking clicks on the button beneath.
 */

interface MapResetButtonProps {
  onClick: () => void;
  /** True when the camera is already at its default — visually mutes the button. */
  disabled: boolean;
}

export function MapResetButton({ onClick, disabled }: MapResetButtonProps) {
  return (
    <div className="absolute bottom-4 right-4 z-10 group">
      <button
        type="button"
        onClick={onClick}
        disabled={disabled}
        aria-label="Reset to default view"
        className="inline-flex items-center justify-center w-8 h-8 rounded-md bg-white/95 backdrop-blur-sm border border-gray-200 shadow-sm text-[var(--color-text-primary)] hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-[var(--color-teal)] focus:border-transparent disabled:opacity-40 disabled:cursor-not-allowed transition-opacity"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 4v6h6M20 20v-6h-6M20 8A8 8 0 006.7 5.3L4 8m16 8a8 8 0 01-13.3 2.7L4 16"
          />
        </svg>
      </button>
      <div
        role="tooltip"
        className="absolute bottom-full right-0 mb-1.5 px-2 py-1 rounded-md bg-gray-900 text-white text-xs whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none shadow-md"
      >
        Reset to default view
      </div>
    </div>
  );
}
