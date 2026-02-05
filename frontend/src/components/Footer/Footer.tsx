interface FooterProps {
  onShare?: () => void;
}

export function Footer({ onShare }: FooterProps) {
  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: 'California Canine Cancer Registry Dashboard',
        text: 'Explore population-based cancer incidence data for dogs in California',
        url: window.location.href,
      });
    } else if (onShare) {
      onShare();
    } else {
      navigator.clipboard.writeText(window.location.href);
      alert('Link copied to clipboard!');
    }
  };

  return (
    <footer className="bg-gray-50 border-t border-gray-200 mt-8">
      <div className="max-w-[1400px] mx-auto px-6 py-6">
        {/* Methodology note */}
        <div className="bg-white rounded-lg border border-gray-200 p-4 mb-6">
          <h4 className="text-sm font-semibold text-[var(--color-text-primary)] mb-2">
            Methodology &amp; Data Sources
          </h4>
          <div className="text-xs text-[var(--color-text-secondary)] space-y-2">
            <p>
              This dashboard presents population-based cancer incidence and mortality data for dogs 
              across California counties. Data is collected from participating veterinary hospitals, 
              animal cancer registries, and partner institutions.
            </p>
            <p>
              <strong>Rate Calculation:</strong> Age-adjusted rates are calculated per 10,000 dog 
              population using the direct method with the 2020 California dog population as the 
              standard. Rates based on fewer than 16 cases are suppressed due to statistical instability.
            </p>
            <p>
              <strong>Data Suppression:</strong> County-level data may be suppressed when case counts 
              are too small to report reliably or when reporting could potentially identify individual 
              animals or owners.
            </p>
            <p className="text-[var(--color-teal)] italic">
              Note: This is a demonstration dashboard with simulated data for educational purposes.
            </p>
          </div>
        </div>

        {/* Actions and credits */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div className="text-xs text-[var(--color-text-secondary)]">
            <p>
              © 2024 UC Davis Veterinary Medical Teaching Hospital. All rights reserved.
            </p>
            <p className="mt-1">
              For questions, contact:{' '}
              <a 
                href="mailto:vmthcancerregistry@ucdavis.edu" 
                className="text-[var(--color-teal)] hover:underline"
              >
                vmthcancerregistry@ucdavis.edu
              </a>
            </p>
          </div>
          
          <div className="flex items-center gap-3">
            <button
              onClick={handleShare}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-white 
                         bg-[var(--color-primary-orange)] hover:bg-[var(--color-primary-orange-dark)] 
                         rounded-md transition-colors duration-150"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
              </svg>
              Share Dashboard
            </button>
            
            <button
              onClick={() => window.print()}
              className="flex items-center gap-2 px-4 py-2 text-sm font-medium 
                         text-[var(--color-teal)] border border-[var(--color-teal)]
                         hover:bg-[var(--color-teal)] hover:text-white
                         rounded-md transition-colors duration-150"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M17 17h2a2 2 0 002-2v-4a2 2 0 00-2-2H5a2 2 0 00-2 2v4a2 2 0 002 2h2m2 4h6a2 2 0 002-2v-4a2 2 0 00-2-2H9a2 2 0 00-2 2v4a2 2 0 002 2zm8-12V5a2 2 0 00-2-2H9a2 2 0 00-2 2v4h10z" />
              </svg>
              Print
            </button>
          </div>
        </div>
      </div>
      
      {/* Bottom bar */}
      <div className="bg-[#052049] text-white py-3 px-6">
        <div className="max-w-[1400px] mx-auto flex flex-col md:flex-row justify-between items-center gap-2 text-xs">
          <div className="flex items-center gap-4">
            <a href="#" className="hover:underline">Privacy Policy</a>
            <span className="opacity-50">|</span>
            <a href="#" className="hover:underline">Terms of Use</a>
            <span className="opacity-50">|</span>
            <a href="#" className="hover:underline">Accessibility</a>
          </div>
          <div className="opacity-70">
            Dashboard version 1.0.0
          </div>
        </div>
      </div>
    </footer>
  );
}
