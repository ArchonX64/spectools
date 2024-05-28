from spectools import *


if __name__ == "__main__":
    # Set global variables
    ClusterSpectrum.min_freq = 2000
    ClusterSpectrum.max_freq = 8000

    # Load spectra into variables
    s0 = get_spectrum("Example Spectrum1.ft", "1", inten_thresh=4.5e-5, prominence=9e-5)
    s1 = get_spectrum("Example Spectrum2.ft", "2", inten_thresh=4.5e-5, prominence=9e-5)
    s2 = get_spectrum("Example Spectrum 3.ft", "3", inten_thresh=4.5e-5, prominence=9e-5)
    asn1 = get_fit("Fit1.cat", "F1")
    asn2 = get_fit("Fit2.cat", "F2")
    asn3 = get_fit("Fit3.cat", "F3")

    # Remove peaks found in assignments
    s1.remove_peaks_of(other={asn1, asn2, asn3, s0.peaks}, freq_variability=0.050)
    s2.remove_peaks_of(other={asn1, asn2, asn3, s0.peaks}, freq_variability=0.050)

    # Plot cut spectra
    s1.plot()
    s2.plot()

    # Show graph
    show()
