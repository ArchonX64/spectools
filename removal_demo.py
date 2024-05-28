from spectools import *


if __name__ == "__main__":
    # Set global variables
    ClusterSpectrum.min_freq = 2000
    ClusterSpectrum.max_freq = 8000

    # Load spectra into variables
    s0 = get_spectrum("No CO2.ft", "TFE", inten_thresh=4.5e-5, prominence=9e-5)
    s1 = get_spectrum("1% CO2.ft", "1%", inten_thresh=4.5e-5, prominence=9e-5)
    s2 = get_spectrum("2% CO2.ft", "2%", inten_thresh=4.5e-5, prominence=9e-5)
    asn1 = get_fit("ASN1.cat", "ASN1")
    asn2 = get_fit("ASN2.cat", "ASN2")
    asn3 = get_fit("ASN3.cat", "ASN3")
    asn4 = get_fit("ASN4.cat", "ASN4")
    asn5 = get_fit("ASN5.cat", "ASN5")
    asn6 = get_fit("ASN6.cat", "ASN6")
    asn7 = get_fit("ASN7.cat", "ASN7")
    asn8 = get_fit("ASN8.cat", "ASN8")
    ne_20 = get_fit("TFE_Ne_20.cat", "TFE_20")
    ne_22 = get_fit("TFE_Ne_22.cat", "TFE_22")

    # Remove peaks found in assignments
    s1.remove_peaks_of(other={asn1, asn2, asn3, asn4, asn5, asn6, asn7, asn8, ne_20, ne_22, s0.peaks},
                       freq_variability=0.050)
    s2.remove_peaks_of(other={asn1, asn2, asn3, asn4, asn5, asn6, asn7, asn8, ne_20, ne_22, s0.peaks},
                       freq_variability=0.050)

    # Plot cut spectra
    s1.plot()
    s2.plot()


    # Show graph
    show(interactive=False)
