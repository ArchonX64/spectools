from spectools import *

if __name__ == "__main__":
    # Load spectra
    s0 = get_spectrum("No CO2.ft", "TFE", inten_thresh=4.5e-5, prominence=9e-5)
    s2 = get_spectrum("2% CO2.ft", "2%", inten_thresh=4.5e-5, prominence=9e-5)
    s1 = get_spectrum("1% CO2.ft", "1%", inten_thresh=4.5e-5, prominence=9e-5)

    # Compute the ratios of 2% / 1%, and store the computation in a variable
    ratios = s2.divide_by(s1, freq_variability=0.010)

    # Keep the ratios within a specified range, using the specified ratio (2% / 1%)
    s2.keep_ratio_of(ratios, lbound=0.9, ubound=1.1)

    s2.plot()

    show()
