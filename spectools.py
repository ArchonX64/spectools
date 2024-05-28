from __future__ import annotations

from typing import Union

import numpy as np
import matplotlib as mpl
import pandas as pd

import scipy.signal as sp_sig

import matplotlib.pyplot as plt

mpl.use("QtAgg")

fig = plt.figure()
ax = fig.add_subplot()


class Spectrum:
    """
        The abstract object of a spectrum. Only guarantees a name, a method that returns an array of frequencies, and
        a method that returns an array of intensities. The two methods should return concurrent arrays.
        :ivar name: A string containing a simple name for the spectrum
    """
    def __init__(self, name: str):
        self.name = name

    def get_frequencies(self) -> np.ndarray:
        """
        An abstract method for obtaining frequencies of a spectrum. Must be implemented manually
        """
        raise TypeError("The method get_frequencies needs to be overridden in class" + self.__name__)

    def get_intensities(self) -> np.ndarray:
        """
                An abstract method for obtaining intensities of a spectrum. Must be implemented manually
                """
        raise TypeError("The method get_intensities needs to be overridden in class" + self.__name__)


class Ratio:
    """
        An object containing ratios of a spectrum that correspond to an index in a peak list. The internal DataFrame
        contains three columns:
            p_inds: The indexes of the corresponding parent spectrum\n
            a_inds: The indexes of the corresponding spectrum parent was divided against\n
            ratios: The value of the calculated ratio
        :ivar data: A DataFrame containing the quantitative data
        :ivar parent: The spectrum that a ratio was generated for (i.e. divide_by was called on it)
        :ivar against: The spectrum that parent was divided against
        :ivar freq_var: The frequency variability that was used in creating the ratios


    """
    def __init__(self, parent: ExperimentalSpectrum, against: ExperimentalSpectrum, freq_var: float,
                 ratios: np.ndarray = None, parent_inds: np.ndarray = None, against_inds: np.ndarray = None,
                 df: pd.DataFrame = None):
        self.data = pd.DataFrame({
            "p_inds": parent_inds,
            "a_inds": against_inds,
            "ratios": ratios
        }) if df is None else df
        self.parent: ExperimentalSpectrum = parent
        self.against: ExperimentalSpectrum = against
        self.freq_var: float = freq_var

    def copy(self, new_owner: ExperimentalSpectrum):
        """
        Generates a new ratio object with the data copied. Usually called when a spectrum is being copied, so
        the parent may change.
        :param new_owner: The parent spectrum that this object will have
        :return: A copy of this ratio object
        """
        return Ratio(parent=new_owner, against=self.against, df=self.data.copy(True), freq_var=self.freq_var)
    

class ExperimentalSpectrum(Spectrum):
    """
    Represents a spectrum that was recorded experimentally. Will have a large amounts of values that may or may not be
    of value, and must be sorted.
    :ivar dataframe: A DataFrame containing the frequencies ("freq") and intensities (the name of the spectrum) of the
    spectrum
    :ivar freqs: Easy access of "freq" column of dataframe
    :ivar inten: Easy access of "inten" column of dataframe
    :ivar baseline: The minimum value that should be considered when looking at this spectrum. Anything below this value
    is considered noise
    :ivar peaks: An object containing the calculated peaks of this spectrum
    :ivar ratios: DEPRECATED
    """
    # Do not call without filling in peaks!
    def __init__(self, name: str, freq: np.ndarray, inten: np.ndarray, baseline: float = 0, ratios: list[Ratio] = None):
        super().__init__(name)
        self.dataframe = pd.DataFrame({"freq": freq, name: inten})
        self.freqs: pd.Series = self.dataframe["freq"]
        self.inten: pd.Series = self.dataframe[name]
        self.baseline: float = baseline
        self.peaks: Union[DeterminedPeaks, None] = None  # Will ONLY be None until get_spectrum produces peaks
        self.ratios: list[Ratio] = [] if ratios is None else ratios

    def __add__(self, other) -> SpectrumCollection:
        if isinstance(other, ExperimentalSpectrum):
            return SpectrumCollection([self, other])

    def plot_baseline(self) -> None:
        """
        Plot a horizontal line to v
        """
        plt.axhline(y=self.baseline)

    # Peaks is the INDEX in peak_inds
    def remove_peaks(self, peaks: np.ndarray):
        # Set the width of each peak equal to the baseline, creating horizontal lines where the peak was.
        for peak in peaks:
            self.inten[self.peaks.left_bases[peak]:self.peaks.right_bases[peak]] = self.baseline
        self.peaks.remove_peaks(peaks)

        # Ratios depend on indexes of peaks, so they must be recalculated
        old_rats = self.peaks.ratios.copy()  # Save old ratios so that the spectrum they were calculated against is known
        self.peaks.ratios = []  # Clear old ratios
        for ratio in old_rats:
            self.divide_by(ratio.against, ratio.freq_var)

    def remove_peaks_of(self, other: Union[SpectrumPeaks, ExperimentalSpectrum, set[SpectrumPeaks]],
                        freq_variability: float):
        if isinstance(other, SpectrumPeaks):
            self._remove_peaks_of(other, freq_variability)
        elif isinstance(other, ExperimentalSpectrum):
            self._remove_peaks_of(other.peaks, freq_variability)
        elif isinstance(other, set):
            for each in other:
                self._remove_peaks_of(each, freq_variability)

    def _remove_peaks_of(self, other: SpectrumPeaks, freq_variability: float):
        self_inds, other_inds = self.same_peaks_as(other, freq_variability)
        self.remove_peaks(np.unique(self_inds))

    def same_peaks_as(self, other: Union[SpectrumPeaks, ExperimentalSpectrum, set[SpectrumPeaks]],
                      freq_variability: float, unique: bool = False) -> (np.ndarray, np.ndarray):
        if isinstance(other, SpectrumPeaks):
            return self._same_peaks_as(other, freq_variability, unique)
        elif isinstance(other, ExperimentalSpectrum):
            return self._same_peaks_as(other.peaks, freq_variability, unique)
        elif isinstance(other, set):
            for each in other:
                return self._same_peaks_as(each, freq_variability, unique)

    # Returns: Indexes of peak_inds
    def _same_peaks_as(self, other: SpectrumPeaks, freq_variability: float, unique: bool) -> (np.ndarray, np.ndarray):
        self_freqs = self.peaks.get_frequencies()
        other_freqs = other.get_frequencies()

        # Indexes now contains locations closest to other peaks
        # Each index i represents an index of other_freqs, and indexes[i] represents index of self_freqs
        indexes = np.searchsorted(self_freqs, other_freqs)

        # Iteratively check each peak of other against the closest frequencies in self to determine whether
        #   they are the "same" (are within freq_variability), and add them to a list
        # np.searchsorted give the indexes of where each peak in other would land within self's peaks, so the peaks
        #   that are before and after need be checked if they are within freq_variability. Additionally, if two peaks
        #   are within the allowed variance, the peak with the least variance is used.
        self_inds = np.ndarray(shape=len(indexes))
        other_inds = np.ndarray(shape=len(indexes))
        freq_len = len(self_freqs)
        counter = 0

        for i in range(0, len(indexes)):
            # Prevent 0 or max indexing, as values that are too low/high will be placed there
            if indexes[i] == 0 or indexes[i] >= freq_len:
                continue
            remove_next = None
            has_after = False

            after_variance = self_freqs[indexes[i]] - other_freqs[i]
            on_variance = other_freqs[i] - self_freqs[indexes[i] - 1]

            if after_variance < freq_variability:
                remove_next = indexes[i]
                has_after = True
            if on_variance < freq_variability:
                if not has_after:
                    remove_next = indexes[i] - 1
                elif on_variance < after_variance:
                    remove_next = indexes[i]

            if remove_next is not None:
                self_inds[counter] = remove_next
                other_inds[counter] = i
                counter += 1

        # Shave off unused space
        self_inds = self_inds[:counter - 1]
        other_inds = other_inds[:counter - 1]

        if unique:
            other_inds, indices = np.unique(other_inds, return_index=True)
            self_inds = self_inds[indices]

        # Convert back to numpy array
        return np.asarray(self_inds, dtype="int"), np.asarray(other_inds, dtype="int")

    def divide_by(self, other: ExperimentalSpectrum, freq_variability) -> Ratio:
        """
            Generate ratios of the intensities of one spectrum divided by another\n
            BOTH SPECTRA MUST HAVE THE SAME FREQUENCY AT EACH INDEX
            :param other: The spectrum to be divided against
            :param freq_variability: The maximum frequency (in MHz) that peaks can be separated by for matching
            :return: An object containing the calculated ratios based on peak indexes, along with other information
            """
        ratio_object = self._divide_by(other, freq_variability)
        self.peaks.ratios.append(ratio_object)
        return ratio_object

    def _divide_by(self, other: ExperimentalSpectrum, freq_variability) -> Ratio:
        # Check if a ratio calculation already exists
        for ratio in self.peaks.ratios:
            if ratio.against == other:
                return ratio

        # If ratio has not yet been calculated,
        # Figure out which peaks are matching in each spectrum
        self_inds, other_inds = self.same_peaks_as(other, freq_variability)

        # Divide each matching point by each other to generate a ratio for each
        ratios = self.peaks.get_intensities()[self_inds] / other.peaks.get_intensities()[other_inds]

        ratio_object = Ratio(parent=self, against=other, ratios=ratios, parent_inds=self_inds,
                             against_inds=other_inds, freq_var=freq_variability)
        return ratio_object

    def keep_ratio_of(self, ratio: Ratio, lbound: float = 0, ubound: float = np.inf):
        # Remove all values that do not have a ratio
        non_ratio = np.arange(len(self.peaks.peak_inds))
        non_ratio = np.delete(non_ratio, ratio.data["p_inds"])
        self.remove_peaks(non_ratio)

        ratio = self.divide_by(ratio.against, ratio.freq_var)
        inds = ratio.data[(ratio.data["ratios"] > ubound) | (ratio.data["ratios"] < lbound)]["p_inds"].to_numpy()
        self.remove_peaks(np.asarray(inds, dtype="int"))

    def plot(self) -> None:
        plt.plot(self.freqs, self.inten, label=self.name)
        plt.axhline(self.baseline, ls="--")

    def export(self, out_type: str, name: str = None) -> None:
        name = name if name is not None else self.name + "_data"

        match out_type:
            case "csv":
                self.dataframe.to_csv(name + ".csv")
            case "txt":
                self.dataframe.to_csv(name + ".txt", sep=" ", header=False, index=False)
            case "ft":
                self.dataframe.to_csv(name + ".ft", sep=" ", header=False, index=False)
            case _:
                raise ValueError("Unsupported export type for spectrum: " + self.name)

    def get_frequencies(self) -> np.ndarray:
        return self.dataframe["freqs"]

    def get_intensities(self) -> np.ndarray:
        return self.dataframe[self.name]

    def copy(self) -> ExperimentalSpectrum:
        exp = ExperimentalSpectrum(
            name=self.name,
            freq=self.get_frequencies().copy(),
            inten=self.get_intensities().copy(),
            baseline=self.baseline,
        )
        exp.peaks = self.peaks.copy(self)
        return exp


class SpectrumPeaks(Spectrum):
    peak_color_index = 1

    def plot(self):
        plt.stem(self.get_frequencies(), self.get_intensities(), label=self.name, markerfmt=" ",
                 linefmt="C" + str(SpectrumPeaks.peak_color_index))
        SpectrumPeaks.peak_color_index += 1


class DeterminedPeaks(SpectrumPeaks):
    def __init__(self, spectrum: ExperimentalSpectrum, inten_thresh: float = None, prominence: float = None,
                 wlen: float = None,
                 peaks: np.ndarray = None, left_bases: np.ndarray = None, right_bases: np.ndarray = None):
        super().__init__(name=spectrum.name + " (peaks)")

        if prominence is not None:
            (peaks, properties) = sp_sig.find_peaks(spectrum.inten, height=inten_thresh, prominence=prominence)
            (prominences, left_bases, right_bases) = sp_sig.peak_prominences(spectrum.inten, peaks, wlen=wlen)

        self.spectrum = spectrum
        self.peak_inds = peaks
        self.left_bases = left_bases
        self.right_bases = right_bases
        self.ratios: list[Ratio] = []

    def remove_peaks(self, peaks: np.ndarray):
        self.peak_inds = np.delete(self.peak_inds, peaks)
        self.left_bases = np.delete(self.left_bases, peaks)
        self.right_bases = np.delete(self.right_bases, peaks)

    def get_frequencies(self) -> np.ndarray:
        return self.spectrum.freqs[self.peak_inds].to_numpy()

    def get_intensities(self) -> np.ndarray:
        return self.spectrum.inten[self.peak_inds].to_numpy()

    def copy(self, new_parent: ExperimentalSpectrum) -> DeterminedPeaks:
        det = DeterminedPeaks(
            spectrum=new_parent,
            peaks=self.peak_inds.copy(),
            left_bases=self.left_bases.copy(),
            right_bases=self.right_bases.copy(),
        )
        det.ratios = [ratio.copy(new_parent) for ratio in self.ratios]
        return det

    def export(self, ext: str, name: str = None):
        name = name if name is not None else self.name + "_data"

        df = pd.DataFrame({
            "Frequency (MHz)": self.get_frequencies(),
            "Intensity (V)": self.get_intensities(),
        })

        for ratio in self.ratios:
            ratio_series = np.ndarray(shape=len(self.get_intensities()))
            ratio_series[:] = np.nan
            ratio_series[ratio.data["p_inds"]] = ratio.data["ratios"]
            df = pd.concat([df, pd.DataFrame({ratio.parent.name + " / " + ratio.against.name: ratio_series})], axis=1)

        match ext:
            case "csv":
                df.to_csv(name + ".csv", index=False)
            case "txt":
                df.to_csv(name + ".txt", sep=" ", header=False, index=False)
            case "ft":
                df.to_csv(name + ".ft", sep=" ", header=False, index=False)
            case _:
                raise ValueError("Unsupported export type for spectrum: " + self.name)


class ClusterSpectrum(SpectrumPeaks):
    min_freq = 0
    max_freq = None

    def __init__(self, name: str, freq: np.ndarray, rel_inten: np.ndarray):
        super().__init__(name=name)
        self.freq: np.ndarray = freq
        self.rel_inten: np.ndarray = rel_inten

    def plot(self):
        plt.stem(self.freq, self.rel_inten, label=self.name, markerfmt=" ")

    def get_frequencies(self) -> np.ndarray:
        return self.freq

    def get_intensities(self) -> np.ndarray:
        return self.rel_inten


class SpectrumCollection:
    class Ratio:
        def __init__(self, against: ExperimentalSpectrum, held: list[ExperimentalSpectrum]):
            self.against: ExperimentalSpectrum = against
            self.held: list[ExperimentalSpectrum] = held
            self.dataframe = pd.DataFrame()

    def __init__(self, spectrum_list: list[ExperimentalSpectrum]):
        if len(spectrum_list) == 0:
            raise ValueError("SpectrumCollection list argument must contain at least one value!")

        self.spectrum_list = spectrum_list.copy()
        self.ratios: list[SpectrumCollection.Ratio] = []

    def __add__(self, other) -> Union[ExperimentalSpectrum, SpectrumCollection]:
        if isinstance(other, ExperimentalSpectrum):
            return SpectrumCollection(self.spectrum_list + [other])
        elif isinstance(other, SpectrumCollection):
            return SpectrumCollection(self.spectrum_list + other.spectrum_list)

    def remove_peaks_of(self, others: set[SpectrumPeaks], freq_variability: float):
        for spectrum in self.spectrum_list:
            spectrum.remove_peaks_of(others, freq_variability)

    def create_ratios(self, against: ExperimentalSpectrum):
        pass

    def plot(self) -> None:
        for spectrum in self.spectrum_list:
            spectrum.plot()


def on_click(event):
    for text in fig.texts:
        text.remove()

    fig.text(0.5, 1.10, "Click At: {x:.4f}, {y:.4f}".format(x=event.xdata, y=event.ydata), verticalalignment="top",
             horizontalalignment="center", transform=ax.transAxes, fontsize=10)


def show(interactive: bool = False):
    if interactive:
        plt.connect("button_press_event", on_click)

    plt.legend(loc=2)
    plt.show()


def get_spectrum(path: str, name: str, inten_thresh: float, prominence: float,
                 wlen: int = None) -> ExperimentalSpectrum:
    data = np.loadtxt(path)

    wlen = 20 if wlen is None else wlen

    spectrum = ExperimentalSpectrum(name=name, freq=data[:, 0], inten=data[:, 1])
    spectrum.peaks = DeterminedPeaks(spectrum, inten_thresh, prominence, wlen)
    return spectrum


def get_fit(path: str, name: str, min_freq: float = ClusterSpectrum.min_freq,
            max_freq: float = ClusterSpectrum.max_freq) -> ClusterSpectrum:
    if path.split(sep=".")[-1] != "cat":
        raise ValueError("Fitted spectra must be provided using cat file!")

    # Columns are not homogenous so columns must be used
    data = np.loadtxt(path, usecols=(0, 2))

    # Keep only needed values
    if max_freq is not None:
        data = data[(data[:, 0] > min_freq) & (data[:, 0] < max_freq)]
    else:
        data = data[(data[:, 0] > min_freq)]
    return ClusterSpectrum(name, freq=data[:, 0], rel_inten=np.power(10, data[:, 1]))
