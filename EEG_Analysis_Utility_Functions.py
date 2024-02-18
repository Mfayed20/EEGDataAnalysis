import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import plotly.express as px
import pyedflib
from mne.io import RawArray
from scipy import stats


def plotEEGData(data, times, title, channel_index, start_time=None, end_time=None):
    """
    Plots EEG data for a specific channel over a given time range.

    Parameters:
    - data (numpy.ndarray): The EEG data array.
    - times (numpy.ndarray): The time points corresponding to the data samples.
    - title (str): The title of the plot.
    - channel_index (int): The index of the channel to plot.
    - start_time (float, optional): The start time for plotting. Defaults to None.
    - end_time (float, optional): The end time for plotting. Defaults to None.
    """
    if start_time is not None and end_time is not None:
        time_mask = (times >= start_time) & (times <= end_time)
        data, times = data[:, time_mask], times[time_mask]

    if not (0 <= channel_index < data.shape[0]):
        print("Invalid channel index!")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(times, data[channel_index, :])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()


def readProcessedEEGData(h5_file_path):
    """
    Reads processed EEG data from an HDF5 file.

    Parameters:
    - h5_file_path (str): The path to the HDF5 file.

    Returns:
    - tuple: A tuple containing arrays of notch filtered, bandpass filtered, and Hilbert-transformed data.
    """
    with h5py.File(h5_file_path, "r") as h5_file:
        data_notch_filtered = h5_file["notch_filtered"][:]
        data_bandpass_filtered = h5_file["bandpass_filtered"][:]
        data_hilbert = h5_file["hilbert"][:]

    return data_notch_filtered, data_bandpass_filtered, data_hilbert


def plotSpikeDetectionResults(
    data, times, spikes, channels, channel_names, start_time=None, end_time=None
):
    """
    Plots the results of spike detection for specified channels and methods.

    Parameters:
    - data (numpy.ndarray): The EEG data array.
    - times (numpy.ndarray): The time points corresponding to the data samples.
    - spikes (dict): A dictionary containing spike indices for each channel and method.
    - channels (list): A list of channel indices to plot.
    - channel_names (list): A list of channel names corresponding to the indices.
    - start_time (float, optional): The start time for plotting. Defaults to None.
    - end_time (float, optional): The end time for plotting. Defaults to None.
    """
    num_methods = len(spikes[0])
    num_plots = len(channels) * num_methods
    fig, axes = plt.subplots(num_plots, 1, figsize=(15, num_plots * 3))
    axes = np.array([axes]) if num_plots == 1 else axes

    for i, channel_num in enumerate(channels):
        for j, (method_name, method_spikes) in enumerate(spikes[channel_num].items()):
            idx = i * num_methods + j
            ax = axes[idx]

            if start_time is not None and end_time is not None:
                time_mask = (times >= start_time) & (times <= end_time)
                filtered_times = times[time_mask]
                filtered_data = data[channel_num][time_mask]
                spike_times = times[method_spikes]
                spike_mask = (spike_times >= start_time) & (spike_times <= end_time)
                filtered_spike_times = spike_times[spike_mask]
                adjusted_spike_indices = (
                    method_spikes[spike_mask] - np.where(time_mask)[0][0]
                )

                ax.plot(filtered_times, filtered_data, label="Data")
                ax.scatter(
                    filtered_spike_times,
                    filtered_data[adjusted_spike_indices],
                    color="red",
                    label="Spikes",
                )
            else:
                ax.plot(times, data[channel_num], label="Data")
                ax.scatter(
                    times[method_spikes],
                    data[channel_num][method_spikes],
                    color="red",
                    label="Spikes",
                )

            ax.set_title(
                f"{channel_names[channel_num]} - Channel {channel_num} - Spike Detection ({method_name})"
            )
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.legend()

    plt.tight_layout()
    plt.show()


def plotChannelSegments(
    data,
    times,
    channel_num,
    segments,
    channel_name=None,
    start_time=None,
    end_time=None,
):
    """
    Plots EEG data for a specific channel with highlighted segments.

    Parameters:
    - data (numpy.ndarray): The EEG data array.
    - times (numpy.ndarray): The time points corresponding to the data samples.
    - channel_num (int): The index of the channel to plot.
    - segments (list): A list of tuples indicating the start and end indices of segments.
    - channel_name (str, optional): The name of the channel. Defaults to None.
    - start_time (float, optional): The start time for plotting. Defaults to None.
    - end_time (float, optional): The end time for plotting. Defaults to None.
    """
    start_idx = 0 if start_time is None else np.searchsorted(times, start_time)
    end_idx = len(times) if end_time is None else np.searchsorted(times, end_time)

    filtered_times = times[start_idx:end_idx]
    filtered_data = data[channel_num, start_idx:end_idx]

    plt.figure(figsize=(15, 5))
    plt.plot(filtered_times, filtered_data, label="Data")

    for i, (start, end) in enumerate(segments):
        segment_start = max(start, start_idx)
        segment_end = min(end, end_idx)
        if segment_start < segment_end:
            plt.plot(
                times[segment_start:segment_end],
                data[channel_num, segment_start:segment_end],
                label=f"Segment {i+1}",
            )

    title = (
        f"Channel {channel_num} - Data with Segments"
        if not channel_name
        else f"{channel_name} - Channel {channel_num} - Data with Segments"
    )
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plotAnnotationsMatplotlib(edf_file, description=None, time_buffer=10):
    """
    Plots annotations from an EDF file using Matplotlib.

    Parameters:
    - edf_file (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - description (str, optional): The description of the annotations to plot. Defaults to None.
    - time_buffer (int, optional): The time buffer before and after the annotation. Defaults to 10.
    """
    if description:
        description = description.lower()

    matched_annotations = (
        False  # Flag to track if any annotations matched the description
    )

    for index, annotation in enumerate(edf_file.annotations):
        if description and annotation["description"].lower() != description:
            continue

        matched_annotations = True  # Set flag to True if a matching annotation is found

        print(
            f'Annotation {index + 1}: Start: {annotation["onset"]}, Duration: {annotation["duration"]}, Description: {annotation["description"]}'
        )

        plot_start_time = max(0, annotation["onset"] - time_buffer)
        plot_end_time = annotation["onset"] + annotation["duration"] + time_buffer

        edf_file.plot(
            start=plot_start_time,
            duration=plot_end_time - plot_start_time,
            n_channels=min(10, len(edf_file.ch_names)),
            title=f"Annotation {index + 1}",
            scalings="auto",
            show_options=False,
        )

    if not matched_annotations and description:
        print(f"No annotations found with description '{description}'.")


def plotAnnotationsPlotly(raw_data, description=None, time_buffer=10):
    """
    Plots annotations from an EDF file using Plotly.

    Parameters:
    - raw_data (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - description (str, optional): The description of the annotations to plot. Defaults to None.
    - time_buffer (int, optional): The time buffer before and after the annotation. Defaults to 10.
    """
    if description:
        description = description.lower()

    matched_annotations = (
        False  # Flag to track if any annotations matched the description
    )

    for index, annotation in enumerate(raw_data.annotations):
        if description and annotation["description"].lower() != description:
            continue

        matched_annotations = True  # Set flag to True if a matching annotation is found

        start_time = max(0, annotation["onset"] - time_buffer)
        end_time = min(
            annotation["onset"] + annotation["duration"] + time_buffer,
            raw_data.n_times / raw_data.info["sfreq"] - 0.001,
        )

        segment = raw_data.copy().crop(tmin=start_time, tmax=end_time)
        segment_data = segment.get_data(picks=range(min(10, segment.info["nchan"])))
        segment_times = segment.times

        data_frame = pd.DataFrame(
            segment_data.T, columns=segment.ch_names[: min(10, len(segment.ch_names))]
        )
        data_frame["Time"] = segment_times

        plot_title = f"Annotation {index + 1}" + (
            f" - {description}" if description else ""
        )
        figure = px.line(
            data_frame, x="Time", y=data_frame.columns[:-1], title=plot_title
        )
        figure.show()

    if not matched_annotations and description:
        print(f"No annotations found with description '{description}'.")


def detectSpikesUsingMAD(data, threshold=8):
    """
    Detect spikes in EEG data using the Median Absolute Deviation (MAD) method.

    This function identifies spikes in the input data by calculating the modified Z-score
    of the data points. A data point is considered a spike if its modified Z-score exceeds
    a specified threshold.

    Parameters:
    - data (numpy.ndarray): The EEG data array where spikes are to be detected.
    - threshold (float, optional): The threshold for the modified Z-score above which a data point
      is considered a spike. Defaults to 8.

    Returns:
    - numpy.ndarray: An array of indices where spikes have been detected in the input data.
    """

    median = np.median(data)
    deviation = np.abs(data - median)
    mad = np.median(deviation)
    modified_z_score = 0.6745 * deviation / mad
    return np.where(modified_z_score > threshold)[0]


def detectSpikesInChannel(channel_data):
    """
    Detects spikes in a single EEG channel using the MAD method.

    Parameters:
    - channel_data (numpy.ndarray): The data of the channel where spikes are to be detected.

    Returns:
    - dict: A dictionary with the key "MAD" and value as the indices of detected spikes.
    """
    return {"MAD": detectSpikesUsingMAD(channel_data)}


def detectSpikesParallel(data):
    """
    Detects spikes in parallel across all EEG channels using the MAD method.

    Parameters:
    - data (numpy.ndarray): The EEG data array where spikes are to be detected.

    Returns:
    - list: A list of dictionaries with detected spikes for each channel.
    """
    with ThreadPoolExecutor() as executor:
        return list(executor.map(detectSpikesInChannel, data))


def calculateEventRelatedSynchronization(data, baseline):
    """
    Calculates the Event-Related Synchronization (ERS) percentage.

    Parameters:
    - data (numpy.ndarray): The EEG data array.
    - baseline (float): The baseline value for normalization.

    Returns:
    - numpy.ndarray: The ERS percentage for the data.
    """
    return ((data - baseline) / baseline) * 100


def calculateERSForChannel(channel_num, data, all_baselines):
    """
    Calculates the ERS percentage for a specific channel.

    Parameters:
    - channel_num (int): The index of the channel.
    - data (numpy.ndarray): The EEG data array.
    - all_baselines (dict): A dictionary containing baseline values for all channels.

    Returns:
    - tuple: A tuple containing the channel index and its ERS percentage.
    """
    return channel_num, calculateEventRelatedSynchronization(
        data[channel_num], all_baselines[channel_num]
    )


def calculateERSPercentageForAllChannels(data, all_baselines):
    """
    Calculates the ERS percentage for all channels in parallel.

    Parameters:
    - data (numpy.ndarray): The EEG data array.
    - all_baselines (dict): A dictionary containing baseline values for all channels.

    Returns:
    - dict: A dictionary containing the ERS percentage for each channel.
    """
    if isinstance(data, tuple):
        data = data[0]  # Assuming the first element of the tuple is the data array
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda channel_num: calculateERSForChannel(
                channel_num, data, all_baselines
            ),
            range(data.shape[0]),
        )
    return {channel_num: ERS_percentage for channel_num, ERS_percentage in results}


def normalizeChannelData(channel_data):
    """
    Normalizes the data of a single EEG channel.

    Parameters:
    - channel_data (numpy.ndarray): The data of the channel to be normalized.

    Returns:
    - numpy.ndarray: The normalized data of the channel.
    """
    return (channel_data - np.mean(channel_data)) / np.std(channel_data)


def normalizeDataParallel(data):
    """
    Normalizes the data of all EEG channels in parallel.

    Parameters:
    - data (numpy.ndarray): The EEG data array to be normalized.

    Returns:
    - numpy.ndarray: The normalized data array.
    """
    with ThreadPoolExecutor() as executor:
        return np.array(list(executor.map(normalizeChannelData, data)))


def selectTopEEGChannels(data, percentage=0.20):
    """
    Selects the top EEG channels based on their mean absolute values.

    Parameters:
    - data (numpy.ndarray): The EEG data array.
    - percentage (float, optional): The percentage of top channels to select. Defaults to 0.20.

    Returns:
    - numpy.ndarray: An array of indices of the selected top channels.
    """
    mean_values = np.abs(np.mean(data, axis=1))
    num_top_channels = int(percentage * data.shape[0])
    return np.argpartition(mean_values, -num_top_channels)[-num_top_channels:]


def performTTestOnChannelMeans(data):
    """
    Performs a one-sample t-test on the mean values of all EEG channels.

    Parameters:
    - data (numpy.ndarray): The EEG data array.

    Returns:
    - scipy.stats.ttest_1sampResult: The result of the t-test.
    """
    mean_values = np.abs(np.mean(data, axis=1))
    return stats.ttest_1samp(mean_values, 0)


def displayChannelNames(edf_file):
    """
    Displays the names of all channels in an EDF file.

    Parameters:
    - edf_file (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    """
    for i, ch_name in enumerate(edf_file.ch_names):
        print(f"Channel {i}: {ch_name}")


def removeSelectedChannels(raw, channels_to_remove):
    """
    Removes specified channels from an EDF file.

    Parameters:
    - raw (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - channels_to_remove (list): A list of channel names to remove.

    Returns:
    - mne.io.Raw: The EDF file with the specified channels removed.
    """
    return raw.copy().drop_channels(channels_to_remove)


def groupAndCreateBipolarChannels(channels):
    """
    Groups channels and creates bipolar channels.

    Parameters:
    - channels (list): A list of channel names.

    Returns:
    - list: A list of bipolar channel names.
    """
    bipolar_channels = []
    current_group = None
    group_channels = []
    for channel in channels:
        cleaned_channel = channel.replace("-Ref1", "")
        group = re.match(r"[a-zA-Z]+", cleaned_channel).group()
        numeric_part = int(re.search(r"\d+", cleaned_channel).group())
        if (
            current_group is not None
            and group != current_group
            and (group + "'") != current_group
            and ("'" + group) != current_group
        ):
            # Pair only consecutive channels within the group
            for i in range(len(group_channels) - 1):
                if group_channels[i][1] + 1 == group_channels[i + 1][1]:
                    bipolar_channels.append(
                        f"{group_channels[i][0]}-{group_channels[i + 1][0]}"
                    )
            group_channels = []
        current_group = group
        group_channels.append((cleaned_channel, numeric_part))
    # Pair only consecutive channels within the last group
    for i in range(len(group_channels) - 1):
        if group_channels[i][1] + 1 == group_channels[i + 1][1]:
            bipolar_channels.append(
                f"{group_channels[i][0]}-{group_channels[i + 1][0]}"
            )
    return bipolar_channels


def generateBipolarChannelData(raw, bipolar_channels):
    """
    Generates data for bipolar channels from an EDF file.

    Parameters:
    - raw (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - bipolar_channels (list): A list of bipolar channel names.

    Returns:
    - mne.io.RawArray: An MNE RawArray object containing the bipolar channel data.
    """
    original_data = raw.get_data()
    bipolar_data = []
    for bipolar_channel in bipolar_channels:
        ch1_name, ch2_name = bipolar_channel.split("-")
        ch1_idx = raw.ch_names.index(ch1_name + "-Ref1")
        ch2_idx = raw.ch_names.index(ch2_name + "-Ref1")
        bipolar_data.append(original_data[ch1_idx] - original_data[ch2_idx])
    bipolar_data = np.array(bipolar_data)
    bipolar_info = mne.create_info(bipolar_channels, raw.info["sfreq"], ch_types="eeg")
    bipolar_raw = RawArray(bipolar_data, bipolar_info)
    return bipolar_raw


def processEEGChunkOptimized(chunk_data, info, notch_freqs, l_freq, h_freq):
    """
    Processes a chunk of EEG data with optimized filtering and Hilbert transform.

    Parameters:
    - chunk_data (numpy.ndarray): The EEG data chunk to process.
    - info (mne.Info): The MNE Info object containing information about the EEG setup.
    - notch_freqs (list): Frequencies for the notch filter.
    - l_freq (float): Lower frequency bound for the band-pass filter.
    - h_freq (float): Upper frequency bound for the band-pass filter.

    Returns:
    - numpy.ndarray: The processed EEG data chunk.
    """
    raw_chunk = mne.io.RawArray(chunk_data, info, verbose=False)
    # Apply notch filter
    if notch_freqs is not None:
        raw_chunk.notch_filter(
            notch_freqs, method="fir", fir_design="firwin", verbose=False
        )
    # Apply band-pass filter
    raw_chunk.filter(
        l_freq,
        h_freq,
        method="fir",
        fir_design="firwin",
        l_trans_bandwidth="auto",
        h_trans_bandwidth="auto",
        verbose=False,
    )
    # Apply Hilbert transform without the 'method' argument
    raw_chunk.apply_hilbert(envelope=True, verbose=False)
    return raw_chunk.get_data().astype(np.float32)


def processAndStoreEEGChunk(
    chunk_data, info, notch_freqs, l_freq, h_freq, h5_file, start_sample
):
    """
    Processes a chunk of EEG data and saves it into an HDF5 file.

    Parameters:
    - chunk_data: The EEG data chunk to process.
    - info: The MNE Info object containing information about the EEG setup.
    - notch_freqs: Frequencies for the notch filter.
    - l_freq: Lower frequency bound for the band-pass filter.
    - h_freq: Upper frequency bound for the band-pass filter.
    - h5_file: The HDF5 file object to save the processed data into.
    - start_sample: The starting sample index in the HDF5 file for this chunk.
    """
    # Convert the chunk data into an MNE RawArray for processing
    raw_chunk = mne.io.RawArray(chunk_data, info, verbose=False)

    # Apply notch filter if frequencies are provided and save
    if notch_freqs is not None:
        notch_filtered = (
            raw_chunk.copy()
            .notch_filter(notch_freqs, method="fir", fir_design="firwin", verbose=False)
            .get_data()
            .astype(np.float32)
        )
        h5_file["notch_filtered"][
            :, start_sample : start_sample + notch_filtered.shape[1]
        ] = notch_filtered

    # Apply band-pass filter and save
    bandpass_filtered = (
        raw_chunk.copy()
        .filter(
            l_freq,
            h_freq,
            method="fir",
            fir_design="firwin",
            l_trans_bandwidth="auto",
            h_trans_bandwidth="auto",
            verbose=False,
        )
        .get_data()
        .astype(np.float32)
    )
    h5_file["bandpass_filtered"][
        :, start_sample : start_sample + bandpass_filtered.shape[1]
    ] = bandpass_filtered

    # Apply Hilbert transform to get the analytic signal and save
    hilbert_transformed = (
        raw_chunk.copy()
        .apply_hilbert(envelope=True, verbose=False)
        .get_data()
        .astype(np.float32)
    )
    h5_file["hilbert"][
        :, start_sample : start_sample + hilbert_transformed.shape[1]
    ] = hilbert_transformed


def processEEGDataInChunksParallel(
    edf_file, h5_file_path, samples_per_chunk, notch_freqs=60, l_freq=25, h_freq=250
):
    """
    Processes EEG data in parallel chunks and stores the results in an HDF5 file.

    Parameters:
    - edf_file (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - h5_file_path (str): Path to the HDF5 file where the processed data will be stored.
    - samples_per_chunk (int): Number of samples per chunk for processing.
    - notch_freqs (float): Frequency for the notch filter. Defaults to 60Hz.
    - l_freq (float): Lower frequency bound for the band-pass filter. Defaults to 25Hz.
    - h_freq (float): Upper frequency bound for the band-pass filter. Defaults to 250Hz.
    """
    with h5py.File(h5_file_path, "w") as h5_file:
        for dataset_name in ["notch_filtered", "bandpass_filtered", "hilbert"]:
            h5_file.create_dataset(
                dataset_name,
                (edf_file.info["nchan"], len(edf_file.times)),
                dtype="float32",
            )

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    processAndStoreEEGChunk,
                    edf_file.get_data(
                        start=start_sample, stop=start_sample + samples_per_chunk
                    ),
                    edf_file.info,
                    notch_freqs,
                    l_freq,
                    h_freq,
                    h5_file,
                    start_sample,
                )
                for start_sample in range(0, len(edf_file.times), samples_per_chunk)
            ]

            for future in as_completed(futures):
                future.result()  # Wait for all futures to complete


def calculateBaselineAvoidingSpikes(
    data,
    times,
    annotations,
    spike_indices,
    duration,
    num_segments,
    min_segment_gap,
    channel_num,
    max_attempts,
):
    """
    Calculates the baseline of EEG data avoiding spikes and annotations.

    Parameters:
    - data (numpy.ndarray): The EEG data.
    - times (numpy.ndarray): The time points corresponding to the data.
    - annotations (list): Time points of annotations to avoid.
    - spike_indices (list): Indices of spikes to avoid.
    - duration (float): Duration of each segment to consider for baseline calculation.
    - num_segments (int): Number of segments to calculate the baseline over.
    - min_segment_gap (float): Minimum gap between segments.
    - channel_num (int): The channel number to calculate the baseline for.
    - max_attempts (int): Maximum number of attempts to find suitable segments.

    Returns:
    - float: The calculated baseline.
    - list: The segments used for baseline calculation.
    """
    assert data.shape[1] == len(times), "The data and times must have the same length."
    total_duration = times[-1]
    duration_indices = int(duration * len(times) / total_duration)
    min_gap_indices = int(min_segment_gap * len(times) / total_duration)
    annotation_indices = sorted(
        [int(a * len(times) / total_duration) for a in annotations]
    )
    baseline_segments = []

    attempts = 0
    start_index = 0
    while (
        len(baseline_segments) < num_segments
        and start_index < len(times) - duration_indices
    ):
        if any(
            start_index <= spike < start_index + duration_indices
            for spike in spike_indices
        ) or any(abs(start_index - a) < min_gap_indices for a in annotation_indices):
            start_index += 1
            attempts += 1
            if attempts > max_attempts:
                print(
                    f"Channel {channel_num}: Maximum attempts reached ({max_attempts}). Stopping search for this channel."
                )
                break
            continue
        baseline_segments.append((start_index, start_index + duration_indices))
        start_index += duration_indices + min_gap_indices

    if not baseline_segments:
        return np.nan, []

    baseline_segments_data = [
        data[channel_num][start:end] for start, end in baseline_segments
    ]
    baseline = np.mean(baseline_segments_data)
    return baseline, baseline_segments


def calculateBaselineAndSegmentsForChannel(
    data,
    times,
    annotations,
    channel_num,
    spikes,
    duration,
    num_segments,
    min_segment_gap,
    max_attempts,
    max_retries,
):
    """
    Calculates the baseline and segments for a specific channel, retrying with adjusted parameters if necessary.

    Parameters:
    - data (numpy.ndarray): The EEG data.
    - times (numpy.ndarray): The time points corresponding to the data.
    - annotations (list): Time points of annotations to avoid.
    - channel_num (int): The channel number to calculate the baseline and segments for.
    - spikes (dict): Dictionary containing spike information for each channel.
    - duration (float): Duration of each segment to consider for baseline calculation.
    - num_segments (int): Number of segments to calculate the baseline over.
    - min_segment_gap (float): Minimum gap between segments.
    - max_attempts (int): Maximum number of attempts to find suitable segments.
    - max_retries (int): Maximum number of retries with adjusted parameters.

    Returns:
    - float: The calculated baseline.
    - list: The segments used for baseline calculation.
    """
    spikes_for_channel = spikes[channel_num]["MAD"]
    print(f"Processing Channel {channel_num} with {len(spikes_for_channel)} spikes...")

    retries = 0
    while retries < max_retries:
        baseline, segments = calculateBaselineAvoidingSpikes(
            data,
            times,
            annotations,
            spikes_for_channel,
            duration,
            num_segments,
            min_segment_gap,
            channel_num,
            max_attempts,
        )

        if len(segments) == num_segments:
            print(f"Done with Channel {channel_num}")
            return baseline, segments

        retries += 1
        max_attempts = int(max_attempts * 1.1)
        min_segment_gap *= 0.8
        num_segments_new = int(num_segments * 1.1)
        duration_new = (duration * num_segments) / num_segments_new
        duration, num_segments, min_segment_gap = (
            duration_new,
            num_segments_new,
            min_segment_gap * 0.7,
        )
        print(
            f"Retrying Channel {channel_num} with adjusted parameters (attempt {retries})..."
        )

    print(
        f"Channel {channel_num}: Maximum retries reached ({max_retries}). Setting duration to zero and returning last result."
    )
    return baseline, segments


def calculateBaselinesAndSegmentsForAllChannels(
    data,
    times,
    annotations,
    spikes,
    duration=10,
    num_segments=30,
    min_segment_gap=2,
    max_attempts=50000,
    max_retries=5,
):
    """
    Calculates baselines and segments for all channels in the EEG data.

    Parameters:
    - data (numpy.ndarray): The EEG data.
    - times (numpy.ndarray): The time points corresponding to the data.
    - annotations (list): Time points of annotations to avoid.
    - spikes (dict): Dictionary containing spike information for each channel.
    - duration (float): Duration of each segment to consider for baseline calculation. Defaults to 10.
    - num_segments (int): Number of segments to calculate the baseline over. Defaults to 30.
    - min_segment_gap (float): Minimum gap between segments. Defaults to 2.
    - max_attempts (int): Maximum number of attempts to find suitable segments. Defaults to 50000.
    - max_retries (int): Maximum number of retries with adjusted parameters. Defaults to 5.

    Returns:
    - dict: A dictionary of calculated baselines for each channel.
    - dict: A dictionary of segments used for baseline calculation for each channel.
    """
    all_baselines = {}
    all_segments = {}
    args = [
        (
            data,
            times,
            annotations,
            channel_num,
            spikes,
            duration,
            num_segments,
            min_segment_gap,
            max_attempts,
            max_retries,
        )
        for channel_num in range(data.shape[0])
    ]
    with ThreadPoolExecutor() as executor:
        results = executor.map(
            lambda x: calculateBaselineAndSegmentsForChannel(*x), args
        )
    for channel_num, (baseline, segments) in enumerate(results):
        all_baselines[channel_num] = baseline
        all_segments[channel_num] = segments
    return all_baselines, all_segments


def reportChannelsWithSegmentIssues(baseline_segments, times, expected_num_segments):
    """
    Reports channels with either missing or extra segments compared to the expected number.

    Parameters:
    - baseline_segments (dict): A dictionary containing the segments for each channel.
    - times (numpy.ndarray): The time points corresponding to the data.
    - expected_num_segments (int): The expected number of segments for each channel.
    """
    channels_with_issues = {"missing": [], "extra": []}

    for channel, segments in baseline_segments.items():
        segment_count = len(segments)
        if segment_count < expected_num_segments:
            channels_with_issues["missing"].append(channel)
        elif segment_count > expected_num_segments:
            channels_with_issues["extra"].append(channel)

    for issue_type, channels in channels_with_issues.items():
        if channels:
            issue_description = (
                "more than the expected"
                if issue_type == "extra"
                else "the full number of"
            )
            print(
                f"\nChannels for which the function found {issue_description} segments:"
            )
            for channel in channels:
                total_duration = sum(
                    times[end] - times[start]
                    for start, end in baseline_segments[channel]
                )
                print(
                    f"Channel {channel}: Found {len(baseline_segments[channel])} segments, expected {expected_num_segments}. Total duration of segments: {total_duration:.2f} seconds"
                )

    if not channels_with_issues["missing"] and not channels_with_issues["extra"]:
        print("All channels have the full number of segments.")


def _generateChannelMetadata(channel_idx, channel_label, signal_data, sampling_rate):
    """
    Generates metadata for a single channel.

    Parameters:
    - channel_idx (int): The index of the channel.
    - channel_label (str): The label of the channel.
    - signal_data (numpy.ndarray): The signal data for the channel.
    - sampling_rate (float): The sampling rate of the data.

    Returns:
    - dict: A dictionary containing the metadata for the channel.
    """
    max_val = round(np.max(signal_data[channel_idx]), 5)
    min_val = round(np.min(signal_data[channel_idx]), 5)

    return {
        "label": channel_label,
        "dimension": "uV",
        "sample_rate": sampling_rate,
        "physical_max": max_val,
        "physical_min": min_val,
        "digital_max": 32767,
        "digital_min": -32768,
        "transducer": "",
        "prefilter": "",
    }


def saveEDFSegment(edf, start, end, output_path):
    """
    Saves a segment of an EDF file to a new file.

    Parameters:
    - edf (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - start (float): The start time of the segment.
    - end (float): The end time of the segment.
    - output_path (str): The path to save the new EDF file.

    Returns:
    - None
    """
    start, end = float(start), float(end)
    assert end > start, "end_time must be greater than start_time"

    segment = edf.copy().crop(tmin=start, tmax=end)
    signal_data = segment.get_data()
    channel_count = signal_data.shape[0]
    sampling_rate = edf.info["sfreq"]

    channel_info = list(
        ThreadPoolExecutor().map(
            _generateChannelMetadata,
            range(channel_count),
            edf.info["ch_names"],
            [signal_data] * channel_count,
            [sampling_rate] * channel_count,
        )
    )

    with pyedflib.EdfWriter(output_path, channel_count) as writer:
        for i, info in enumerate(channel_info):
            writer.setSignalHeader(i, info)
        writer.writeSamples(signal_data)

    print(f"Segment from {start} to {end} saved to {output_path}")


def extractAnnotationsToEDF(
    edf, annotations, description, file_prefix="buffer", buffer_secs=10, save_dir="."
):
    """
    Extracts segments from an EDF file based on annotations and saves them as new EDF files.

    Parameters:
    - edf (mne.io.Raw): The EDF file loaded as an mne.io.Raw object.
    - annotations (list): A list of annotations from which segments will be extracted.
    - description (str): The description of the annotations to be extracted.
    - file_prefix (str, optional): Prefix for the output file names. Defaults to 'buffer'.
    - buffer_secs (int, optional): Number of seconds to buffer before and after the annotation. Defaults to 10.
    - save_dir (str, optional): Directory where the extracted segments will be saved. Defaults to '.'.

    Returns:
    - list: A list of paths to the saved EDF files.
    """
    extracted_files = []

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for idx, ann in enumerate(annotations):
        # Check if the annotation description matches the target description
        if ann["description"].lower() == description.lower():
            # Construct the file name and path
            file_name = f"{buffer_secs}s_{file_prefix}_{description}_{idx}_data.edf"
            full_path = os.path.join(save_dir, file_name)
            extracted_files.append(full_path)

            # Log the annotation being processed
            print(
                f"Annotation {idx + 1}: Start: {ann['onset']}, Duration: {ann['duration']}, Description: {ann['description']}"
            )

            # Calculate the start and end times with buffer
            start, end = (
                max(0, ann["onset"] - buffer_secs),
                ann["onset"] + ann["duration"] + buffer_secs,
            )

            # Extract the segment from the EDF
            segment = edf.copy().crop(tmin=start, tmax=end)
            signal_data = segment.get_data()
            channel_count = signal_data.shape[0]
            sampling_rate = edf.info["sfreq"]

            # Generate metadata for each channel
            channel_info = list(
                ThreadPoolExecutor().map(
                    _generateChannelMetadata,
                    range(channel_count),
                    edf.info["ch_names"],
                    [signal_data] * channel_count,
                    [sampling_rate] * channel_count,
                )
            )

            # Write the segment to a new EDF file
            with pyedflib.EdfWriter(full_path, channel_count) as writer:
                for i, info in enumerate(channel_info):
                    writer.setSignalHeader(i, info)
                writer.writeSamples(signal_data)

    return extracted_files
