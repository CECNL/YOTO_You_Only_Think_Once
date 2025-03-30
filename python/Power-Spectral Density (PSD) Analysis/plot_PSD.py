import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from config import channel_order, channel_plot_order, filt_trigger, dataset_path, trigger_info
from dataset import Subject
from pathlib import Path
import pdb
import pickle
from scipy.signal import periodogram, welch
import mne 
from matplotlib import colors

def calculate_band_powers(eeg_data, sfreq=250, method="welch"):
    if method == "welch":
        freqs, psd = welch(eeg_data, sfreq, nperseg=250)
    elif method == "periodogram":
        freqs, psd = periodogram(eeg_data, sfreq, scaling='density', window="hann")
    else:
        raise NotImplementedError
    # print(freqs)

    # Define EEG bands
    delta_band = (1, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)

    # Function to calculate band power
    def band_power(psd, freqs, band):
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.trapz(psd[:, idx_band], freqs[idx_band], axis=-1)

    return np.array([band_power(psd, freqs, delta_band), band_power(psd, freqs, theta_band), band_power(psd, freqs, alpha_band), band_power(psd, freqs, beta_band)])

def topo_PSD(data: np.ndarray, ax, vlim, cmap, mask=None, maskParam=None):
    """_summary_

    Args:
        data (np.ndarray): Delta of log power variance with shape [C, 1]
        channel (list): Names of each channel in the same order of data.
        title (str, optional): title of figure.
    """
    # create montage
    standard_montage = mne.channels.make_standard_montage("standard_1020")
    data = data.reshape(-1, 1)
    assert len(channel_order) == len(data)
    channel_location = standard_montage.get_positions()["ch_pos"]
    channel_location = {ch : channel_location[ch] for ch in channel_order}

    nasion = nasion=standard_montage.get_positions()["nasion"]
    lpa = standard_montage.get_positions()["lpa"]
    rpa = standard_montage.get_positions()["rpa"]
    montage = mne.channels.make_dig_montage(ch_pos=channel_location, nasion=nasion, lpa=lpa, rpa=rpa)
    ch_info = mne.create_info(ch_names=montage.ch_names, sfreq=250, ch_types="eeg")

    evoked = mne.EvokedArray(data, ch_info)
    evoked.set_montage(montage)
    if cmap  is not None:
        mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, vmin=vlim[0], vmax=vlim[1], axes=ax, names=channel_order, show_names=True, show=False, cmap=cmap, mask=mask, mask_params=maskParam)
    else:
        mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, vmin=vlim[0], vmax=vlim[1], axes=ax, names=channel_order, show_names=True, show=False, mask=mask, mask_params=maskParam)

def get_cmap(cval, color):
    norm=plt.Normalize(min(cval),max(cval))
    tuples = list(zip(map(norm,cval), color))
    return colors.LinearSegmentedColormap.from_list("", tuples)

def plot_PSD_topo(args):
    triggers = filt_trigger(args.include_trigger, args.exclude_trigger, args.include_category, args.exclude_category)
    if len(triggers) == 0:
        print("No trigger meet the condition.")
        return
    ch_plot_order = np.array(channel_plot_order).transpose(1, 0)[::-1]
    for sid in args.subject:
        print(f"Subject: {sid}")
        fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(25.5, 13.25))
        fixationPSD = np.zeros((0, 4, 30))
        imageryPSD = np.zeros((0, 4, 30))
        for rowId, section in enumerate(["fixation", "imagery"]):
            for session in args.session:
                if args.baseline:
                    if not os.path.exists(f"{dataset_path}/S{sid:03d}/session_{session}/resting_epoch/trial_2_resting.pickle"):
                        print(f"Baseline not found {dataset_path}/S{sid:03d}/session_{session}/resting_epoch/trial_2_resting.pickle... skip")
                        break
                    with open(f"{dataset_path}/S{sid:03d}/session_{session}/resting_epoch/trial_2_resting.pickle", "rb") as f:
                        baselineData = pickle.load(f)
                        baselinePSD = calculate_band_powers(baselineData["data"][:, 250:])

                sessionDataPath = f"{dataset_path}/S{sid:03d}/session_{session}"
                if not os.path.exists(sessionDataPath):
                    continue
                sessionDataPath = Path(sessionDataPath)
                for filePath in sessionDataPath.rglob(f"*{section}*.pickle"):
                    with open(filePath, "rb") as f:
                        newTrial = pickle.load(f)
                    if f"{section}_{newTrial['trigger']}" in triggers:
                        if section == "fixation":
                            if args.baseline:
                                fixationPSD = np.vstack((fixationPSD, [calculate_band_powers(newTrial["data"][:, 250:]) - baselinePSD]))
                            else:
                                fixationPSD = np.vstack((fixationPSD, [calculate_band_powers(newTrial["data"][:, 250:])]))
                        elif section == "imagery":
                            if args.baseline:
                                imageryPSD = np.vstack((imageryPSD, [calculate_band_powers(newTrial["data"][:, 250:]) - baselinePSD]))
                            else:
                                imageryPSD = np.vstack((imageryPSD, [calculate_band_powers(newTrial["data"][:, 250:])]))
                        else:
                            print("Unknown section")
        if fixationPSD.shape[0] == 0 or imageryPSD.shape[0] == 0:
            print(f"Subject {sid} has no data... skipped")
            continue
        # print(baselinePSD)
        fixationPSD = fixationPSD.mean(axis=0)
        imageryPSD = imageryPSD.mean(axis=0)
        fig.suptitle(f"subject: {sid}, session: {args.session}")  # , trigger: {triggers}, trials: {bandPSD.shape[0]}
        
        vlims = [[min(0, bandFPSD.min(), bandIPSD.min()), max(bandFPSD.max(), bandIPSD.max())] for bandFPSD, bandIPSD in zip(fixationPSD, imageryPSD)]
        if args.baseline:
            vlims = [[-10, 10] for _ in vlims]
            cmaps = [get_cmap([bandVlim[0], 0, bandVlim[1]], ["blue", "white", "red"]) for bandVlim in vlims]
        else:
            cmaps = [get_cmap([bandVlim[0], bandVlim[1]], ["white", "red"]) for bandVlim in vlims]
        for bandId, (bandName, vlim, cmap) in enumerate(zip(["delta", "theta", "alpha", "beta"], vlims, cmaps)):
            axes[0][bandId].set_title(bandName)
            topo_PSD(fixationPSD[bandId], axes[0][bandId], vlim, cmap)
            topo_PSD(imageryPSD[bandId], axes[1][bandId], vlim, cmap)
                
        deltaPSD = imageryPSD - fixationPSD
        absMax = np.abs(deltaPSD).max(axis=1, keepdims=True)
        vlims = np.concatenate([-absMax, absMax], axis=1)
        cmaps = [get_cmap([vlim[0], 0, vlim[1]], ["blue", "white", "red"]) for vlim in vlims]
        for bandId, (bandName, vlim, cmap) in enumerate(zip(["delta", "theta", "alpha", "beta"], vlims, cmaps)):
            # topo_PSD(deltaPSD[bandId], axes[2][bandId], vlim, cmap)
            if args.baseline:
                topo_PSD(deltaPSD[bandId], axes[2][bandId], [-10, 10], get_cmap([-10, 0, 10], ["blue", "white", "red"]))
            else:
                topo_PSD(deltaPSD[bandId], axes[2][bandId], [-10, 10], get_cmap([-10, 0, 10], ["blue", "white", "red"]))
                
        for colId in range(axes.shape[1]):
            fig.colorbar(axes[0, colId].images[-1], ax=axes[1, colId], location='right', 
                            shrink=0.7, pad=0.05, aspect=20, label='PSD (uV^2/Hz)')
            fig.colorbar(axes[2, colId].images[-1], ax=axes[2, colId], location='right', 
                            shrink=0.7, pad=0.05, aspect=20, label='PSD difference (uV^2/Hz)')
        if args.save:
            os.makedirs(f"../../figure/{args.savedir}", exist_ok=True)
            plt.savefig(f"../../figure/{args.savedir}/S{sid:03d}_{args.save}")
        if not args.hide:
            plt.show()
        else:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", nargs="+", type=int, help="Subject id(when type=subject)", default=range(1, 27))
    parser.add_argument("-ss", "--session", nargs="+", type=int, help="session id(1 or 2)", default=[1, 2])
    parser.add_argument("-it", "--include_trigger", nargs="+", type=str, help="include triggers", default=[])
    parser.add_argument("-et", "--exclude_trigger", nargs="+", type=str, help="exclude triggers", default=[])
    parser.add_argument("-ic", "--include_category", nargs="+", type=str, help="include trigger that contains any given categories", default=[])
    parser.add_argument("-ec", "--exclude_category", nargs="+", type=str, help="exclude trigger that contains one of the given categories", default=[])
    parser.add_argument("--save", type=str, help="figure name to save", default="")
    parser.add_argument("--savedir", type=str, help="figure name to save", default=".")
    parser.add_argument("--hide", action="store_true", help="do not show figure", default=False)
    parser.add_argument("-b", "--baseline", action="store_true", help="substract resting baseline", default=False)
    args = parser.parse_args()
    
    print(args)
    plot_PSD_topo(args)

if __name__=="__main__":
    main()
    