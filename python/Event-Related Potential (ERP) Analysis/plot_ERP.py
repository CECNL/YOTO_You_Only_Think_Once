import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from config import channel_order, channel_plot_order, filt_trigger, dataset_path, trigger_info
from dataset import Subject
from pathlib import Path
import pickle
import mne

def plot_ERP_subject(args):
    triggers = filt_trigger(args.include_trigger, args.exclude_trigger, args.include_category, args.exclude_category)
    if len(triggers) == 0:
        print("No trigger meet the condition.")
        return
    ch_plot_order = np.array(channel_plot_order).transpose(1, 0)[::-1]
    for sid in args.subject:
        print(f"Subject: {sid}")
        fig, axes = plt.subplots(nrows=5, ncols=6, sharey=True, layout="tight", figsize=(25.5, 13.25))
        erp = None
        for session in args.session:
            sessionDataPath = f"{dataset_path}/S{sid:03d}/session_{session}/task_250_clean.set"
            if not os.path.exists(sessionDataPath):
                continue
            with mne.utils.use_log_level("Error"):
                raw = mne.io.read_raw_eeglab(sessionDataPath, preload=False)
                events, eventMap = mne.events_from_annotations(raw)
                # print(events)
                # print(eventMap)
                # for eventId, latency in eventAdjust.items():
                #     events[events[:, 2] == eventMap[eventId], 0] += int(latency * raw.info['sfreq'])  # music_D
                eventIds = [eventMap[t.split("_")[-1]] for t in triggers]
                eventMask = np.isin(events[:, 2], eventIds)
                if args.eventOffset < 0:
                    eventMask[:args.eventOffset] = eventMask[-args.eventOffset:]
                    eventMask[args.eventOffset:] = False
                elif args.eventOffset > 0:
                    eventMask[args.eventOffset:] = eventMask[:-args.eventOffset]
                    eventMask[-args.eventOffset:] = False
                selectedEvents = events[eventMask]
                epochs = mne.Epochs(raw, selectedEvents, event_id=None, 
                        tmin=args.interval[0], tmax=args.interval[1], baseline=(args.interval[0], args.interval[0]+args.baseline), preload=True)
                # print(epochs.get_data().shape)
                if erp is None:
                    erp = epochs.get_data() * 10**6
                else:
                    erp = np.concatenate([erp, epochs.get_data() * 10**6], axis=0)
        if erp is None or erp.shape[0] == 0:
            print("No trial meet the condition.")
            continue
        # fig.canvas.manager.set_window_title(f"{args}, trials: {trial_cnt}")
        fig.suptitle(f"subject: {sid}, session: {args.session}, trigger: {triggers}, trials: {erp.shape[0]}")
        erp_stderr = erp.std(axis=0) / np.sqrt(erp.shape[0])
        erp = erp.mean(axis=0)
        baselineLength = int(args.baseline * 250)
        for rax, rch in zip(axes, ch_plot_order):
            for ax, ch in zip(rax, rch):
                ch_id = channel_order.index(ch)
                ax.set_title(ch)
                ax.fill_between(np.arange(erp.shape[-1]), erp[ch_id, :] + erp_stderr[ch_id, :], erp[ch_id, :] - erp_stderr[ch_id, :], alpha=0.5)
                ax.plot(erp[ch_id, :], label="ERP")

                xticks = np.array([0, *np.arange(args.baseline, args.interval[1]-args.interval[0]+0.1, step=0.1)]) * 250
                ax.set_xticks(xticks.astype(int))
                ax.set_xticklabels((ax.get_xticks() - baselineLength) / 250)
                ax.set_xlim(baselineLength-25)

                ax.tick_params(axis='x', labelrotation=90)
                # ax.set_ylim(-10, 10)
                # ax.set_ylim(-1.2, 1.2)
                ax.grid("on")
        axes[0, -1].legend(loc=1)
        # fig.canvas.manager.window.showMaximized()
        if args.save:
            os.makedirs(f"../../figure/{args.savedir}", exist_ok=True)
            plt.savefig(f"../../figure/{args.savedir}/S{sid:03d}_{args.save}")
        if not args.hide:
            # fig_manager = plt.get_current_fig_manager()
            # fig_manager.window.showMaximized()
            plt.show()
        else:
            plt.close(fig)

def plot_ERP_channel(args):
    triggers = filt_trigger(args.include_trigger, args.exclude_trigger, args.include_category, args.exclude_category)
    if len(triggers) == 0:
        print("No trigger meet the condition.")
        return
    # subjects = [Subject(sid) for sid in range(1, 27)]
    for ch in args.channel:
        print(f"Channel: {ch}")
        fig, axes = plt.subplots(nrows=5, ncols=6, sharey=False, layout="tight", figsize=(25.5, 13.25))
        fig.suptitle(f"channel: {ch}, session: {args.session}, trigger: {triggers}")
        for rsid, rax in zip(np.arange(1, 31).reshape(-1, 6), axes):
            for sid, ax in zip(rsid, rax):
                if sid > 26:
                    continue
                erp = None
                for session in args.session:
                    sessionDataPath = f"{dataset_path}/S{sid:03d}/session_{session}/task_250_clean.set"
                    if not os.path.exists(sessionDataPath):
                        continue
                    with mne.utils.use_log_level("Error"):
                        raw = mne.io.read_raw_eeglab(sessionDataPath, preload=False)
                        raw.pick_channels([ch])
                        events, eventMap = mne.events_from_annotations(raw)
                        # print(events)
                        # print(eventMap)
                        # for eventId, latency in eventAdjust.items():
                        #     events[events[:, 2] == eventMap[eventId], 0] += int(latency * raw.info['sfreq'])  # music_D
                        eventIds = [eventMap[t.split("_")[-1]] for t in triggers]
                        eventMask = np.isin(events[:, 2], eventIds)
                        if args.eventOffset < 0:
                            eventMask[:args.eventOffset] = eventMask[-args.eventOffset:]
                            eventMask[args.eventOffset:] = False
                        elif args.eventOffset > 0:
                            eventMask[args.eventOffset:] = eventMask[:-args.eventOffset]
                            eventMask[-args.eventOffset:] = False
                        selectedEvents = events[eventMask]
                        epochs = mne.Epochs(raw, selectedEvents, event_id=None, 
                                tmin=args.interval[0], tmax=args.interval[1], baseline=(args.interval[0], args.interval[0]+args.baseline), preload=True)
                        # print(epochs.get_data().shape)
                        if erp is None:
                            erp = epochs.get_data() * 10**6
                        else:
                            erp = np.concatenate([erp, epochs.get_data() * 10**6], axis=0)
                if erp is None or erp.shape[0] == 0:
                    break                    
                erp = erp[:, 0, :]
                # fig.canvas.manager.set_window_title(f"{args}, trials: {trial_cnt}")
                ax.set_title(f"S{sid:03d}, trials: {erp.shape[0]}")   
                baselineLength = int(args.baseline*250)
                xticks = np.array([0, *np.arange(args.baseline, args.interval[1]-args.interval[0]+0.1, step=0.1)]) * 250
                ax.set_xticks(xticks.astype(int))
                ax.set_xticklabels((ax.get_xticks() - baselineLength) / 250)
                ax.set_xlim(baselineLength-25, (args.interval[1]-args.interval[0])*250)
                    
                erp_stderr = erp.std(axis=0) / np.sqrt(erp.shape[0])
                erp = erp.mean(axis=0)
                ax.fill_between(np.arange(erp.shape[-1]), erp + erp_stderr, erp - erp_stderr, alpha=0.5)
                ax.plot(erp, label="ERP")
                ax.tick_params(axis='x', labelrotation=90)
                # ax.set_ylim(-10, 20)
                # ax.set_ylim(-1.2, 1.2)
                ax.grid("on")
        axes[0, -1].legend(loc=1)
        if args.save:
            os.makedirs(f"../../figure/{args.savedir}", exist_ok=True)
            plt.savefig(f"../../figure/{args.savedir}/{ch}_{args.save}")
        if not args.hide:
            plt.show()
        else:
            plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", nargs="+", type=int, help="Subject id(when type=subject)", default=range(1, 27))
    parser.add_argument("-ss", "--session", nargs="+", type=int, help="session id(1 or 2)", default=[1, 2])
    parser.add_argument("-ch", "--channel", nargs="+", type=str, help="channel names(when type=channel)", default=channel_order)
    parser.add_argument("-it", "--include_trigger", nargs="+", type=str, help="include triggers", default=[])
    parser.add_argument("-et", "--exclude_trigger", nargs="+", type=str, help="exclude triggers", default=[])
    parser.add_argument("-ic", "--include_category", nargs="+", type=str, help="include trigger that contains any given categories", default=[])
    parser.add_argument("-ec", "--exclude_category", nargs="+", type=str, help="exclude trigger that contains one of the given categories", default=[])
    parser.add_argument("--interval", nargs="+", type=float, help="time interval (s)", default=[])
    parser.add_argument("-b", "--baseline", type=float, help="baseline length (s)", default=1)
    parser.add_argument("--eventOffset", type=int, help="baseline length (s)", default=0)
    parser.add_argument("--save", type=str, help="figure name to save", default="")
    parser.add_argument("--savedir", type=str, help="figure name to save", default=".")
    parser.add_argument("--hide", action="store_true", help="do not show figure", default=False)
    parser.add_argument("--type", type=str, help="subject or channel", default="subject")
    args = parser.parse_args()
    
    print(args)
    if args.type == "subject":
        plot_ERP_subject(args)
    elif args.type == "channel":
        plot_ERP_channel(args)
    else:
        print(f"Unknown type: {args.type}")
    
    
if __name__=="__main__":
    main()
    