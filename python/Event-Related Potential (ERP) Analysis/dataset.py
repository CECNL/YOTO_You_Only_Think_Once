import os
import pdb
import math
from torch.utils.data import Dataset

from typing import Iterable, Union, Tuple
from copy import deepcopy
from scipy.io import loadmat
import pickle
import numpy as np
from functools import cmp_to_key
from pathlib import Path
from config import dataset_path, dataset_config, filt_trigger, trigger_info

class Subject():
    def __init__(self, subjectID: int) -> None:
        self.ID = subjectID
        self.dataset_dir = f"{dataset_path}/S{self.ID:03d}"
        self.session = []
        self.session.append(self.Session(f"{self.dataset_dir}/session_1"))
        self.session.append(self.Session(f"{self.dataset_dir}/session_2"))
    class Session():
        def __init__(self, dataset_dir: str) -> None:
            self.dataset_dir = dataset_dir
            self.resting1 = self.get_resting(1)
            self.task = self.get_task()
            self.resting2 = self.get_resting(2)
            self.response = self.get_response()
        
        def get_resting(self, id: int) -> dict:
            if not os.path.exists(f"{self.dataset_dir}/resting{id}_epoch.mat"):
                return None
            rest_data = loadmat(f"{self.dataset_dir}/resting{id}_epoch.mat")["epoch_data"]
            return {resting_type[0]: data.astype(np.float32) for resting_type, data in zip(rest_data["type"][0], rest_data["resting"][0])}
        
        def get_task(self) -> np.ndarray:
            if not os.path.exists(f"{self.dataset_dir}/task_epoch.mat"):
                return None
            task_data = loadmat(f"{self.dataset_dir}/task_epoch.mat", squeeze_me=True)["epoch_data"]
            target = np.stack(task_data["type"]).astype(np.int32)
            fixation = np.stack(task_data["fixation"]).astype(np.float32)
            stimuli = np.stack(task_data["stimuli"]).astype(np.float32)
            imagery = np.stack(task_data["imagery"]).astype(np.float32)
            return {"target": target, "fixation": fixation, "stimuli": stimuli, "imagery": imagery}
            
        def get_response(self)-> np.ndarray:
            if not os.path.exists(f"{self.dataset_dir}/response.txt"):
                return None
            return np.loadtxt(f"{self.dataset_dir}/response.txt")[:, 1:]

class YOTO_data():
    def __init__(self,  
                 data_setting: list,  
                 window_size: int = 1,
                 stride:int = 0,
                 loffset: int = 0,
                 roffset: int = 0,
                 srate: int = 250) -> None:
        """_summary_
        Save the identifier of each data segment.
        Args:
            data_key (str): "resting" | "fixation" | "stimuli" | "imagery".
            window_size (int, optional): Window size in the unit of seconds. Defaults to 1.
            stride (int, optional): Stride of sliding window in the unit of seconds. Defaults to 0.
            loffset (int, optional): Offset at the beginning. Defaults to 0.
            roffset (int, optional): Offset at the end. Defaults to 0.
            srate (int, optional): Sampling rate of the data. Defaults to 250.
        """
        self.data_setting = deepcopy(data_setting)
        self.data_key = self.data_setting[2][0].split("_")[0]
        self.epoch_dir = "resting_epoch" if self.data_key == "resting" else "task_epoch"
        self.srate = srate
        self.data_idxs = np.zeros((0, 5), dtype=int)  # Before calling sliding window: [subject, session, trial, loffset, roffset]
                                                      # After calling sliding window: [subject, session, trial, offset]
        self.window_size = int(window_size * srate)
        self.stride = int(stride * srate)
        self.loffset = int(loffset * srate)
        self.roffset = int(roffset * srate)
    
    def __len__(self) -> int:
        return len(self.data_idxs)
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        subject, session, trial, offset = self.data_idxs[idx]
        # data = loadmat(f"{dataset_path}/S{subject:03d}/session_{session}/{self.epoch_dir}/trial_{trial}_{self.data_key}", squeeze_me=True)
        with open(f"{dataset_path}/S{subject:03d}/session_{session}/{self.epoch_dir}/trial_{trial}_{self.data_key}.pickle", "rb") as f:
            data = pickle.load(f)
        return data["data"][:, offset:offset+self.window_size], data["trigger"]
    
    def add_data(self, 
                 subject: int, 
                 session: int, 
                 trigger: Union[int, Iterable],
                 trial: int = 0,
                 vivid_thres: int = 0) -> None:
        """_summary_
            Find corresponding epoch files and add to the data_idxs
        Args:
            subject (int): Subject ID
            session (int): Session ID
            trigger (Union[int, Iterable]): Accepted triggers
            trial (int, optional): Trial ID, 0 indicates all trials. Defaults to 0.
            vivid_thres (int, optional): threshold of vividness to accept a trial. Defaults to 0.
        Return:
            count: Number of data added
        """
        root = f"{dataset_path}/S{subject:03d}/session_{session}/{self.epoch_dir}"
        count = 0
        if isinstance(trigger, int):
            trigger = [trigger]
        if len(trigger) == 0:
            return 0
        if os.path.exists(root):
            root = Path(root)
            if trial == 0:
                # pattern = f"trial_[0-9]*_{self.data_key}.mat"
                pattern = f"trial_[0-9]*_{self.data_key}.pickle"
            else:
                # pattern = f"trial_{trial}_{self.data_key}.mat"
                pattern = f"trial_{trial}_{self.data_key}.pickle"
            for file in root.glob(pattern):
                # data = loadmat(file, appendmat=False, squeeze_me=True)
                with open(file, "rb") as f:
                    data = pickle.load(f)
                trial = int(file.stem.split("_")[1])
                if data["trigger"] not in trigger:
                    continue
                if "vividness" in data and data["vividness"] < vivid_thres:
                    continue
                count += 1
                self.data_idxs = np.vstack((self.data_idxs, [[subject, session, trial, self.loffset, self.roffset]]))
        return count
    
    def sliding_window(self) -> None:
        """_summary_
            Calculate the sliding window of each record in self.data_idxs.
        Returns:
            None
        """
        if self.data_idxs.shape[1] != 5:  # already called
            return
        new_data_idxs = np.zeros((0, 4), dtype=int)  # [subject, session, trial, offset]
        
        idx = None        
        for subject, session, trial, _, _ in self.data_idxs:
            if idx is None:
                # filepath = f"{dataset_path}/S{subject:03d}/session_{session}/{self.epoch_dir}/trial_{trial}_{self.data_key}.mat"
                filepath = f"{dataset_path}/S{subject:03d}/session_{session}/{self.epoch_dir}/trial_{trial}_{self.data_key}.pickle"
                # data = loadmat(filepath, appendmat=False, squeeze_me=True)
                with open(filepath, "rb") as f:
                    data = pickle.load(f)
                if self.stride:
                    idx = np.arange(self.loffset, data["data"].shape[1]-self.roffset-self.window_size+1, self.stride)
                else:
                    idx = np.array([self.loffset])
            data_idxs = np.repeat([[subject, session, trial, 0]], len(idx), axis=0)
            data_idxs[:, -1] = idx
            new_data_idxs = np.vstack((new_data_idxs, data_idxs))
        self.data_idxs = new_data_idxs
        return
        
    def train_test_split(self, 
                         valid_ratio: float = 0.3, 
                         custom_random: np.ndarray = None,
                         shuffle: bool = True) -> None:
        """_summary_
            Split the data into tarin and test parts.
        Args:
            ratio (float, optional): Ratio of validation data. Defaults to 0.3.
            custom_random (np.ndarray, optional): If is not None, use the given index as the random shuffle index. Defaults to None.
            shuffle (bool, optional): Whether to shuffle the data order. If custom_random is given right, this is ignored.
        Returns:
            None
        """
        train = deepcopy(self)
        test = deepcopy(self)
        if custom_random and len(custom_random) == len(self.data_idxs):
            self.data_idxs = self.data_idxs[custom_random]
        elif shuffle:
            np.random.shuffle(self.data_idxs)

        train_len = int(len(self.data_idxs) * (1 - valid_ratio))
        train.data_idxs = self.data_idxs[:train_len]
        test.data_idxs = self.data_idxs[train_len:]
        return train, test
        
class YOTO_task_dataset(Dataset):
    def __init__(self, args, mode, nchannel=30) -> None:
        super().__init__()
        assert mode == "train" or mode == "test" or mode == "finetune"
        assert args.scheme == "ID" or args.scheme == "SI" or args.scheme == "SD" or args.scheme == "SIFT"
        assert args.subject != None
        
        self.scheme = args.scheme
        self.test_subject = args.subject
        self.nchannel = nchannel
        self.mode = mode
        self.dck = args.dataset_config_key
        self.window_size = args.window_size
        
        # get target space and classes
        self.dc = deepcopy(dataset_config[self.dck])
        len_total = int(sum(self.dc["segment_lens"]) * 250)
        self.dc["segment_lens"] = [int(l*250) for l in self.dc["segment_lens"][:-1]]  # srate
        if len_total > sum(self.dc["segment_lens"]):
            self.dc["segment_lens"].append(len_total - sum(self.dc["segment_lens"]))
        self.trigger_space = filt_trigger(*self.dc["space"])
        self.nclass = len(self.dc["classes"])
        self.data_bank = [YOTO_data(*data_setting) for data_class in self.dc["classes"] for data_setting in data_class]
        self.db_class = [i for i, c in enumerate(self.dc["classes"]) for _ in range(len(c))]
        for i, _ in enumerate(self.data_bank):
            self.__load_data(self.data_bank[i])
        self.data_idxs = []    
    
    def get_dataset(self, 
                    valid_ratio: float = 0., 
                    shuffle: bool = True) -> Tuple["YOTO_task_dataset", "YOTO_task_dataset"]:
        train_dataset = deepcopy(self)
        train_dataset.data_bank = []
        test_dataset = deepcopy(self)
        test_dataset.data_bank = []
        
        for db in self.data_bank:
            train, test = db.train_test_split(valid_ratio=valid_ratio if db.data_key != "resting" else 0, shuffle=shuffle)
            train_dataset.data_bank.append(train)
            train_dataset.data_bank[-1].sliding_window()
            test_dataset.data_bank.append(test)
            # if test_dataset.data_bank[-1].stride: 
            #     test_dataset.data_bank[-1].stride = 1 * test_dataset.data_bank[-1].srate
            test_dataset.data_bank[-1].sliding_window()
            
        train_dataset.generate_index(shuffle=shuffle, mix=self.dc["db_mix"] != 0 or self.dc["class_mix"] != 0 or self.dc["cross_class_mix"] != 0)
        test_dataset.generate_index(shuffle=shuffle)
        return train_dataset, test_dataset
    
    def generate_index(self, shuffle=True, mix=False) -> None:
        id_shape = (len(self.dc["segment_lens"]), 2) # if mix else (1, 2) # 2 -> [data bank index, data id in the data bank]
        self.data_idxs = np.zeros((0, *id_shape), dtype=int)  
        for i, db in enumerate(self.data_bank):
            idxs = np.zeros((len(db), *id_shape), dtype=int)
            idxs[:, :, 0] = i
            idxs[:, :, 1] = np.arange(len(db)).repeat(id_shape[0]).reshape(-1, id_shape[0])
            self.data_idxs = np.vstack((self.data_idxs, idxs))
        
        if mix:
            if self.dc["db_mix"]:
                st = 0
                for i, db in enumerate(self.data_bank):  # mix segments in a data bank
                    if len(db) < 1:
                        continue
                    rnd_idxs = np.random.randint(st, st+len(db), size=self.dc["db_mix"] * id_shape[0])
                    self.data_idxs = np.vstack((self.data_idxs, self.data_idxs[rnd_idxs, 0, :].reshape(-1, *id_shape)))                
                    st += len(db)
                    print(f"\tData bank {i:02d}: Add {self.dc['db_mix']} mixed instances. (Same data bank)")
            
            class_len = np.zeros(self.nclass, dtype=int)
            for i, db in enumerate(self.data_bank):  # count size of each class
                class_len[self.db_class[i]] += len(db)
            
            if self.dc["class_mix"]:
                st = 0    
                for i in range(self.nclass):  # mix segments in a class
                    if class_len[i] < 1:
                        continue
                    rnd_idxs = np.random.randint(st, st+class_len[i], size=self.dc["class_mix"] * id_shape[0])
                    self.data_idxs = np.vstack((self.data_idxs, self.data_idxs[rnd_idxs, 0, :].reshape(-1, *id_shape)))
                    st += class_len[i]
                    print(f"\tClass {i:02d}: Add {self.dc['class_mix']} mixed instances. (Same class)")
            
            if self.dc["cross_class_mix"]:
                st = 0
                pool = np.zeros(0, dtype=int)
                for i in range(self.nclass):  # mix segments in a class
                    if class_len[i] < 1:
                        continue
                    if class_len[i] == np.sum(class_len):
                        break
                    rnd_idxs = np.random.randint(st, st+class_len[i], size=(self.dc["cross_class_mix"], (id_shape[0] - 1)))  # Not good
                    rnd_idxs = np.concatenate((rnd_idxs, 
                                            np.random.choice(np.concatenate((pool, 
                                                                                np.arange(st+class_len[i], np.sum(class_len)))), 
                                                                replace=True, size=(self.dc["cross_class_mix"], 1))), axis=1)
                    rnd_idxs = np.take_along_axis(rnd_idxs, np.random.rand(*rnd_idxs.shape).argsort(axis=1), axis=1).reshape(-1)
                    self.data_idxs = np.vstack((self.data_idxs, self.data_idxs[rnd_idxs, 0, :].reshape(-1, *id_shape)))
                    pool = np.concatenate((pool, np.arange(st, st+class_len[i])))
                    st += class_len[i]
                    print(f"\tClass {i:02d}: Add {self.dc['class_mix']} mixed instances. (Different classes)")
                    
                # for i in range(self.nclass):  # mix segments across classes
                #     rnd_idxs = np.random.randint(0, np.sum(class_len), size=self.dc["cross_class_mix"] * id_shape[0])
                #     self.data_idxs = np.vstack((self.data_idxs, self.data_idxs[rnd_idxs, 0, :].reshape(-1, *id_shape)))        
        if shuffle:
            np.random.shuffle(self.data_idxs)

    def __len__(self) -> int:
        return len(self.data_idxs)
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, int, int]:
        """_summary_
            get data of given index
        Args:
            idx (_type_): data index

        Returns:
            Tuple[np.ndarray, int, int]: signal, class id, trigger
        """
        data = np.zeros((self.nchannel, self.window_size), dtype=np.float32)
        class_id = np.zeros(self.nclass, dtype=int)
        seg_classes = []
        triggers = []
        st = 0
        for i, (db_id, data_id) in enumerate(self.data_idxs[idx]):
            segment, trigger = self.data_bank[db_id][data_id]
            data[:, st:st+self.dc["segment_lens"][i]] = segment[:, st:st+self.dc["segment_lens"][i]]
            st += self.dc["segment_lens"][i]
            seg_classes.append(self.db_class[db_id])
            triggers.append(trigger)
            class_id[self.db_class[db_id]] += self.dc["segment_lens"][i]
            # class_id[self.db_class[db_id]] += 1
        assert st == self.window_size
        return data, np.argmax(class_id), seg_classes, triggers

    def stat(self):
        print(f"Number of instances: {len(self.data_idxs)}")
        print("Number of instances in each data bank:")
        class_count = np.zeros(self.nclass, dtype=int)
        for i, db in enumerate(self.data_bank):
            print(f"\tdata bank {i:02d}: {len(db)}({len(db)/len(self.data_idxs)*100:.2f}%)\tclass {self.db_class[i]:02d}")
            class_count[self.db_class[i]] += len(db)
        print(f"Number of mixed instances : {len(self.data_idxs) - np.sum(class_count)}")
        print("Number of instances in each class:")
        # TODO: need to consider mixed trials
        for db_seq in self.data_idxs[np.sum(class_count):, :, 0]:
            count = np.zeros(self.nclass)
            for i, db_id in enumerate(db_seq):
                count[self.db_class[db_id]] += self.dc["segment_lens"][i]
            class_count[np.argmax(count)] += 1
        for i, n in enumerate(class_count):
            print(f"\tclass {i:02d}: {n}({n/len(self.data_idxs)*100:.2f}%)")
        lcm = math.lcm(*class_count)
        return [lcm / n if n != 0 else 0 for n in class_count]
    
    def __load_data(self, db: "YOTO_data") -> None:
        if self.mode == "test":  # test session data from test subject
            id_pool = {(self.test_subject, 2)}
        elif self.mode == "finetune" or self.scheme == "ID": # SIFT fintune or ID train
            id_pool = {(self.test_subject, 1)}
        else:
            id_pool = {(subject, session) for subject in range(1, 27) for session in range(1, 3)}
            if self.scheme == "SD":
                id_pool.remove((self.test_subject, 2))
            else:  # SI or SIFT
                id_pool.remove((self.test_subject, 1))
                id_pool.remove((self.test_subject, 2))
        accepted_trigger = [int(t_str.split("_")[1]) for t_str in filt_trigger(*db.data_setting, self.trigger_space)]
        # print(db.data_key)
        # print(accepted_trigger)
        for subject, session in id_pool:
            db.add_data(subject=subject, session=session, trigger=accepted_trigger, vivid_thres=0)

class YOTO_Siamese_task_dataset(YOTO_task_dataset):
    def __init__(self, args, mode, nchannel=30) -> None:
        # super().__init__(args, mode, nchannel)
        assert mode == "train" or mode == "test" or mode == "finetune"
        assert args.scheme == "ID" or args.scheme == "SI" or args.scheme == "SD" or args.scheme == "SIFT"
        assert args.subject != None
        
        self.scheme = args.scheme
        self.test_subject = args.subject
        self.nchannel = nchannel
        self.mode = mode
        self.dck = args.dataset_config_key
        self.window_size = args.window_size
        
        # get target space and classes
        self.dc = deepcopy(dataset_config[self.dck])
        len_total = int(sum(self.dc["segment_lens"]) * 250)
        self.dc["segment_lens"] = [int(l*250) for l in self.dc["segment_lens"][:-1]]  # srate
        if len_total > sum(self.dc["segment_lens"]):
            self.dc["segment_lens"].append(len_total - sum(self.dc["segment_lens"]))
        self.trigger_space = filt_trigger(*self.dc["space"])
        self.nclass = len(self.dc["classes"])
        self.data_bank = [YOTO_data(*data_setting) for data_class in self.dc["classes"] for data_setting in data_class]
        self.db_class = [i for i, c in enumerate(self.dc["classes"]) for _ in range(len(c))]
        for i, _ in enumerate(self.data_bank):
            self.__load_data(self.data_bank[i])
        self.data_idxs = []  
        # self.resting = dict()
    
    def __load_data(self, db: "YOTO_data") -> None:
        if self.mode == "test":  # test session data from test subject
            id_pool = {(self.test_subject, 2)}
        elif self.mode == "finetune" or self.scheme == "ID": # SIFT fintune or ID train
            id_pool = {(self.test_subject, 1)}
        else:
            id_pool = {(subject, session) for subject in range(1, 27) for session in range(1, 3)}
            if self.scheme == "SD":
                id_pool.remove((self.test_subject, 2))
            else:  # SI or SIFT
                id_pool.remove((self.test_subject, 1))
                id_pool.remove((self.test_subject, 2))
        accepted_trigger = [int(t_str.split("_")[1]) for t_str in filt_trigger(*db.data_setting, self.trigger_space)]
        for subject, session in id_pool:
            if os.path.exists(f"{dataset_path}/S{subject:03d}/session_{session}/resting_epoch/trial_2_resting.pickle"):
                db.add_data(subject=subject, session=session, trigger=accepted_trigger, vivid_thres=0)
    
    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """_summary_
            Get data by index
        Args:
            idx (_type_): index of data
        Returns:
            Tuple[np.ndarray, np.ndarray, int, int]: [eyeopen resting at beginning, signal to classify, class id, trigger id]
        """
        data = np.zeros((self.nchannel, self.window_size), dtype=np.float32)
        class_id = np.zeros(self.nclass, dtype=int)
        seg_classes = []
        triggers = []
        st = 0
        subject = -1
        session = -1
        for i, (db_id, data_id) in enumerate(self.data_idxs[idx]):
            segment, trigger = self.data_bank[db_id][data_id]
            data[:, st:st+self.dc["segment_lens"][i]] = segment[:, st:st+self.dc["segment_lens"][i]]
            st += self.dc["segment_lens"][i]
            seg_classes.append(self.db_class[db_id])
            triggers.append(trigger)
            class_id[self.db_class[db_id]] += self.dc["segment_lens"][i]
            # TODO cross subject resting branch choice
            subject, session, _, _ = self.data_bank[db_id].data_idxs[data_id]
            
        with open(f"{dataset_path}/S{subject:03d}/session_{session}/resting_epoch/trial_2_resting.pickle", "rb") as f:  # eye open resting
            t = np.random.randint(1, 30) * self.data_bank[db_id].srate
            resting_data = pickle.load(f)["data"][:, t:t+self.data_bank[db_id].srate]
            # resting_data = pickle.load(f)["data"][:, self.data_bank[db_id].srate:]    
        return resting_data, data, np.argmax(class_id), seg_classes, triggers
        # return resting_data, data, self.db_class[db_id], trigger
        
        # Try cache
        # if f"{subject:03d}_{session}" in self.resting:
        #     resting_data = self.resting[f"{subject:03d}_{session}"]
        # else:
        #     resting_data = loadmat(f"{dataset_path}/S{subject:03d}/session_{session}/resting_epoch/trial_2_resting.mat", squeeze_me=True)["data"][:, self.data_bank[db_id].srate:]
        #     self.resting[f"{subject:03d}_{session}"] = resting_data
        # resting_data = loadmat(f"{dataset_path}/S{subject:03d}/session_{session}/resting_epoch/trial_2_resting.mat", squeeze_me=True)["data"][:, self.data_bank[db_id].srate:]
        
        
              
# class YOTO_task_dataset():
#     def __init__(self, args, mode, nchannel=30, sliding=False) -> None:
#         super().__init__()
#         assert mode == "train" or mode == "test" or mode == "finetune"
#         assert args.scheme == "ID" or args.scheme == "SI" or args.scheme == "SD" or args.scheme == "SIFT"
#         assert args.subject != None
        
#         self.scheme = args.scheme
#         self.test_subject = args.subject
#         self.data_key = []
#         for key in ["fixation", "stimuli", "imagery"]:
#             if key in list(args.data_key):
#                 self.data_key += [key]
#         self.nchannel = nchannel
#         self.sliding = sliding
#         self.window_size = args.window_size
#         self.stride = args.window_stride
#         self.mode = mode
#         self.dck = args.dataset_config_key

#         # get target space and classes
#         dc = deepcopy(dataset_config[self.dck])
#         self.trigger_space = filt_trigger(*dc["space"])
        
#         # load data
#         len_map = {"fixation": 500, "stimuli": 500, "imagery": 1000}
#         self.signals = np.array([]).reshape(0, self.nchannel, np.sum([len_map[data_key] for data_key in self.data_key]))
#         self.triggers = torch.zeros(0, dtype=int)
#         self.__load_data()
#         print(f"Total number of trials: {len(self.signals)}")
    
#     def get_dataset(self, valid_ratio=0., shuffle=False):
#         ids = np.arange(len(self.signals))
#         if shuffle:
#             np.random.shuffle(ids)
#         train_len = int(len(self.signals) * (1 - valid_ratio))
#         train = _YOTO_task_dataset(self.signals[ids[:train_len]], self.triggers[ids[:train_len]], self.data_key, self.dck, 
#                                    self.sliding, self.window_size, self.stride)
#         if train_len != len(ids):
#             valid = _YOTO_task_dataset(self.signals[ids[train_len:]], self.triggers[ids[train_len:]], self.data_key, self.dck,
#                                        self.sliding, self.window_size, self.stride)
#             return train, valid
#         return train
    
#     def __load_data(self):
#         def add_task_data(data, mask):
#             self.signals = np.vstack([self.signals, np.concatenate([data[data_key][mask] for data_key in self.data_key], axis=2)])
#             self.triggers = np.concatenate([self.triggers, data["target"][mask]], axis=0)
#         def add_resting_data(data):
#             if self.resting:
#                 self.resting = np.vstack([self.resting, np.vstack([v for k, v in data])])
            
        
#         # iterate all data
#         for subject_id in range(1, 27):
#             if (self.mode == "test" or self.scheme == "ID") and subject_id != self.test_subject:
#                 continue
#             subject = Subject(subject_id)
#             for session_id in range(2):
#                 if subject.session[session_id].task is None:
#                     continue
#                 task_data = subject.session[session_id].task
#                 response = subject.session[session_id].response
                
#                 mask = [tt in self.trigger_space and resp >= 0 for tt, resp in zip(task_data["target"], response[:, 1])]
#                 if subject_id != self.test_subject:
#                     if self.scheme != "ID" and self.mode == "train":  # all session data from other subjects
#                         add_task_data(task_data, mask)
#                         add_resting_data(resting1)
#                         add_resting_data(resting2)
#                 else:
#                     if self.mode == "test" and session_id == 1:  # test session data from test subject
#                         add_task_data(task_data, mask)
#                         add_resting_data(resting1)
#                         add_resting_data(resting2)
#                         return
#                     elif self.mode == "train" and self.scheme in ["ID", "SD"] and session_id == 0:  # train session data from test subject 
#                         add_task_data(task_data, mask)
#                         add_resting_data(resting1)
#                         add_resting_data(resting2)
#                     elif self.mode == "finetune" and self.scheme == "SIFT" and session_id == 0:  # train session data from test subject
#                         add_task_data(task_data, mask)
#                         add_resting_data(resting1)
#                         add_resting_data(resting2)
#                         return

      
# class _YOTO_task_dataset(Dataset):
#     def __init__(self, signals, triggers, resting, data_key, dck, sliding=False, window_size=250, stride=25) -> None:
#         self.signals = torch.FloatTensor(signals)
#         self.triggers = triggers
#         self.resting = resting
#         self.data_key = data_key.copy()
#         self.sliding = sliding  # currently not used
#         self.window_size = window_size
#         self.stride = stride
        
#         self.rearrange()
        
#         self.data_addr = np.zeros((0, 2), dtype=int)  # [trial id, st]
#         self.trigger_target = np.zeros(0, dtype=int)
#         self.signal2addr(dck)
        
#         # get target space and classes
#         dc = deepcopy(dataset_config[dck])
#         self.trigger_space = filt_trigger(*dc["space"])
#         self.classes = [filt_trigger(*c, self.trigger_space) for c in dc["classes"]]
#         assert len(self.classes) > 1
        
#         # map trigger label to class label
#         self.class_target = torch.zeros(self.trigger_target.shape, dtype=int)
#         self.class_count = np.zeros((len(self.classes)), dtype=int)
#         for i, t in enumerate(self.trigger_target):
#             for ci, c in enumerate(self.classes):
#                 if t in c:
#                     self.class_target[i] = ci
#                     self.class_count[ci] += 1
    
#     def rearrange(self):
#         tail = 0
#         for p, v in enumerate(self.triggers):
#             if "visual" not in trigger_info[v]:
#                 self.signals[[tail, p]] = self.signals[[p, tail]]
#                 self.triggers[[tail, p]] = self.triggers[[p, tail]]
#                 tail += 1
    
#     def signal2addr(self, dck):
#         if dck == "sequential_plot":
#             self.data_addr = np.zeros((len(self.signals), 2), dtype=int)
#             self.data_addr[:, 0] = np.arange(len(self.signals))
#             self.trigger_target = self.triggers
#             self.window_size = self.signals.shape[-1]
#             return
        
#         def add_data(addr, target):
#             self.data_addr = np.vstack([self.data_addr, addr])
#             self.trigger_target = np.concatenate([self.trigger_target, target], axis=0)
            
#         offset = 0
#         for key in self.data_key:
#             if key == "fixation":
#                 offset += 250  # start at 1s to avoid fixation ERP
#                 addr = np.zeros((len(self.signals), 2), dtype=int)
#                 addr[:, 0] = np.arange(len(self.signals))
#                 addr[:, 1] = offset
#                 add_data(addr, np.zeros(len(self.signals)) + 6)
#                 offset += 250  # end of fixation
#             elif key == "stimuli":
#                 addr = np.zeros((len(self.signals), 2), dtype=int)
#                 addr[:, 0] = np.arange(len(self.signals))
#                 addr[:, 1] = offset
#                 add_data(addr, self.triggers)
#                 offset += 500  # end of stimuli
#             elif key == "imagery":
#                 visual_begin = 0
#                 while visual_begin < len(self.triggers):
#                     if "visual" in trigger_info[self.triggers[visual_begin]]:
#                         break
#                     visual_begin += 1
#                 # only auditory
#                 addr = np.zeros((visual_begin, 2), dtype=int)
#                 addr[:, 0] = np.arange(len(addr))
#                 for st in range(0, 1000-self.window_size+1, self.stride):
#                     addr[:, 1] = offset + st
#                     add_data(addr, self.triggers[:visual_begin])
#                 # others that contain visual
#                 offset += 250  # avoid visual ERP
#                 addr = np.zeros((len(self.signals) - visual_begin, 2), dtype=int)
#                 addr[:, 0] = np.arange(len(addr)) + visual_begin
#                 for st in range(0, 750-self.window_size+1, self.stride):
#                     addr[:, 1] = offset + st
#                     add_data(addr, self.triggers[visual_begin:])
#                 offset += 750  # end of imagery
#         ids = np.arange(len(self.data_addr))
#         np.random.shuffle(ids)
#         self.data_addr = self.data_addr[ids]
#         self.trigger_target = self.trigger_target[ids]
    
#     def stat(self):
#         print(f"Number of trials: {len(self.signals)}")
#         print(f"Number of instances: {len(self.data_addr)}")
#         print("Instances in each classes:")
#         total = np.sum(self.class_count)
#         for c, n in zip(self.classes, self.class_count):
#             print(f"\t{c}: {n}({n/total*100:.2f}%)")
#         lcm = math.lcm(*self.class_count[self.class_count!=0])
#         return [lcm / cc if cc != 0 else 0 for cc in self.class_count]
        
#     def __len__(self) -> int:
#         return len(self.data_addr)
#     def __getitem__(self, id):
#         return self.signals[self.data_addr[id][0], :, self.data_addr[id][1]:self.data_addr[id][1]+self.window_size], self.class_target[id], \
#             self.trigger_target[id],  self.data_addr[id]