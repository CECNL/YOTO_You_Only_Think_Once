from pathlib import Path
import re
raw_data_path = str((Path(__file__).parent.resolve() / "../../raw_data").resolve())
# dataset_path = str((Path(__file__).parent.resolve() / "../../dataset_noAvgRef").resolve())
# dataset_path = str((Path(__file__).parent.resolve() / "../../dataset_noAvgRef_cleegn").resolve())
dataset_path = str((Path(__file__).parent.resolve() / "../../dataset_noAvgRef_ASR_ICA").resolve())
# dataset_path = str((Path(__file__).parent.resolve() / "../../dataset_noAvgRef_rtCLEEGN").resolve())
trigger_info = {
    3: ["resting", "close"],
    5: ["resting", "open"],
    6: ["fixation"],
    21: ["visual", "face", "male"],
    22: ["visual", "face", "female"],
    23: ["visual", "square"],
    24: ["auditory", "speech", "a"],
    25: ["auditory", "speech", "o"],
    26: ["auditory", "speech", "i"],
    27: ["auditory", "music", "C"],
    28: ["auditory", "music", "D"],
    29: ["auditory", "music", "E"],
    30: ["mix", "visual", "auditory", "square", "speech", "a"],
    31: ["mix", "visual", "auditory", "square", "speech", "o"],
    32: ["mix", "visual", "auditory", "square", "speech", "i"],
    33: ["mix", "visual", "auditory", "square", "music", "C"],
    34: ["mix", "visual", "auditory", "square", "music", "D"],
    35: ["mix", "visual", "auditory", "square", "music", "E"],
    36: ["mix", "visual", "auditory", "face", "male", "speech", "a"],
    37: ["mix", "visual", "auditory", "face", "male", "speech", "o"],
    38: ["mix", "visual", "auditory", "face", "male", "speech", "i"],
    39: ["mix", "visual", "auditory", "face", "female", "speech", "a"],
    40: ["mix", "visual", "auditory", "face", "female", "speech", "o"],
    41: ["mix", "visual", "auditory", "face", "female", "speech", "i"],
    42: ["mix", "visual", "auditory", "face", "male", "music", "C"],
    43: ["mix", "visual", "auditory", "face", "male", "music", "D"],
    44: ["mix", "visual", "auditory", "face", "male", "music", "E"],
    45: ["mix", "visual", "auditory", "face", "female", "music", "C"],
    46: ["mix", "visual", "auditory", "face", "female", "music", "D"],
    47: ["mix", "visual", "auditory", "face", "female", "music", "E"],
}
channel_order = ["Fp1","Fp2","F7","F3","Fz","F4","F8","FT7","FC3","FCz",
                 "FC4","FT8","T7","C3","Cz","C4","T8","TP7","CP3","CPz",
                 "CP4","TP8","P7","P3","Pz","P4","P8","O1","Oz","O2"]
channel_plot_order = [["O1","Fp1","Oz","Fp2","O2"],
                      ["F7","F3","Fz","F4","F8"],
                      ["FT7","FC3","FCz","FC4","FT8"],
                      ["T7","C3","Cz","C4","T8"],
                      ["TP7","CP3","CPz", "CP4","TP8"],
                      ["P7","P3","Pz","P4","P8"]]

def filt_trigger(include_trigger: list[str], exclude_trigger: list[str], include_category: list[str], exclude_category: list[str], space: list[str] = None) -> list[str]:
    assert len(set(include_trigger) & set(exclude_trigger)) == 0
    assert len(set(include_category) & set(exclude_category)) == 0
    valid_keys = []
    for trig, info in trigger_info.items():
        if "resting" in info:
            valid_keys.append(f"resting_{trig}")
        elif "fixation" in info:
            valid_keys.append(f"fixation_{trig}")
        else:
            valid_keys.append(f"stimuli_{trig}")
            valid_keys.append(f"imagery_{trig}")
    
    if space is None:  # include all trigger    
        space = valid_keys
    
    def expand_category(categories: list[str]) -> set[str]:
        expand = []
        for k in valid_keys:
            key_cat, key_trig = k.split("_")[:2]
            key_trig = int(key_trig)
            for category in categories:
                cat, subcat = category.split("_")[:2]
                if key_cat == cat and subcat in trigger_info[key_trig]:
                    expand.append(k)
        return set(expand)

    return list(set(include_trigger) | (expand_category(include_category) - expand_category(exclude_category)))

layout = {
    "Traning Record": {
        "loss": ["Multiline", ["loss/train", "loss/validation"]],
        "accuracy": ["Multiline", ["accuracy/train", "accuracy/validation"]],
    },
}

dataset_config = {
    "all_F_S": {
        "space": [[], [], ["resting_resting", "fixation_fixation", "stimuli_auditory", "stimuli_visual"], []],
        "classes": [
            [
                [[[], [], ["fixation_fixation"], []], 1, 0, 1, 0],
            ],
            [
                [[[], [], ["stimuli_auditory"], ["stimuli_mix"]], 1, 0, 0, 1],
                [[[], [], ["stimuli_visual"], ["stimuli_mix"]], 1, 0, 0, 1],
                [[[], [], ["stimuli_mix"], []], 1, 0, 0, 1],
            ]
        ], 
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
    "all_F_I": {
        "space": [[], [], ["fixation_fixation", "imagery_auditory", "imagery_visual"], []],
        "classes": [
            [
                [[[], [], ["fixation_fixation"], []], 1, 0, 1, 0],
            ],
            [
                [[[], [], ["imagery_auditory"], ["imagery_mix"]], 1, 0.2, 1, 0],
                [[[], [], ["imagery_visual"], ["imagery_mix"]], 1, 0.2, 1, 0],
                [[[], [], ["imagery_mix"], []], 1, 0.2, 1, 0],
            ]
        ], 
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
    "all_F_Iavm": {
        "space": [[], [], ["fixation_fixation", "imagery_auditory", "imagery_visual"], []],
        "classes": [
            [
                [[[], [], ["fixation_fixation"], []], 1, 0, 1, 0],
            ],
            [
                [[[], [], ["imagery_auditory"], ["imagery_mix"]], 1, 0.2, 1, 0]
            ],
            [
                [[[], [], ["imagery_visual"], ["imagery_mix"]], 1, 0.2, 1, 0]
            ],
            [
                [[[], [], ["imagery_mix"], []], 1, 0.2, 1, 0]
            ]
        ], 
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
    "all_Savm": {
        "space": [[], [], ["fixation_fixation", "imagery_auditory", "imagery_visual"], []],
        "classes": [
            [
                [[[], [], ["stimuli_auditory"], ["stimuli_mix"]], 1, 0, 0, 1]
            ],
            [
                [[[], [], ["stimuli_visual"], ["stimuli_mix"]], 1, 0, 0, 1]
            ],
            [
                [[[], [], ["stimuli_mix"], []], 1, 0, 0, 1]
            ]
        ], 
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
    "all_Iavm": {
        "space": [[], [], ["fixation_fixation", "imagery_auditory", "imagery_visual"], []],
        "classes": [
            [
                [[[], [], ["imagery_auditory"], ["imagery_mix"]], 1, 0.2, 0, 0]
            ],
            [
                [[[], [], ["imagery_visual"], ["imagery_mix"]], 1, 0.2, 1, 0]
            ],
            [
                [[[], [], ["imagery_mix"], []], 1, 0.2, 1, 0]
            ]
        ], 
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
    "all_FR_S": {
        "space": [[], [], ["resting_resting", "fixation_fixation", "stimuli_auditory", "stimuli_visual"], []],
        "classes": [
            [
                [[[], [], ["fixation_fixation"], []], 1, 0, 1, 0],
                [[[], [], ["resting_close"], []], 1, 0.1, 1, 0],
                [[[], [], ["resting_open"], []], 1, 0.1, 1, 0],
            ],
            [
                [[[], [], ["stimuli_auditory"], ["stimuli_mix"]], 1, 0, 0, 1],
                [[[], [], ["stimuli_visual"], ["stimuli_mix"]], 1, 0, 0, 1],
                [[[], [], ["stimuli_mix"], []], 1, 0, 0, 1],
            ]
        ], 
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
    "all_FR_I": {
        "space": [[], [], ["resting_resting", "fixation_fixation", "imagery_auditory", "imagery_visual"], []],
        "classes": [
            [
                [[[], [], ["fixation_fixation"], []], 1, 0, 1, 0],
                [[[], [], ["resting_close"], []], 1, 0.1, 1, 0],
                [[[], [], ["resting_open"], []], 1, 0.1, 1, 0],
            ],
            [
                [[[], [], ["imagery_auditory"], ["imagery_mix"]], 1, 0.1, 0, 0],
                [[[], [], ["imagery_visual"], ["imagery_mix"]], 1, 0.1, 1, 0],
                [[[], [], ["imagery_mix"], []], 1, 0.1, 1, 0],
            ]
        ],
        "db_mix": 0,
        "class_mix": 0,
        "cross_class_mix": 0,
        "segment_lens": [1]
    },
}