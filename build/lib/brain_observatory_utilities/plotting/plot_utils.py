# Plotting style enforcment dictionaries
# to provide colors and labels for different
# data dimensions. Colors are all hexidecimel


DATASTREAM_STYLE_DICT = {
    "licks":                  {"color": "#252525",
                               "label": "licks"},
    "rewards":                {"color": "#6baed6",
                               "label": "rewards"},
    "pupil_area":             {"color": "#fdc086",
                               "label": "pupil area"},
    "dff":                    {"color": "#7fc97f",
                               "label":  "df/f"},
    "running_speed":          {"color": "#beaed4",
                               "label": "running speed (cm/sec)"},
    "stimulus_presentations": {"color": "#cccccc",
                               "label": "stim presentations"}
}
REWARDS_STYLE_DICT = {
    "all":    {"color": "#6baed6", "label": "rewards (all)"},
    "earned": {"color": "#08519c", "label": "rewards (earned)"},
    "auto":   {"color": "#c6dbef", "label": "rewards (auto)"}
}
CRELINE_STYLE_DICT = {
    "Sst-IRES-Cre":      {"color": "#b35806", "label": "Sst"},
    "Vip-IRES-Cre":      {"color": "#fdb863", "label": "Vip"},
    "Slc17a7-IRES2-Cre": {"color": "#8073ac", "label": "Slc"}
}

NOVELTY_STYLE_DICT = {
    "TRAINING_0": {"color": "#525252", "label": "training: gratings"},
    "TRAINING_1": {"color": "#525252", "label": "training: gratings"},
    "TRAINING_2": {"color": "#525252", "label": "training: gratings"},
    "TRAINING_3": {"color": "#bdbdbd", "label": "training: images"},
    "TRAINING_4": {"color": "#bdbdbd", "label": "training: images"},
    "TRAINING_5": {"color": "#bdbdbd", "label": "training: images"},
    "OPHYS_0":    {"color": "#f7f7f7", "label": "habituation"},
    "OPHYS_1":    {"color": "#2166ac", "label": "familiar 1"},
    "OPHYS_2":    {"color": "#67a9cf", "label": "familiar 2"},
    "OPHYS_3":    {"color": "#d1e5f0", "label": "familiar 3"},
    "OPHYS_4":    {"color": "#b2182b", "label": "novel 1"},
    "OPHYS_5":    {"color": "#ef8a62", "label": "novel 2"},
    "OPHYS_6":    {"color": "#fddbc7", "label": "novel 3"}
}

TRAINING_STYLE_DICT = {
    "TRAINING_0": {"color": "#525252", "label": "TRAINING 0 gratings"},
    "TRAINING_1": {"color": "#737373", "label": "TRAINING 1 gratings"},
    "TRAINING_2": {"color": "#969696", "label": "TRAINING 2 gratings"},
    "TRAINING_3": {"color": "#bdbdbd", "label": "TRAINING 3 images"},
    "TRAINING_4": {"color": "#d9d9d9", "label": "TRAINING 4 images"},
    "TRAINING_5": {"color": "#f7f7f7", "label": "TRAINING 5 images"}
}


IMAGE_SET_STYLE_DICT = {
    "A": {"color": "#a6611a", "label": "A"},
    "B": {"color": "#018571", "label": "B"},
    "G": {"color": "#dfc27d", "label": "G"},
    "H": {"color": "#80cdc1", "label": "H"},
    "gratings": {"color": "#737373", "label": "gratings"}
}

BEHAV_RESP_STYLE_DICT = {
    "hit":            {"color": "#4dac26", "label": "hit"},
    "miss":           {"color": "#d01c8b", "label": "miss"},
    "false_alarm":    {"color": "#f1b6da", "label": "false alarm"},
    "correct_reject": {"color": "#b8e186", "label": "correct reject"},
    "aborted":        {"color": "#525252", "label": "aborted"}
}


def get_color_for_dict_key(dictionary, key):
    color = dictionary[key]['color']
    return color


def get_label_for_dict_key(dictionary, key):
    label = dictionary[key]['label']
    return label


def get_style_for_stim_name(dictionary, stimulus_name):
    dict_keys = dictionary.keys()
    for key in dict_keys:
        if key in stimulus_name:
            return dictionary[key]["color"], dictionary[key]["label"]
        else:
            print("Stimulus name cannot be aligned dictionary keys.")


def get_image_set_style_for_stim_name(stimulus_name):
    return get_style_for_stim_name(IMAGE_SET_STYLE_DICT,
                                   stimulus_name)


def get_novelty_style_for_stim_name(stimulus_name):
    return get_style_for_stim_name(NOVELTY_STYLE_DICT,
                                   stimulus_name)
