import numpy as np


def build_index(arr):
    max_val = max([max(sublist) for sublist in arr if len(sublist) > 0]) + 1
    d = len(arr)
    arr_set = [set(arr[i]) for i in range(len(arr))]
    idx = {}
    
    for i in range(max_val):
        idx[i] = [c for c, sublist in enumerate(arr_set) if i in sublist]
    return idx    

def calc_noise_by_integer_matching(cat_labels, frac_noise = 0.3, seed = 42):
    # returns a dictionary mapping {image index: index of new label}
    # only for indices where the label is to be changed
    index = build_index(cat_labels)
    rng = np.random.default_rng(seed)
    cand_idxs = np.arange(len(cat_labels)) 
    cand_idxs = [i for i in cand_idxs if len(cat_labels[i]) > 0] # can't match items with no categories
    
    to_change_idxs = rng.choice(cand_idxs, int(frac_noise * len(cat_labels)),
                                replace = False)
    change_dict = {}
    for i in to_change_idxs:
        choose_obj = rng.choice(cat_labels[i])
        subset = index[choose_obj]
        subset = np.setdiff1d(subset, [i])
        if len(subset) > 0: 
            change_dict[i] = rng.choice(subset, 1)[0]
    
    return change_dict

def random_noise_dict(num_items, frac_noise = 0.3, seed = 42):
    rng = np.random.default_rng(seed)
    to_change_idxs = rng.choice( np.arange(num_items), int(frac_noise * num_items),
                                replace = False)
    change_dict = {}
    for i in to_change_idxs:
        change_dict[i] = rng.choice(np.delete(np.arange(num_items), i) # avoid matching to self
                                    , 1)[0]
    return change_dict

def noise_given_dict(meta, d):
    meta_c = meta.copy()
    meta_c['gold_sentence'] = meta_c['sentence']
    source_idx = meta.index[list(d.keys())]
    target_idx = meta.index[list(d.values())]
    # meta_c['is_mislabel'] = False
    meta_c.loc[source_idx, 'sentence'] = meta.loc[target_idx, 'sentence'].values
    # meta_c.loc[source_idx, 'is_mislabel'] = True ## can miss a couple due to same annot across samples
    meta_c['is_mislabel'] = (meta_c['sentence'] != meta_c['gold_sentence'])
    # print(len(source_idx) - meta_c['is_mislabel'].sum())
    return meta_c