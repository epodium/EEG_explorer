"""
Helper functions to work with ePODIUM EEG data

"""
import numpy as np


def select_bad_channels(data_raw, time = 100, threshold = 5, include_for_mean = 0.8):
    """
    Function to find suspect channels --> still might need manual inspection!
    
    Args:
    --------
    data_raw: mne object
        
    time: int
        Time window to look for ouliers (time in seconds). Default = 100.
    threshold: float/int
        Relative threshold. Anything channel with variance > threshold*mean OR < threshold*mean
        will be considered suspect. Default = 5.
    include_for_mean: float
        Fraction of variances to calculate mean. This is to ignore the highest and lowest
        ones, which coul dbe far outliers.
    
    """
    sfreq = data_raw.info['sfreq']
    no_channels = len(data_raw.ch_names) -1  # Subtract stimuli channel
    data, times = data_raw[:no_channels, int(sfreq * 10):int(sfreq * (time+10))]
    variances = []
    for i in range(data.shape[0]):
        variances.append(data[i,:].var())
    var_arr = np.array(variances)
    exclude = int((1-include_for_mean)*no_channels/2)
    mean_low = np.mean(np.sort(var_arr)[exclude:(no_channels-exclude)])
    
    suspects = np.where((var_arr > threshold* mean_low) & (var_arr < threshold/mean_low))[0]
    suspects_names = [data_raw.ch_names[x] for x in list(suspects)]
    selected_suspects = [data_raw.ch_names.index(x) for x in suspects_names if not x in ['HEOG', 'VEOG']]
    selected_suspects_names = [x for x in suspects_names if not x in ['HEOG', 'VEOG']]
    print("Suspicious channel(s): ", selected_suspects_names)
    
    return selected_suspects, selected_suspects_names



