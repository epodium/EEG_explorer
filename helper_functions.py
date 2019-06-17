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



def select_bad_episodes(epochs, stimuli, threshold = 5, max_bad_episodes = 10):
    """
    Function to find suspect episodes and channels --> still might need manual inspection!
    
    Args:
    --------
    epochs: epochs object (mne)
    
    stimuli: int/str
        Stimuli to pick episodes for.         
    threshold: float/int
        Relative threshold. Anything channel with variance > threshold*mean OR < threshold*mean
        will be considered suspect. Default = 5.   
    max_bad_episodes: int
        Maximum number of bad episodes. If number is higher for one channel, call it a 'bad' channel
    """
    bad_episodes = set()
    bad_channels = []
    
    from collections import Counter
    
    signals = epochs[str(stimuli)].get_data()
    
    # Find outliers in episode STD and max-min difference:
    signals_std = np.std(signals, axis=2)
    signals_minmax = np.amax(signals, axis=2) - np.amin(signals, axis=2)
    
    outliers = np.where((signals_std > threshold*np.mean(signals_std)) | (signals_minmax > threshold*np.mean(signals_minmax)))
    
    if len(outliers[0]) > 0:
        print("Found", len(set(outliers[0])), "bad episodes in a total of", len(set(outliers[1])), " channels.")
        occurences = [(Counter(outliers[1])[x], x) for x in list(Counter(outliers[1]))]
        for occur, channel in occurences:
            if occur > max_bad_episodes:
                print("Found bad channel (more than", max_bad_episodes, " bad episodes): Channel no: ", channel )
                bad_channels.append(channel)
            else:
                # only add bad episodes for non-bad channels
                bad_episodes = bad_episodes|set(outliers[0][outliers[1] == channel])
        
#        # Remove bad data:
#        signals = np.delete(signals, bad_channels, 1)
#        signals = np.delete(signals, list(bad_episodes), 0)
        
    else:
        print("No outliers found with given threshold.")
    
    return [epochs.ch_names[x] for x in bad_channels], list(bad_episodes)