def permutationTest(sample1, sample2, permutations, sidedness, exact=False, plotresult=False, showprogress=0):
    """
    permutation_test from (cite as):
    Laurens R Krol (2025). Permutation Test (https://github.com/lrkrol/permutationTest), GitHub. Retrieved March 20, 2025.
    
    Created on Thu Mar 20 13:11:51 2025
    @author: rd883
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import combinations
    from scipy.stats import norm
    
    """
    Perform a permutation test to compare the means of two samples.
    
    Parameters:
        sample1 (array-like): Experimental sample.
        sample2 (array-like): Control sample.
        permutations (int): Number of permutations.
        sidedness (str): 'both', 'smaller', or 'larger' for two-sided or one-sided test.
        exact (bool): If True, performs an exact test using all possible combinations.
        plotresult (bool): If True, plots the distribution of differences.
        showprogress (int): If > 0, prints progress every `showprogress` iterations.
    
    Returns:
        p (float): p-value of the test.
        observed_difference (float): Mean(sample1) - Mean(sample2).
        effect_size (float): Hedges' g effect size.
    """
    p, observed_difference, effect_size = 10, 10, 10
    
    sample1, sample2 = np.asarray(sample1), np.asarray(sample2)
    observed_difference = np.nanmean(sample1) - np.nanmean(sample2)
    pooled_std = np.sqrt(((len(sample1) - 1) * np.nanvar(sample1, ddof=1) + 
                          (len(sample2) - 1) * np.nanvar(sample2, ddof=1)) /
                         (len(sample1) + len(sample2) - 2))
    effect_size = observed_difference / pooled_std
    
    all_observations = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    
    if exact:
        all_combinations = list(combinations(range(len(all_observations)), n1))
        permutations = len(all_combinations)
    
    random_differences = np.zeros(permutations)
    
    for i in range(permutations):
        if exact:
            indices = np.array(all_combinations[i])
        else:
            np.random.shuffle(all_observations)
            indices = np.arange(n1)
        
        perm_sample1 = all_observations[indices]
        # perm_sample2 = all_observations[~np.isin(np.arange(len(all_observations)), indices)]
        perm_sample2 = np.delete(all_observations, indices)
        
        random_differences[i] = np.nanmean(perm_sample1) - np.nanmean(perm_sample2)
        
        if showprogress > 0 and i % showprogress == 0:
            print(f"Permutation {i+1} of {permutations}")
    
    if sidedness == 'both':
        p = (np.sum(np.abs(random_differences) > abs(observed_difference)) + 1) / (permutations + 1)
    elif sidedness == 'smaller':
        p = (np.sum(random_differences < observed_difference) + 1) / (permutations + 1)
    elif sidedness == 'larger':
        p = (np.sum(random_differences > observed_difference) + 1) / (permutations + 1)
    else:
        raise ValueError("sidedness must be 'both', 'smaller', or 'larger'")
    
    if plotresult:
        plt.hist(random_differences, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(observed_difference, color='r', linestyle='dashed', linewidth=2, 
                    label=f'Observed Difference\nEffect size: {effect_size:.2f}\np = {p:.6f}')
        plt.xlabel("Random Differences")
        plt.ylabel("Count")
        plt.legend()
        plt.show()
    
    return p, observed_difference, effect_size

