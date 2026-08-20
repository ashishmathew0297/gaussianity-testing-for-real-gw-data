glitch_labels = [
    '1080Lines',
    '1400Ripples',
    'Air_Compressor',
    'Blip',
    'Blip_Low_Frequency',
    'Chirp',
    'Extremely_Loud',
    'Fast_Scattering',
    'Helix',
    'Koi_Fish',
    'Light_Modulation',
    'Low_Frequency_Burst',
    'Low_Frequency_Lines',
    'Paired_Doves',
    'Power_Line',
    'Repeating_Blips',
    'Scattered_Light',
    'Scratchy',
    'Tomte',
    'Violin_Mode',
    'Wandering_Line',
    'Whistle',
    'Clean_Signal']

useful_cols =[
    'label',
    'shapiro_statistic',
    'shapiro_pvalue',
    'shapiro_prediction',
    'ks_statistic',
    'ks_pvalue',
    'ks_prediction',
    'ad_statistic',
    'ad_critical_values',
    'ad_significance_level',
    'ad_prediction',
    'cvm_statistic',
    'cvm_pvalue',
    'cvm_prediction',
    'jb_statistic',
    'jb_pvalue',
    'jb_prediction',
    'normal_test_statistic',
    'normal_test_pvalue',
    'normal_test_prediction',
    'lilliefors_statistic',
    'lilliefors_pvalue',
    'lilliefors_prediction',
    'kurtosis',
    'skew',
    'glitch_present']

ifo = "L1"
run = "O3a"

glitch_times_file = f'./glitches/unique_DQ_glitches_{run}_{ifo}.csv'
clean_segments_file = f"./clean_segments/unique_pre_clean_segments_{run}_{ifo}.csv"