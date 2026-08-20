import pandas as pd
import uuid


# glitches = pd.read_csv("./glitches/DQ_glitches_O3a_L1.csv", usecols=['GPStime', 'ifo', 'label', 'id'])
glitches = pd.read_csv("./glitches/DQ_glitches_O3a_L1.csv", index_col=0)
glitches = glitches[~glitches.duplicated(subset=['GPStime'], keep='first')]
glitches = glitches.drop(columns=['imgUrl'])
glitches = glitches[glitches['label'] != 'No_Glitch']

clean_samples = pd.read_csv("./clean_segments/pre_clean_segments_O3a_L1.csv", usecols=['start_time', 'end_time', 'p_values'])
clean_samples = clean_samples[clean_samples['p_values'] >= 0.05]
clean_samples = clean_samples.drop(columns=['p_values'])

# Add unique IDs to glitches and clean samples
# glitches['id'] = [str(uuid.uuid4()) for _ in range(len(glitches))]
clean_samples['id'] = [str(uuid.uuid4().hex)[:10].upper() for _ in range(len(clean_samples))]

glitches.to_csv("glitches/unique_DQ_glitches_O3a_L1.csv", index=False)
clean_samples.to_csv(f"clean_segments/unique_pre_clean_segments_O3a_L1.csv", index=False)
print(f"Updated files with unique IDs.")