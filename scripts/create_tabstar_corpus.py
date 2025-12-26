import os
import sys
import gc
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.datasets import fetch_openml

# Add the project root to sys.path to allow importing nanotabstar
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from nanotabstar.preparation import TabSTARPreprocessor

MAX_SAMPLES_PER_DATASET = 300000
PROCESSING_BATCH_SIZE = 10000

# Classification datasets (binary + multiclass)
CLASSIF_DATASET_CATALOG_BIG = {
    # Small/medium classics for stability and debugging
    "BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN": 15,                 # breast-w
    "BIN_HEALTHCARE_CELLS_WDBC_WISCONSIN_BREAST_CANCER": 1510,    # wdbc
    "BIN_FINANCIAL_CREDIT_GERMAN": 31,                            # credit-g
    "BIN_FINANCIAL_ADULT_INCOME": 1590,                           # adult
    "BIN_SOCIAL_TIC_TAC_TOE": 50,                                 # tic-tac-toe
    "BIN_ANONYM_MONKS_PROBLEM_2": 334,                            # monks-problems-2
    "BIN_ANONYM_AUSTRALIAN_CREDIT_APPROVAL": 40981,               # Australian
    "BIN_SOCIAL_SPAM_EMAILS_SPAMBASE": 44,                        # spambase
    "BIN_NATURE_MUSHROOM_POISONOUS": 24,                          # mushroom
    "BIN_SOCIAL_POLITICS_US_CONGRESS_VOTES": 56,                  # vote
    "BIN_HEALTHCARE_BLOOD_TRANSFUSION": 1464,                     # blood-transfusion-service-center
    "BIN_HEALTHCARE_LIVER_INDIAN_ILPD": 1480,                     # ilpd
    "MUL_HEALTHCARE_HEART_ARRHYTMIA": 5,                          # arrhythmia
    "BIN_NATURE_OZONE_LEVEL": 1487,                               # ozone-level-8hr
    "BIN_NATURE_DISEASED_TREES_WILT": 40983,                      # wilt
    "BIN_ANONYM_TWONORM": 1507,                                   # twonorm

    # Standard tabular classification with larger sample sizes
    "BIN_FINANCIAL_BANK_MARKETING": 1461,                         # bank-marketing
    "BIN_GEOGRAPHY_NOMAO_SEARCH_ENGINE": 1486,                    # nomao
    "BIN_TRANSPORTATION_ROAD_SAFETY_GENDER": 45038,               # road-safety
    "MUL_TRANSPORTATION_TRAFFIC_ACCIDENTS_FARS": 40672,           # fars
    "MUL_TRANSPORTATION_TRAFFIC_VIOLATION": 42345,                # Traffic_violations
    "BIN_TRANSPORTATION_CAR_BAD_BUY_KICK": 41162,                 # kick
    "BIN_SOCIAL_COMPASS_TWO_YEARS_OFFEND": 45039,                 # compas-two-years
    "BIN_SOCIAL_SPEED_DATING": 40536,                             # SpeedDating

    # Textual and high-cardinality datasets (ideal for TabSTAR's verbalization)
    "BIN_SOCIAL_POLICE_INCIDENTS_SAN_FRANCISCO": 42732,           # sf-police-incidents
    "BIN_SOCIAL_JIGSAW_TOXICITY": 46654,                          # (toxicity)
    "BIN_SOCIAL_HATE_SPEECH_DATASET_DYNAMICALLY_GENERATED": 46683,# Dynamically-Generated-Hate-Speech-Dataset
    "MUL_SOCIAL_STACKOVERFLOW_POLARITY": 43160,                   # StackOverflow-polarity
    "BIN_CONSUMER_HOTEL_REVIEW": 43721,                           # Hotel-Reviews
    "BIN_SOCIAL_TWITTER_DISASTER": 43395,                         # Disaster-Tweets
    "MUL_SOCIAL_OKCUPID_DATING_JOB_STEM": 42734,                  # okcupid-stem
    "MUL_SOCIAL_DPBEDIA": 46686,                                  # DBPedia
    "MUL_SOCIAL_HOLISTIC_BIAS": 46684,                            # HolisticBias
    # "MUL_SOCIAL_WIKIPEDIA_TALK_LABELS_ATTACKS": 46708,            # Wikipedia_Talk_Labels

    # Large reference datasets for distributional diversity
    #"BIN_SCIENCE_PARTICLE_HIGGS": 42769,                          # Higgs
    "MUL_NATURE_FOREST_COVERTYPE": 1596,                          # covertype
    "BIN_ANONYM_PORTO_SEGURO": 42742,                             # porto-seguro
    "BIN_ANONYM_APS_FAILURE": 41138,                              # APSFailure

    # Classic multiclass problems
    "MUL_COMPUTERS_IMAGE_LETTER_RECOGNITION": 6,                  # letter
    "MUL_ANONYM_PENDIGITS": 32,                                   # pendigits
    "MUL_COMPUTERS_PAGE_BLOCK_PARSING": 30,                       # page-blocks
    "MUL_PROFESSIONAL_NURSERY_APPLICATIONS_SLOVENIA": 26,         # nursery
    "MUL_FOOD_WINE_QUALITY_CAT": 40498,                           # wine-quality-white
    "MUL_SPORTS_CONNECT4_GAME": 40668,                            # connect-4
}

# Regression datasets
REG_DATASET_CATALOG_BIG = {
    # Classic tabular regression
    "REG_HOUSES_CALIFORNIA_HOUSES": 44977,                        # california_housing
    "REG_SCIENCE_CONCRETE_COMPRESSIVE_STRENGTH": 44959,           # concrete_compressive_strength
    "REG_SCIENCE_ENERGY_EFFICIENCY": 44960,                       # energy_efficiency
    "REG_SCIENCE_AIRFOIL_SELF_NOISE": 44957,                      # airfoil_self_noise
    "REG_HOUSES_BOSTON_HOUSE": 531,                               # boston
    "REG_NATURE_ABALONE_FISH_RINGS": 42726,                       # abalone

    # Robotics and tabular meta-learning
    "REG_COMPUTERS_ROBOT_KIN8NM": 44980,                          # kin8nm
    "REG_COMPUTERS_CPU_ACTIVITY": 44978,                          # cpu_activity
    "REG_ANONYM_BANK_32NH": 558,                                  # bank32nh
    "REG_COMPUTERS_PUMA_ROBOT_ARM": 44981,                        # pumadyn32nh
    "REG_TRANSPORTATION_NAVAL_PROPULSION_PLANT": 44969,           # naval_propulsion_plant
    "REG_COMPUTERS_YOUTUBE_VIDEO_TRANSCODING": 44974,             # video_transcoding

    # Consumer behavior and pricing
    "REG_CONSUMER_BLACK_FRIDAY": 41540,                           # black_friday
    "REG_CONSUMER_DIAMONDS_PRICES": 42225,                        # diamonds
    "REG_CONSUMER_MEDICAL_CHARGES": 44146,                        # medical_charges
    "REG_CONSUMER_AVOCADO_SALES": 41210,                          # avocado-sales
    "REG_CONSUMER_ONLINE_NEWS_POPULARITY": 46662,                 # news_popularity2

    # Mobility and transportation
    "REG_TRANSPORTATION_ZURICH_PUBLIC_TRANSPORT_DELAY": 40753,    # delays_zurich_transport
    "REG_TRANSPORTATION_US_AIRPORT_PASSENGERS": 43479,            # USA-Airport-Dataset
    "REG_TRANSPORTATION_NYC_TAXI_TIP": 42729,                     # NYC taxi
    "REG_TRANSPORTATION_NYC_TAXI_TRIP_DURATION": 43584,           # NYC taxi trip duration
    "REG_ANONYM_BUZZ_IN_SOCIAL_MEDIA_TWITTER": 4549,              # Buzzinsocialmedia_Twitter

    # Environmental and physical sciences
    "REG_SCIENCE_SUPERCONDUCTIVITY": 44964,                       # superconductivity
    "REG_SCIENCE_SULFUR": 44145,                                  # sulfur
    "REG_SCIENCE_WAVE_ENERGY": 44975,                             # wave_energy
    "REG_SCIENCE_GRID_STABILITY": 44973,                          # grid_stability
    "REG_NATURE_FOREST_FIRES": 44962,                             # forest_fires
    "REG_NATURE_QUAKE_RICHTER": 550,                              # quake
    "REG_NATURE_NO2_POLLUTION_NORWAY": 547,                       # no2
    "REG_NATURE_MYANMAR_AIR_QUALITY": 43748,                      # Myanmar-Air-Quality
    "REG_NATURE_POLLEN_LUXEMBOURG": 43648,                        # Pollen-Luxembourg-1992-2018

    # Socio-economic and miscellaneous
    "REG_SOCIAL_US_CRIME": 42730,                                 # us_crime
    "REG_SOCIAL_OCCUPATION_MOBILITY_SOCMOB": 541,                 # socmob
    "REG_SOCIAL_STRIKES_PER_COUNTRY": 549,                        # strikes
    "REG_PROFESSIONAL_CPS88_WAGES": 44984,                        # cps88wages

    # Financial and high-dimensional tabular data
    "REG_ANONYM_SANTANDER_TRANSACTION_VALUE": 42572,              # Santander_transaction_value
    "REG_ANONYM_ALLSTATE_CLAIM_SEVERITY": 42571,                  # Allstate_Claims_Severity
    "REG_ANONYM_MERCEDES_BENZ_GREENER_MANUFACTURING": 42570,      # Mercedes_Benz_Greener_Manufacturing

    # Sports analytics
    "REG_SPORTS_BASEBALL_HITTER_SALARY": 525,                     # baseball-hitter
    "REG_SPORTS_MONEYBALL": 41021,                                # Moneyball
    #"REG_SPORTS_NBA_2K20_PLAYERS_RATING": 43420,                  # NBA-2k20-player-dataset
    "REG_FINANCIAL_STOCK_AEROSPACE": 223,                         # stock
}

CLASSIF_DATASET_CATALOG_64 = {
    # Binary classics (stable, good signal, fast)
    "BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN": 15,                 # breast-w
    "BIN_HEALTHCARE_CELLS_WDBC_WISCONSIN_BREAST_CANCER": 1510,    # wdbc
    "BIN_FINANCIAL_CREDIT_GERMAN": 31,                            # credit-g
    "BIN_FINANCIAL_ADULT_INCOME": 1590,                           # adult
    "BIN_SOCIAL_TIC_TAC_TOE": 50,                                 # tic-tac-toe
    "BIN_ANONYM_MONKS_PROBLEM_2": 334,                            # monks-problems-2
    "BIN_ANONYM_AUSTRALIAN_CREDIT_APPROVAL": 40981,               # Australian
    "BIN_SOCIAL_SPAM_EMAILS_SPAMBASE": 44,                        # spambase
    "BIN_NATURE_MUSHROOM_POISONOUS": 24,                          # mushroom
    "BIN_SOCIAL_POLITICS_US_CONGRESS_VOTES": 56,                  # vote
    "BIN_HEALTHCARE_BLOOD_TRANSFUSION": 1464,                     # blood-transfusion-service-center
    "BIN_HEALTHCARE_LIVER_INDIAN_ILPD": 1480,                     # ilpd
    "BIN_NATURE_OZONE_LEVEL": 1487,                               # ozone-level-8hr
    "BIN_NATURE_DISEASED_TREES_WILT": 40983,                      # wilt
    "BIN_ANONYM_TWONORM": 1507,                                   # twonorm
    "BIN_FINANCIAL_BANK_MARKETING": 1461,                         # bank-marketing
    "BIN_GEOGRAPHY_NOMAO_SEARCH_ENGINE": 1486,                    # nomao
    "BIN_TRANSPORTATION_ROAD_SAFETY_GENDER": 45038,               # road-safety
    "BIN_TRANSPORTATION_CAR_BAD_BUY_KICK": 41162,                 # kick
    "BIN_SOCIAL_COMPASS_TWO_YEARS_OFFEND": 45039,                 # compas-two-years
    "BIN_SOCIAL_SPEED_DATING": 40536,                             # SpeedDating
    "BIN_ANONYM_PORTO_SEGURO": 42742,                             # porto-seguro
    "BIN_ANONYM_APS_FAILURE": 41138,                              # APSFailure
    "BIN_SCIENCE_PARTICLE_HIGGS": 23512,                          # higgs (version OpenML courante)
    "BIN_FINANCIAL_BANKNOTE_AUTHENTICATION": 1462,                # banknote-authentication
    "BIN_SOCIAL_TITANIC_SURVIVAL": 40945,                         # Titanic

    # Textual / high-cardinality (for TabSTAR interest on tabular text)
    "BIN_SOCIAL_POLICE_INCIDENTS_SAN_FRANCISCO": 42732,           # sf-police-incidents
    "BIN_SOCIAL_JIGSAW_TOXICITY": 46654,                          # jigsaw-toxicity
    "BIN_SOCIAL_TWITTER_DISASTER": 43395,                         # Disaster-Tweets
    "BIN_CONSUMER_HOTEL_REVIEW": 43721,                           # Hotel-Reviews
    "BIN_SOCIAL_HATE_SPEECH_DYNAMICALLY_GENERATED": 46683,        # Dynamically-Generated-Hate-Speech-Dataset
    "MUL_SOCIAL_WIKIPEDIA_TALK_LABELS_ATTACKS": 46708,            # Wikipedia_Talk_Labels

    # Multiclass classics (reasonable C, diversity)
    "MUL_HEALTHCARE_HEART_ARRHYTMIA": 5,                          # arrhythmia
    "MUL_NATURE_FOREST_COVERTYPE": 1596,                          # covertype
    "MUL_COMPUTERS_IMAGE_LETTER_RECOGNITION": 6,                  # letter
    "MUL_ANONYM_PENDIGITS": 32,                                   # pendigits
    "MUL_COMPUTERS_PAGE_BLOCK_PARSING": 30,                       # page-blocks
    "MUL_PROFESSIONAL_NURSERY_APPLICATIONS_SLOVENIA": 26,         # nursery
    "MUL_FOOD_WINE_QUALITY_CAT": 40498,                           # wine-quality-white (cat)
    "MUL_SPORTS_CONNECT4_GAME": 40668,                            # connect-4
    "MUL_TRANSPORTATION_TRAFFIC_ACCIDENTS_FARS": 40672,           # fars
    "MUL_TRANSPORTATION_TRAFFIC_VIOLATION": 42345,                # Traffic_violations
    "MUL_SOCIAL_OKCUPID_DATING_JOB_STEM": 42734,                  # okcupid-stem
    "MUL_AUDIO_SPOKEN_ARABIC_DIGITS": 1503,                       # spoken-arabic-digit
    "MUL_SOCIAL_HOLISTIC_BIAS": 46684,                            # HolisticBias
    "MUL_SOCIAL_STACKOVERFLOW_POLARITY": 43160,                   # StackOverflow-polarity
    "MUL_SYNTHETIC_POKER_HAND": 1567,                             # poker-hand
}

REG_DATASET_CATALOG_64 = {
    # Classic tabular regression
    "REG_HOUSES_CALIFORNIA_HOUSES": 44977,                        # california_housing
    "REG_SCIENCE_CONCRETE_COMPRESSIVE_STRENGTH": 44959,           # concrete_compressive_strength
    "REG_SCIENCE_ENERGY_EFFICIENCY": 44960,                       # energy_efficiency
    "REG_SCIENCE_AIRFOIL_SELF_NOISE": 44957,                      # airfoil_self_noise
    "REG_HOUSES_BOSTON_HOUSE": 531,                               # boston
    "REG_NATURE_ABALONE_FISH_RINGS": 42726,                       # abalone

    # Classic “TFM” regression (often used in tabular ML)
    "REG_COMPUTERS_ROBOT_KIN8NM": 44980,                          # kin8nm
    "REG_COMPUTERS_CPU_ACTIVITY": 44978,                          # cpu_activity
    "REG_ANONYM_BANK_32NH": 558,                                  # bank32nh
    "REG_COMPUTERS_PUMA_ROBOT_ARM": 44981,                        # pumadyn32nh

    # Engineering and transportation (distributional diversity)
    "REG_TRANSPORTATION_NAVAL_PROPULSION_PLANT": 44969,           # naval_propulsion_plant
    "REG_TRANSPORTATION_ZURICH_PUBLIC_TRANSPORT_DELAY": 40753,    # delays_zurich_transport

    # Socio-economic / environmental
    "REG_SOCIAL_US_CRIME": 42730,                                 # us_crime
    "REG_SOCIAL_STRIKES_PER_COUNTRY": 549,                        # strikes
    "REG_NATURE_NO2_POLLUTION_NORWAY": 547,                       # no2
    "REG_NATURE_FOREST_FIRES": 44962,                             # forest_fires

    # Sports analytics
    "REG_SPORTS_BASEBALL_HITTER_SALARY": 525,                     # baseball-hitter
    "REG_SPORTS_MONEYBALL": 41021,                                # Moneyball
}

CLASSIF_DATASET_CATALOG_16 = {
    # Binaires très stables et rapides
    "BIN_HEALTHCARE_BREAST_CANCER_WISCONSIN": 15,                 # breast-w
    "BIN_HEALTHCARE_CELLS_WDBC_WISCONSIN_BREAST_CANCER": 1510,    # wdbc
    "BIN_FINANCIAL_CREDIT_GERMAN": 31,                            # credit-g
    "BIN_FINANCIAL_ADULT_INCOME": 1590,                           # adult
    "BIN_SOCIAL_SPAM_EMAILS_SPAMBASE": 44,                        # spambase
    "BIN_NATURE_MUSHROOM_POISONOUS": 24,                          # mushroom
    "BIN_ANONYM_TWONORM": 1507,                                   # twonorm
    # Textuel binaire pour tester la partie TabSTAR "texte dans le tabulaire"
    "BIN_SOCIAL_TWITTER_DISASTER": 43395,                         # Disaster-Tweets

    # Multiclasses (C>2) pour tester target tokens + CrossEntropy multi-classes
    "MUL_COMPUTERS_IMAGE_LETTER_RECOGNITION": 6,                  # letter
    "MUL_ANONYM_PENDIGITS": 32,                                   # pendigits
    "MUL_HEALTHCARE_HEART_ARRHYTMIA": 5,                          # arrhythmia
}

REG_DATASET_CATALOG_16 = {
    # Régressions classiques, très utilisées, bon signal rapidement
    "REG_HOUSES_CALIFORNIA_HOUSES": 44977,                        # california_housing
    "REG_SCIENCE_CONCRETE_COMPRESSIVE_STRENGTH": 44959,           # concrete
    "REG_SCIENCE_ENERGY_EFFICIENCY": 44960,                       # energy_efficiency
    "REG_COMPUTERS_ROBOT_KIN8NM": 44980,                          # kin8nm
    "REG_NATURE_ABALONE_FISH_RINGS": 42726,                       # abalone
}

def prepare_dataset_locally(dataset_id, name, is_cls=None):
    """
    Downloads a dataset and fits the preprocessor.
    Returns the fitted preprocessor, the dataframe, and target info.
    """
    print(f"  Fetching {name} (ID: {dataset_id}) from OpenML...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
    df = data.frame
    
    if data.target_names and len(data.target_names) > 0:
        target_col = data.target_names[0]
    elif data.target is not None:
        if isinstance(data.target, pd.Series):
            target_col = data.target.name
        elif isinstance(data.target, pd.DataFrame):
            target_col = data.target.columns[0]
        else:
            target_col = df.columns[-1]
    else:
        target_col = df.columns[-1]
    
    print(f"  Detected target column: {target_col}")

    if len(df) > MAX_SAMPLES_PER_DATASET:
        print(f"  Dataset too large ({len(df)} rows). Sub-sampling to {MAX_SAMPLES_PER_DATASET}...")
        df = df.sample(n=MAX_SAMPLES_PER_DATASET, random_state=42).reset_index(drop=True)
    
    if is_cls is None:
        is_cls = not pd.api.types.is_numeric_dtype(df[target_col]) or df[target_col].nunique() < 10
    
    preprocessor = TabSTARPreprocessor(is_cls=is_cls, verbose=False)
    preprocessor.fit(df.drop(columns=[target_col]), df[target_col])
    
    return preprocessor, df, target_col

def create_corpus(reg_catalog, classif_catalog, output_path):
    """
    Builds the HDF5 corpus using batch processing to minimize RAM usage.
    """
    print(f"Starting corpus creation at {output_path}")
    dt = h5py.special_dtype(vlen=str)
    
    tasks = [
        (classif_catalog, True, "Classification"),
        (reg_catalog, False, "Regression")
    ]
    
    with h5py.File(output_path, 'w') as f_out:
        for catalog, is_cls, task_name in tasks:
            for name, ds_id in tqdm(catalog.items(), desc=f"Processing {task_name}"):
                try:
                    preprocessor, df, target_col = prepare_dataset_locally(ds_id, name, is_cls=is_cls)
                    
                    n_samples = len(df)
                    d_out = preprocessor.d_output
                    
                    # We need one transform to get the number of features/columns
                    sample_data = preprocessor.transform(df.iloc[:1].drop(columns=[target_col]), df.iloc[:1][target_col])
                    n_feat_cols = sample_data.x_txt.shape[1] - d_out
                    target_texts = sample_data.x_txt[0, :d_out].astype(str)
                    
                    ds_group = f_out.create_group(name)
                    
                    # Initialize HDF5 datasets
                    ds_feat_txt = ds_group.create_dataset('feature_texts', (n_samples, n_feat_cols), dtype=dt, compression='gzip')
                    ds_target_txt = ds_group.create_dataset('target_texts', (d_out,), dtype=dt, compression='gzip')
                    ds_target_txt[:] = target_texts
                    
                    ds_feat_num = ds_group.create_dataset('feature_num_values', (n_samples, n_feat_cols), dtype=np.float32, compression='gzip')
                    ds_labels = ds_group.create_dataset('labels', (n_samples,), compression='gzip')
                    
                    # Process in batches
                    for start_idx in range(0, n_samples, PROCESSING_BATCH_SIZE):
                        end_idx = min(start_idx + PROCESSING_BATCH_SIZE, n_samples)
                        batch_df = df.iloc[start_idx:end_idx]
                        
                        batch_data = preprocessor.transform(
                            batch_df.drop(columns=[target_col]), 
                            batch_df[target_col]
                        )
                        
                        # Write slices
                        ds_feat_txt[start_idx:end_idx] = batch_data.x_txt[:, d_out:].astype(str)
                        ds_feat_num[start_idx:end_idx] = batch_data.x_num[:, d_out:].astype(np.float32)
                        ds_labels[start_idx:end_idx] = batch_data.y.values
                        
                        del batch_data
                        gc.collect()
                    
                    # Metadata
                    ds_group.attrs['d_output'] = d_out
                    ds_group.attrs['n_features'] = n_feat_cols
                    ds_group.attrs['task_type'] = 'classification' if is_cls else 'regression'
                    
                    print(f"  Saved {name} ({n_samples} samples)")
                    
                    del df, preprocessor
                    gc.collect()
                    
                except Exception as e:
                    print(f"Failed {name}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    output_file = r"c:\Users\issa\dev\TFM\nanoTabStar\data\pretrain_corpus_tabstar_16.h5"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    create_corpus(REG_DATASET_CATALOG_16, CLASSIF_DATASET_CATALOG_16, output_file)
