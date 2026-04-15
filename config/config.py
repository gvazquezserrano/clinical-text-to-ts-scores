#--------------------------------------------------------------------------------#
# Reproducibility                                                                #
#--------------------------------------------------------------------------------#

RANDOM_STATE = 42


#--------------------------------------------------------------------------------#
# Text-to-score transformation                                                   #
#--------------------------------------------------------------------------------#

TEXT_ENCODER = "ISeeU2"  # ["ISeeU2", "CORe"]

# Path to the text classification model used to transform the free-text notes into mortality risk scores
PATH_TEXT_ENCODER = f"./assets/{TEXT_ENCODER}/"

# This path is used in 'text_to_score.ipynb' to transform the notes
# It needs to be used in i) MIMIC-III for MIMIC-III dataset, and in ii) MIMIC-III subset + iii) MIMIC-IV for the MIMIC-III+IV dataset
MIMIC_VERSION = "MIMIC_III"  # ["MIMIC_III", "MIMIC_III_subset", "MIMIC_IV"]
PATH_MIMIC = f"data/{MIMIC_VERSION}/" 

OUTPUT_PATH_NOTES_SCORES = f"data/{MIMIC_VERSION.lower()}/notes_mortality_risk_{TEXT_ENCODER}.pkl"

#--------------------------------------------------------------------------------#
# Dataset                                                                        #
#--------------------------------------------------------------------------------#

# Minimum length-of-stay per patient stay
MIN_LOS_ICU = 24
# Maximum length-of-stay per patient stay
MAX_LOS_ICU = MIN_LOS_ICU + 48

# Amount of data selected per patient stay (in hours from admission)
WINDOW_LENGTH = 24  # [8, 16, 24]

DATASET = "MIMIC_III"  # ["MIMIC_III", "MIMIC_III_IV"]

# Dataset-specific configurations
_DATASET_CONFIGS = {
    "MIMIC_III": {
        "DBNAME": "mimic_iii",
        "DATA_PATH": f"data/MIMIC_III/min{MIN_LOS_ICU:d}h/",
        "DATA_PATH_PREPROCESSING": f"data/mimic_iii/preprocessing/",
        "DATA_PATH_WINDOWING": f"data/mimic_iii/windowing/",
        "DATA_PATH_NOTES_SCORES": f"data/mimic_iii/notes_mortality_risk_{TEXT_ENCODER}.pkl",
        "I_TRAIN_PATH": "./data/i_train_iii.csv",  # icustays indexes for MIMIC-III train (+val) split
    },
    "MIMIC_III_IV": {
        "DBNAME": "mimic_iii_iv",
        "DATA_PATH": f"data/MIMIC_III_IV/min{MIN_LOS_ICU:d}h/",
        "DATA_PATH_PREPROCESSING": f"data/mimic_iii_iv/preprocessing/",
        "DATA_PATH_WINDOWING": f"data/mimic_iii_iv/windowing/",
        "DATA_PATH_NOTES_SCORES": f"data/mimic_iii_iv/notes_mortality_risk_{TEXT_ENCODER}.pkl",  # This pkl file needs to be created with the combination of mimic_iii_subset and mimic_iv pkl files
        "I_TRAIN_PATH": "./data/i_train_iii_iv.csv",  # icustays indexes for MIMIC-III+IV train (+val) split
    }
}

# Set variables based on selected dataset
DBNAME = _DATASET_CONFIGS[DATASET]["DBNAME"]

DATA_PATH = _DATASET_CONFIGS[DATASET]["DATA_PATH"]
DATA_PATH_PREPROCESSING = _DATASET_CONFIGS[DATASET]["DATA_PATH_PREPROCESSING"]
DATA_PATH_WINDOWING = _DATASET_CONFIGS[DATASET]["DATA_PATH_WINDOWING"]
DATA_PATH_NOTES_SCORES = _DATASET_CONFIGS[DATASET]["DATA_PATH_NOTES_SCORES"]

I_TRAIN_PATH = _DATASET_CONFIGS[DATASET]["I_TRAIN_PATH"]
I_TEST_PATH = "./data/i_test_iii.csv"  # icustays indexes for test split (same for both datasets)



#--------------------------------------------------------------------------------#
# Ablation                                                                       #
#--------------------------------------------------------------------------------#

ABLATION_RANDOM_STATE = 21  # [21, 22, 23, 24, 25]

ABLATION_TYPE = ""  # ["", f"_ablation_temporal{ABLATION_RANDOM_STATE}", "_ablation_presence", f"_ablation_random{ABLATION_RANDOM_STATE}"]

# ABLATION TYPES:
# "" - No ablation; standard preprocessing
# f"_ablation_temporal{ABLATION_RANDOM_STATE}" - Shuffle the notes timestamps at patient level
# "_ablation_presence" - Replace the notes scores with a masked value indicating presence (1) or absence (0) of notes in each timestep
# f"_ablation_random{ABLATION_RANDOM_STATE}" - Replace the text-derived scores with random noise (values ranging from 0.0 to 1.0)


#--------------------------------------------------------------------------------#
# Learning framework (centralised, federated, local)                             #
#--------------------------------------------------------------------------------#

USE_FL = False  # [True, False] True for Federated Learning

# Number of clients/hospitals in the Federated / Local Learning framework
# Use 1 for Centralised Learning
CLIENT_COUNT = 1  # [1, 2, 4, 8]


#--------------------------------------------------------------------------------#
# Input layers and layer-specific features                                       #
#--------------------------------------------------------------------------------#

ADD_VITALS = True  # [True, False]
ADD_LABS = True  # [True, False]
ADD_NOTES = True  # [True, False]

VITAL_NAMES = ["heartrate", "sysbp", "diasbp", "meanbp", "resprate", "tempc", "spo2"]
LAB_NAMES = ["albumin", "bun", "bilirubin", "lactate", "bicarbonate", "bands", "chloride", "creatinine", "glucose", "hemoglobin", "hematocrit", "platelet", "potassium", "ptt", "sodium", "wbc"]
NOTES_NAMES = ["mortality_risk_prediction"]
FOLDER_SUFFIX = ""


#--------------------------------------------------------------------------------#
# Backbone model                                                                 #
#--------------------------------------------------------------------------------#

MODEL_ARCHITECTURE = "GRU"  # ["GRU", "LSTM", "transformer"]


#--------------------------------------------------------------------------------#
# Training hyperparameters                                                       #
#--------------------------------------------------------------------------------#

RNN_DIM = 16 # Number of units in RNN (GRU, LSTM); it controls the dimensionality of the output for all three models

if MODEL_ARCHITECTURE == "transformer":

    EMBED_DIM_INTERNAL = 64   
    FEED_FORWARD_DIM = 128   
    NUM_HEADS = 4 
    DROPOUT_RATE = 0.2
    NUM_TRANSFORMER_BLOCKS = 3 

    NUM_EPOCHS = 100
    PEAK_LR = 5e-4
    FINAL_LR = 1e-6
    WEIGHT_DECAY = 1e-5
    CLIP_NORM = 1.0
    STEPS_PER_EPOCH = 100