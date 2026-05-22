# Enhancing Early Mortality Prediction with Clinical Notes as Time-Series Scores

This repository contains the implementation code for the paper: 

"Enhancing Early Mortality Prediction with Clinical Notes as Time-Series Scores"
<br> Gabriel Vázquez-Serrano, Maite Oronoz, Alicia Pérez

## Overview

This project introduces a model-agnostic approach to integrate unstructured clinical notes with structured physiological time-series data for enhanced early mortality prediction in ICU settings. 
Instead of using complex multimodal architectures that process physiological signals and clinical text through separate model branches before fusion, we shift fusion from the architecture level to the data representation level.
Each clinical note is encoded into a single numerical mortality risk score via text classification, producing a time-series feature that is structurally identical to physiological measurements like heart rate or glucose. This text-derived signal can then be fed into any standard time-series model to jointly learn from both data modalities without architectural modification.

#### Key aspects

- **Preprocessing-level fusion** — Alignment of text with physiological signals before modelling.
- **Encoder agnosticism** — Evaluation with two architecturally distinct text classifiers.
- **Model agnosticism** — Validation across GRU, LSTM, and Transformer backbones.
- **Early prediction focus** — Effective prediction with only 8–24&nbsp;h of post-admission data.
- **Framework versatility** — Evaluation under centralised, federated, and local learning settings.

## Repository structure

- `/config`: Configuration file containing paths and experiment variables.

- `/database`: SQL scripts for MIMIC-III/MIMIC-IV database setup.

- `/assets`: Auxiliary scripts, including:

  - `helpers.py`: Core helper functions and classes.

  - `metrics.py`: Evaluation metrics functions and classes.

  - `data_splitting.py`: Train/test split script.

  - `/iseeu2`: ISeeU2 software by Caicedo-Torres & Gutierrez, 2020 [[1]](#ref-1).

- `text_to_score.ipynb`: Transforms raw free-text notes into mortality risk scores.

- `preprocessing.ipynb`: Preprocesses raw patient data into time-series formats.

- `windowing.ipynb`: Selects patient data window lengths.

- `training.ipynb`: Trains and evaluates the backbone models.

## Data

This repository does **not** contain any patient data. Experiments were conducted using:

- MIMIC-III Clinical Database (v1.4) [[2]](#ref-2)
- MIMIC-III Clinical Database CareVue subset (v1.4) [[3]](#ref-3)
- MIMIC-IV (v2.2) [[4]](#ref-4)
- MIMIC-IV-Note (v2.2) [[5]](#ref-5)

All datasets are available through [PhysioNet](https://physionet.org) after completing required credentialling.

## Usage

### Prerequisites

- Install dependencies using:

    ```bash
    pip install -r requirements.txt
    ```
- Obtain access to the MIMIC-III (v1.4) and/or MIMIC-IV (v2.2) datasets through [PhysioNet](https://physionet.org).
- Set up a PostgreSQL database (recommended).
- Modify `/config/config.py` to adjust data paths, experimental parameters, and model hyperparameters.

### Workflow

1. **Database setup**: Run the SQL scripts in `/database` to create the necessary database tables from MIMIC-III/IV datasets.
2. **Text-to-score transformation**: Run `text_to_score.ipynb` to convert free-text clinical notes into mortality risk scores.
3. **Data preprocessing**: Run `preprocessing.ipynb` to preprocess raw patient data into time-series format suitable for machine learning.
4. **Window selection**: Run `windowing.ipynb` to select the data window length for early prediction (8-24 hours post-admission).
5. **Data splitting**: Run `/assets/data_splitting.py` to obtain the train/test splits.
6. **Model training**: Run `training.ipynb` to train backbone models and evaluate performance.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{VazquezSerrano_Enhancing_2026,
  title={{Enhancing Early Mortality Prediction with Clinical Notes as Time-Series Scores}},
  author={V{\'a}zquez-Serrano, Gabriel and Oronoz, Maite and P{\'e}rez, Alicia},
  journal={IEEE Transactions on Human-Machine Systems},
  year={2026},
  volume={XX},
  number={XX},
  pages={XX-XX},
  doi={DOI-PLACEHOLDER}
}
```

## Licence

This project is licensed under the GNU General Public License v3.0 (GPL-3.0).

## Acknowledgements

This work builds upon and adapts code from several open-source projects:

- ISeeU2 software by Caicedo-Torres & Gutierrez (2020) [[1]](#ref-1)
- Database scripts based on Mondrejevski et al. (2022) [[6]](#ref-6)
- Preprocessing, windowing, and training pipelines adapted from Randl et al. (2023) [[7]](#ref-7)

We thank these authors for making their code publicly available.

## References

1. <a id="ref-1"></a>W. Caicedo-Torres and J. Gutierrez, "ISeeU2: Visually interpretable mortality prediction inside the ICU using deep learning and free-text medical notes", *Expert Systems with Applications*, vol. 202, p. 117190, 2022.

2. <a id="ref-2"></a>A. Johnson, T. Pollard, and R. Mark, "MIMIC-III Clinical Database (version 1.4)", 2016. PhysioNet.

3. <a id="ref-3"></a>A. Johnson, T. Pollard, and R. Mark, "MIMIC-III Clinical Database CareVue subset (version 1.4)", 2022. PhysioNet.

4. <a id="ref-4"></a>A. Johnson, L. Bulgarelli, T. Pollard, S. Horng, L. A. Celi, and R. Mark, "MIMIC-IV (version 2.2)", 2023. PhysioNet.

5. <a id="ref-5"></a>A. Johnson, T. Pollard, S. Horng, L. A. Celi, and R. Mark, "MIMIC-IV-Note: Deidentified Free-Text Clinical Notes (version 2.2)", 2023. PhysioNet.

6. <a id="ref-6"></a>L. Mondrejevski, I. Miliou, A. Montanino, D. Pitts, J. Hollmen, and P. Papapetrou, "FLICU: A Federated Learning Workflow for Intensive Care Unit Mortality Prediction", in *2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS)*. Los Alamitos, CA, USA: IEEE Computer Society, 2022, pp. 32–37.

7. <a id="ref-7"></a>K. Randl, N. L. Armengol, L. Mondrejevski, and I. Miliou, "Early prediction of the risk of ICU mortality with Deep Federated Learning", in *2023 IEEE 36th International Symposium on Computer-Based Medical Systems (CBMS)*. L’Aquila, Italy: IEEE Computer Society, 2023, pp. 706–711.
