## Create database tables from MIMIC

1. Build postgres-functions (`postgres-functions.sql`).
2. Build flicu_icustay_detail (`flicu_icustay_detail.sql`).
3. Build pivoted_vital (`pivoted_vital.sql`).
4. Build flicu_pivoted_lab (`flicu_pivoted_lab.sql`). Alternatively, build `pivoted_lab.sql` if lab values before ICU admission are needed.

## Acknowledgements

The scripts on this directory are re-used from Mondrejevski et al. (2022) [[1]](#ref-1).

## References

1. <a id="ref-1"></a>L. Mondrejevski, I. Miliou, A. Montanino, D. Pitts, J. Hollmen, and P. Papapetrou, "FLICU: A Federated Learning Workflow for Intensive Care Unit Mortality Prediction", in *2022 IEEE 35th International Symposium on Computer-Based Medical Systems (CBMS)*. Los Alamitos, CA, USA: IEEE Computer Society, 2022, pp. 32–37.
