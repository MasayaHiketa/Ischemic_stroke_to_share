##Preprocessing data and logistic regression##

By typing these command, you will get result.
    pip install -r requirement.txt
    python LR_2_4.py
    python LR_6_8.py
    python LR_10_12.py

Which is doing

    -match age
    -select features
    -20times 5-fold cross validation logistic regression  

Terminal Results will be:

    -Static Analysis
    -Age Statistics by Survival Status:
    -Gender Statistics by Survival Status
    -selected_features
    -Coefficients and OddRatio
    -Logistic Regression AUC curve graph 
    -Eeach metrics values by all threshold( in all_threshold_metrics_0.001_time_time.csv)

(Optional)Extract 5-min window hrv data from mimic-3 data(for 2-4hours,6-8hours,10-12hours),it will take several hours for each.

The extracted data csv  which I already did is 

    extracted_data_2_4.csv
    extracted_data_6_8.csv
    extracted_data_10_12.csv

if you want to extract by yourself,

    pip install -r requirement.txt
    (for installing neccesary things)

    python extract_data_2_4.py
    (extracted data will be extracted_data_2_4.csv)

    python extract_data_6_8.py
    (extracted data will be extracted_data_6_8.csv)

    python extract_data_10_12.py
    (extracted data will be extracted_data_10_12.csv)



