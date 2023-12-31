vcaml
==============================

An end-to-end pipeline designed to estimate QoE for WebRTC-based video conferencing applications (VCAs) without using application layer headers.

# 1. Download Datasets

- [In-Lab](https://drive.google.com/file/d/1XmFqwCKzdJtYg7TQHS8gCvA5CeI_499P/view?usp=sharing)
- [Real World](https://drive.google.com/file/d/1kASPQlokHiUlhWry6I8qM-Hc0AvHz5eq/view?usp=sharing)

# 2. Install Dependencies

1. If you intend to train and evaluate our models over your own PCAPs, you will need to create CSVs using the script `src/util/pcap2csv.py`. It requires a working `tshark` installation.
2. For dependencies related to data collection, refer to [the data collection README](src/data/real-world/README.md).
3. Inside a Python3 virtual environment, execute `setup.py` to install dependencies.

# 3. Collect Additional Data

Refer to [In-Lab Data Collection](src/data/in-lab) and [Real-World Data Collection](src/data/real-world) for more details.

# 4. Prepare Inference pipeline

For reproducing the results in our paper, download and place the datasets under `data/`.

If you intend to use your own traces, place your files under `data/` with the same directory structure as our datasets. Do not forget to modify [config.py](src/models/config.py) as per your requirments.

# 5. Train and test models

If you want to conduct an independent analysis on our trained models, refer to the below links. These contain our model intermediates. Place them in `data/`. Then run the [notebooks](notebooks/) straight away.

- [In-lab Model Intermediates](https://drive.google.com/file/d/1w5zR-jAxcUNBAk23Q_YcuC5loOT2Ijr9/view?usp=sharing)
- [Real-world Model Intermediates](https://drive.google.com/file/d/1vnLC1Sw-v_ARnf9rePOqUcOR15DjNTgA/view?usp=sharing)

An example directory structure for [In-lab Model Intermediates](https://drive.google.com/file/d/1w5zR-jAxcUNBAk23Q_YcuC5loOT2Ijr9/view?usp=sharing) is shown as under:

```
vcaml/data/in_lab_data_intermediates  
├── cv_splits.pkl  --> A dictionary that stores the cross validation splits
├── framesReceivedPerSecond_ip-udp-heuristic_LSTATS-TSTATS_in_lab_data_cv_1
│   ├── model.pkl  --> A dictionary for each VCA for a single cross validation
│   ├── predictions_meet.pkl  --> Predictions for a single VCA
│   ├── predictions_teams.pkl
│   └── predictions_webex.pkl
├── framesReceivedPerSecond_ip-udp-heuristic_LSTATS-TSTATS_in_lab_data_cv_2
│   ├── model.pkl
│   ├── predictions_meet.pkl
│   ├── predictions_teams.pkl
│   └── predictions_webex.pkl
├── framesReceivedPerSecond_ip-udp-heuristic_LSTATS-TSTATS_in_lab_data_cv_3
│   ├── model.pkl
│   ├── predictions_meet.pkl
│   ├── predictions_teams.pkl
│   └── predictions_webex.pkl
```

To train and evaluate models from scratch, refer to [run_model.py](src/models/run_model.py). Modify the below part of the code according to your requirements.

```python
if __name__ == '__main__':

    metrics = ['framesReceivedPerSecond', 'bitrate',
               'frame_jitter', 'frameHeight']  # what to predict
    estimation_methods = ['ip-udp-ml', 'rtp-ml', 'ip-udp-heuristic', 'rtp-heuristic']  # how to predict
    # groups of features as per `features.feature_extraction.py`
    feature_subsets = [['LSTATS', 'TSTATS']]
    data_dir = ['/home/taveesh/Documents/vcaml/data/in_lab_data']

    bname = os.path.basename(data_dir[0])

    # Get a list of pairs (trace_csv_file, ground_truth)

    fp = FileProcessor(data_directory=data_dir[0])
    file_dict = fp.get_linked_files()

    # Create 5-fold cross validation splits and validate files. Refer `src/util/validator.py` for more details

    kcv = KfoldCVOverFiles(5, file_dict, project_config, bname)
    file_splits = kcv.split()

    vca_preds = defaultdict(list)

    param_list = [metrics, estimation_methods, feature_subsets, data_dir]

    # Run models over 5 cross validations

    for metric, estimation_method, feature_subset, data_dir in product(*param_list):
        if metric == 'frameHeight' and 'heuristic' in estimation_method:
            continue
        models = []
        cv_idx = 1
        for fsp in file_splits:
            model_runner = ModelRunner(
                metric, estimation_method, feature_subset, data_dir, cv_idx)
            vca_model = model_runner.train_model(fsp)
            predictions = model_runner.get_test_set_predictions(fsp, vca_model)
            models.append(vca_model)

            for vca in predictions:
                vca_preds[vca].append(pd.concat(predictions[vca], axis=0))

            cv_idx += 1
```

While the models run, a file `log.txt` is created to track the progress. An example is shown below:

```
2023-09-16 14:09:14.841418	VCA: teams || Experiment : framesReceivedPerSecond_ip-udp-ml_LSTATS-TSTATS_in_lab_data_cv_1 || MAE_avg = 1.93 || Accuracy_avg = 77.35
2023-09-16 14:09:14.841507	VCA: meet || Experiment : framesReceivedPerSecond_ip-udp-ml_LSTATS-TSTATS_in_lab_data_cv_1 || MAE_avg = 1.31 || Accuracy_avg = 87.64
2023-09-16 14:09:14.841556	VCA: webex || Experiment : framesReceivedPerSecond_ip-udp-ml_LSTATS-TSTATS_in_lab_data_cv_1 || MAE_avg = 0.85 || Accuracy_avg = 90.9
2023-09-16 14:13:23.324799	VCA: teams || Experiment : framesReceivedPerSecond_ip-udp-ml_LSTATS-TSTATS_in_lab_data_cv_2 || MAE_avg = 1.83 || Accuracy_avg = 82.06
2023-09-16 14:13:23.324886	VCA: meet || Experiment : framesReceivedPerSecond_ip-udp-ml_LSTATS-TSTATS_in_lab_data_cv_2 || MAE_avg = 1.36 || Accuracy_avg = 86.45
```

# 6. Cite our work

```
@inproceedings{10.1145/3618257.3624828,
    author = {Sharma, Taveesh and Mangla, Tarun and Gupta, Arpit and Jiang, Junchen and Feamster, Nick},
    title = {Estimating WebRTC Video QoE Metrics Without Using Application Headers},
    year = {2023},
    isbn = {9798400703829},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3618257.3624828},
    doi = {10.1145/3618257.3624828},
    booktitle = {Proceedings of the 2023 ACM on Internet Measurement Conference},
    pages = {485–500},
    numpages = {16},
    keywords = {machine learning, access networks, quality of experience, video conferencing},
    location = {Montreal QC, Canada},
    series = {IMC '23}
}
```