# UCI Human Activity Recognition dataset

## Download the data

* Run: `download_dataset.py`
* Note: the dataset can also be manually downloaded and extracted from : 
```
https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
```
* You should have this tree structure in the current folder: 
```
.
├── UCI HAR Dataset
│   ├── activity_labels.txt
│   ├── features_info.txt
│   ├── features.txt
│   ├── README.txt
│   ├── test
│   │   ├── Inertial Signals
│   │   │   ├── body_acc_x_test.txt
│   │   │   ├── body_acc_y_test.txt
│   │   │   ├── body_acc_z_test.txt
│   │   │   ├── body_gyro_x_test.txt
│   │   │   ├── body_gyro_y_test.txt
│   │   │   ├── body_gyro_z_test.txt
│   │   │   ├── total_acc_x_test.txt
│   │   │   ├── total_acc_y_test.txt
│   │   │   └── total_acc_z_test.txt
│   │   ├── subject_test.txt
│   │   ├── X_test.txt
│   │   └── y_test.txt
│   └── train
│       ├── Inertial Signals
│       │   ├── body_acc_x_train.txt
│       │   ├── body_acc_y_train.txt
│       │   ├── body_acc_z_train.txt
│       │   ├── body_gyro_x_train.txt
│       │   ├── body_gyro_y_train.txt
│       │   ├── body_gyro_z_train.txt
│       │   ├── total_acc_x_train.txt
│       │   ├── total_acc_y_train.txt
│       │   └── total_acc_z_train.txt
│       ├── subject_train.txt
│       ├── X_train.txt
│       └── y_train.txt
└── UCI HAR Dataset.zip
```

## Convert the data
* Run `convert_to_csv.py`
