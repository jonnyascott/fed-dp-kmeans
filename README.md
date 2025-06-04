# Differentially Private Federated *k*-Means Clustering with Server-Side Data

This repository provides an implementation of 
the ICML paper, **"Differentially Private Federated *k*-Means 
Clustering with Server-Side Data"**.

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the experiments via the `run.py` script. Below are the usage instructions for 
different privacy settings and datasets.

---

### Data Point-Level Privacy

#### Mixture of Gaussians Dataset
The following command runs FedDP-KMeans on a mixture of Gaussians with 100 total clients
in a data-point-level privacy setting.

```bash
python run.py --args_config configs/gaussians_data_privacy.yaml
```

#### Folktables Dataset
The following command runs FedDP-KMeans on the folktables dataset
in a data-point-level privacy setting.

```bash
python run.py --args_config configs/folktables.yaml
```

---

### Client-Level Privacy

#### Mixture of Gaussians Dataset
The following command runs FedDP-KMeans on a mixture of Gaussians with 2000 total clients
in a client-level privacy setting.

```bash
python run.py --args_config configs/gaussians_client_privacy.yaml
```

#### StackOverflow Dataset
For StackOverflow first download and process the dataset using the following commands 
(takes some time to run). Set NJPS to parallelize, in total 3*NJPS threads will
be run.
```bash
cd ./data/stackoverflow
. process_stackoverflow.sh $NJPS
```

After preprocessing the dataset, use the following command to run 
FedDP-KMeans on the dataset with topic tags github and pdf,
for a total of 9237 clients.


```bash
python run.py --args_config configs/stackoverflow.yaml
```

---
