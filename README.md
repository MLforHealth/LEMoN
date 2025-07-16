# LEMoN: Label Error Detection using Multimodal Neighbors

## Paper
If you use this code in your research, please cite the following paper:

```
@inproceedings{
    zhang2025lemon,
    title={{LEM}oN: Label Error Detection using Multimodal Neighbors},
    author={Haoran Zhang and Aparna Balagopalan and Nassim Oufattole and Hyewon Jeong and Yan Wu and Jiacheng Zhu and Marzyeh Ghassemi},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025}
}
```

## To replicate the experiments in the paper:

### Step 0: Environment and Prerequisites

Run the following commands to clone this repo and create the Conda environment:

```
git clone git@github.com:MLforHealth/LEMoN.git
cd LEMoN
conda env create -f environment.yml
conda activate lemon
```

### Step 1: Preprocessing Data
CIFAR-10 and CIFAR-100 are downloaded automatically by the codebase. To preprocess the remaining datasets, follow the instructions in [DataSources.md](DataSources.md).


### Step 2: Running Experiments

To run a single evaluation, call `run_lemon.py` with the appropriate arguments, for example:

```
python -m run_lemon \ 
    --output_dir /output/dir \
    --dataset mscoco \
    --noise_type cat \
    --noise_level 0.4 
```


To reproduce the experiments in the paper which involve training a grid of models using different hyperparameters, use `sweep.py` as follows:

```
python sweep.py launch \
    --experiment {experiment_name} \
    --output_dir {output_root} \
    --command_launcher {launcher} 
```

where:
- `experiment_name` corresponds to experiments defined as classes in `experiments.py`
- `output_root` is a directory where experimental results will be stored.
- `launcher` is a string corresponding to a launcher defined in `launchers.py` (i.e. `slurm` or `local`).

### Step 3: Aggregating Results

After the `lemon_all` experiment has finished running, to create Tables 2 and 3, run `notebooks/agg_results.ipynb` and `notebooks/hparam_drop.ipynb`
