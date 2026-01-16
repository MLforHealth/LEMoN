# LEMoN: Label Error Detection using Multimodal Neighbors

## Paper
If you use this code in your research, please cite our [ICML 2025 paper](https://arxiv.org/pdf/2407.18941):

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


## Post-publication Baseline Sensitivity Analysis (January 2026): Logit Scaling
The CLIP Logits baseline in our paper uses temperature=1 for logit scaling, consistent with the mathematical formulation described in [Feng et al.](https://arxiv.org/abs/2408.10012). We note that [Liang et al.](https://arxiv.org/abs/2310.10463v1), who also concurrently proposed this baseline, specify temperature as a hyperparameter. In their updated version (which we observed after our publication), this is set to 0.01.

We now evaluated the baseline across multiple temperature values (0.01, 0.015, 0.07, 1) and verified that our method consistently outperforms it on three out of four classification datasets. However, the magnitude of AUROC gains of our method varies: lower performance gains of 0.7% (temperature = 0.01; still significant) to over 3% (temperatures of 0.07 or 1) for label error detection over the best baseline. Thus, the degree of performance gains are somewhat sensitive to temperature scaling in this baseline.

Downstream filtering performance is stable across temperatures, with low impact to observed trends in most datasets even at a temperature of 0.01. Note that CLIP Logits relies on a pre-defined set of classification labels and is therefore not a baseline for captioning datasets.
