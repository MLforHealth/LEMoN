import sys
import getpass
import os
import torch
from pathlib import Path, PosixPath
import numpy as np
from itertools import combinations
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score
import json
import re

def path_serial(obj):
    if isinstance(obj, (Path, PosixPath)):
        return str(obj)
    raise TypeError(f"Object of type '{type(obj).__name__}' is not JSON serializable")

def flatten(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result

def powerset(S):
    return [list(j) for n in range(0, len(S) + 1) for j in list(combinations(S, n))]

def to_device(obj, device):
    if torch.is_tensor(obj) or isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return [obj.to(device) for i in obj]
    elif isinstance(obj, dict):
        return {a: b.to(device) if torch.is_tensor(b) or isinstance(b, torch.nn.Module) else b for a,b in obj.items() }
    else:
        raise ValueError("invalid object type passed to to_device")

def normalize_vectors(vectors):
    return torch.nn.functional.normalize(vectors, p=2, dim=1)

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class EarlyStopping:
    # adapted from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    def __init__(self, patience=5, lower_is_better = True):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.step = 0
        self.lower_is_better = lower_is_better

    def __call__(self, metric, step, state_dict, path):  
        if self.lower_is_better:
            score = -metric
        else:
            score = metric

        if self.best_score is None:
            self.best_score = score
            self.step = step
            save_model(state_dict, path)
        elif score < self.best_score:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            save_model(state_dict, path)
            self.best_score = score
            self.step = step
            self.counter = 0
    
def save_model(state_dict, path):
    torch.save(state_dict, path)


class NumpyEncoder(json.JSONEncoder):
    # reproduced from https://github.com/hmallen/numpyencoder/blob/f8199a61ccde25f829444a9df4b21bcb2d1de8f2/numpyencoder/numpyencoder.py
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)      

# functions for checkpoint/reload in case of job pre-emption on our slurm cluster
# will have to customize if you desire this functionality
# otherwise, the training script will still work fine as-is
def save_checkpoint(model, optimizer, sampler_dicts, start_step, es, rng):   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')        
    
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/').exists():        
        torch.save({'model_dict': model.state_dict(),
                    'optimizer_dict': optimizer.state_dict(),
                    'sampler_dicts': sampler_dicts,
                    'start_step': start_step,
                    'es': es,
                    'rng': rng
        } 
                   , 
                   Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').open('wb')                  
                  )
        
        
def has_checkpoint():
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if slurm_job_id is not None and Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt').exists():
        return True
    return False           

def load_checkpoint():   
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    fname = Path(f'/checkpoint/{getpass.getuser()}/{slurm_job_id}/chkpt')
    if slurm_job_id is not None and fname.exists():
        return torch.load(fname)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)

def eval_metrics(targets, model, task, train_features, test_df):
    res = {'n_samples': len(targets)}
    if task == 'regression':
        preds = model.predict(test_df[train_features])
        res['mse'] = mean_squared_error(targets, preds)
        res['mae'] = mean_absolute_error(targets, preds)
    elif task == 'classification':
        preds = model.predict_proba(test_df[train_features])[:, 1]
        res['auroc'] = roc_auc_score(targets, preds)
        res['auprc'] = average_precision_score(targets, preds)    
    return res

def str_to_dist(st):
    st1 = st.replace(" ", "")
    marginal = re.findall(r"[p|P]\(([0-9a-zA-Z_-]+)\)", st1)
    if marginal:
        return marginal[0][0]
    cond = re.findall(r"[p|P]\(([0-9a-zA-Z_-]+)\|([0-9a-zA-Z,_-]+)\)", st1)
    if cond:
        a,b = cond[0]
        return (frozenset(b.split(',')), a)
    return st    

