from sklearn.cluster import KMeans
import faiss
import numpy as np
from transformers import AutoTokenizer
import torch

from tqdm import trange
import time

from lib.models.utils import algorithm_class_from_scratch
from lib.utils.utils import normalize_vectors

class FaissKMeans:
    def __init__(self, embed_func = None, n_clusters=8, n_init=5, max_iter=300, seed = 42, use_gpu = False):
        self.embed_func = embed_func
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.kmeans = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.seed = seed
        self.use_gpu = use_gpu

    def fit(self, X):
        self.kmeans = faiss.Kmeans(d=X.shape[1],
                                   k=self.n_clusters,
                                   niter=self.max_iter,
                                   nredo=self.n_init,
                                   gpu = self.use_gpu,
                                   seed = self.seed,
                                   verbose = True)
        self.kmeans.cp.max_points_per_centroid = 1024 # default is 256
        self.kmeans.train(X.astype(np.float32))
        self.cluster_centers_ = self.kmeans.centroids
        self.inertia_ = self.kmeans.obj[-1]

    def predict(self, X):
        if isinstance(X, (list, tuple)):
            X = self.embed_func(X)        
        return self.kmeans.index.search(X.astype(np.float32), 1)[1]
    

def embed_text(clip_model, text_list, batch_size, device):
    algorithm, tokenizer = algorithm_class_from_scratch(
        clip_model, text_base_name='openai/clip-vit-base-patch32', img_base=None, return_tokenizer=True
    )    
    algorithm = algorithm.eval().to(device)

    emb_txt = []
    for idx in trange(0, len(text_list), batch_size, desc='Embedding text'):   
        batch = text_list[idx: idx + batch_size]     
        if clip_model == 'biomed_clip':
            encodings = tokenizer(batch).to(device)            
            with torch.no_grad():
                emb_txt.append(algorithm.encode_text(encodings).detach().cpu())
        else:
            encodings = tokenizer(
                    batch, padding="max_length", truncation=True)
            input_ids = torch.tensor(encodings["input_ids"]).to(device)
            attention_mask = torch.tensor(encodings["attention_mask"]).to(device)
            
            with torch.no_grad():
                emb_txt.append(algorithm.encode_text(input_ids, attention_mask).detach().cpu())
    emb_txt = normalize_vectors(torch.concat(emb_txt)).numpy()
    return emb_txt


def cluster_caption_text(clip_model, text_list, n_clusters = 100, device = 'cuda', cluster_use_gpu = False, random_state = 42, batch_size = 128):
    emb_txt = embed_text(clip_model, text_list, batch_size, device)
    embed_func = lambda x: embed_text(clip_model, x, batch_size, device)
    km = FaissKMeans(embed_func = embed_func, n_clusters = n_clusters, seed = random_state, use_gpu = cluster_use_gpu)
    km.fit(emb_txt)

    return km, km.predict(emb_txt).squeeze()
