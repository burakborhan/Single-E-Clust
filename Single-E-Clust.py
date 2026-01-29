# -*- coding: utf-8 -*-
import os
import time
import logging
import numpy as np
import pandas as pd
import torch

from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from torch.utils.data import DataLoader, Dataset
import umap.umap_ as umap
from sentence_transformers import SentenceTransformer, models
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

try:
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None



SEEDS = ['''0''']

ITER_N = 1
EPOCHS = 1
BATCH_SIZE = 64
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
ACCUMULATION_STEPS = 1
USE_AMP = False

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RESULTS_DIR = r"~\results_dir"
CACHE_DIR = os.path.join(RESULTS_DIR, "cache_embeddings")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ────────── Loglama ──────────
def setup_logger(results_dir: str) -> logging.Logger:
    lg = logging.getLogger("ablation")
    lg.setLevel(logging.INFO)
    lg.propagate = False

    if lg.handlers:
        return lg

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    lg.addHandler(ch)

    log_path = os.path.join(results_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    lg.addHandler(fh)

    lg.info(f"[Logger] File logging → {log_path}")
    return lg

logger = setup_logger(RESULTS_DIR)

# ────────── Yardımcılar ──────────
def prepare_texts(df: pd.DataFrame, text_col) -> list:
    if isinstance(text_col, list):
        texts = (
            df[text_col]
            .astype(str)
            .agg(" ".join, axis=1)
            .tolist()
        )
        logger.info(f"[TextPrep] Loaded {len(texts)} texts from cols={text_col}")
    else:
        texts = df[text_col].astype(str).tolist()
        logger.info(f"[TextPrep] Loaded {len(texts)} texts from col={text_col}")
    return texts



def estimate_mle_dim(X: np.ndarray, k: int = 20) -> int:
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    dists = dist[:, 1:]
    dk = dists[:, -1][:, None]
    logs = np.log(dk / dists[:, :-1])
    with np.errstate(divide='ignore', invalid='ignore'):
        m = 1.0 / logs.mean(axis=1)
    valid = m[np.isfinite(m)]
    dim = int(valid.mean()) if valid.size > 1 else X.shape[1]
    logger.info(f"[DimEst] MLE dim = {dim} (k={k})")
    return dim

def clustering_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cm = contingency_matrix(y_true, y_pred)
    if linear_sum_assignment is None:
        raise ImportError(
            "Accuracy için scipy gerekiyor: pip install scipy "
            "(scipy.optimize.linear_sum_assignment bulunamadı)."
        )
    r, c = linear_sum_assignment(-cm)
    return float(cm[r, c].sum()) / float(cm.sum() + 1e-12)

class ConDataset(Dataset):
    def __init__(self, txt, lab):
        self.txt, self.lab = txt, lab
    def __len__(self): return len(self.txt)
    def __getitem__(self, i): return self.txt[i], self.lab[i],i

def collate(batch, tok, dev):
    texts, labels,idx = zip(*batch)
    enc = tok(list(texts))
    enc = {k: v.to(dev) for k, v in enc.items()}
    return enc, torch.tensor(labels, device=dev), torch.tensor(idx)

class SupConLoss(torch.nn.Module):
    def __init__(self, T=0.07):
        super().__init__()
        self.T = T
    def forward(self, x, y):
        y = y.view(-1, 1)
        mask = (y == y.T).float().to(x.device)
        mask = mask * (1 - torch.eye(len(x), device=x.device))
        x = torch.nn.functional.normalize(x, dim=1)
        logit = x @ x.T / self.T
        exp = torch.exp(logit) * (1 - torch.eye(len(x), device=x.device))
        logp = logit - torch.log(exp.sum(1, keepdim=True) + 1e-12)
        loss_i = -((mask * logp).sum(1) / (mask.sum(1) + 1e-12))
        return loss_i.mean()

def build_sbert(pool_mean=True, pool_max=False, pool_cls=False) -> SentenceTransformer:
    tr = models.Transformer(MODEL_NAME, max_seq_length=256)
    pl = models.Pooling(
        tr.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=pool_mean,
        pooling_mode_max_tokens=pool_max,
        pooling_mode_cls_token=pool_cls
    )
    return SentenceTransformer(modules=[tr, pl])

def compute_final_space(method: str, emb: np.ndarray, d_hat: int, seed: int) -> np.ndarray:

    if method.endswith("_umap"):
        reducer = umap.UMAP(
            n_components=d_hat,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            set_op_mix_ratio=1.0,
            local_connectivity=1,
            random_state=seed
        )
        return reducer.fit_transform(emb)
    return emb

def eval_cluster(space: np.ndarray, y_true: np.ndarray, seed: int) -> Tuple[np.ndarray, float, float, float, float]:
    K = len(np.unique(y_true))
    preds = KMeans(K, random_state=seed, n_init="auto").fit_predict(space)
    sil = silhouette_score(space, preds)
    ari = adjusted_rand_score(y_true, preds)
    nmi = normalized_mutual_info_score(y_true, preds)
    acc = clustering_accuracy(y_true, preds)
    return preds, sil, ari, nmi, acc

def supcon_step(
    model,
    texts,
    pseudo,
    loss_fn,
    dev,
    seed,
    log_tag=None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    grad_clip_norm: float = GRAD_CLIP_NORM,
    accumulation_steps: int = ACCUMULATION_STEPS,
    use_amp: bool = USE_AMP
):
    # Dataset + DataLoader
    ds = ConDataset(texts, pseudo)

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate(b, model.tokenize, dev)
    )

    # Total step count (epoch * num_batches)
    num_batches = len(loader)
    total_steps = epochs * max(1, num_batches)
    warmup_steps = int(warmup_ratio * total_steps)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # AMP (FP16)
    scaler = GradScaler(enabled=(use_amp and dev.type == "cuda"))

    model.train()

    global_step = 0
    first = True

    for epoch in range(epochs):
        epoch_loss = 0.0

        for step, (enc, lab, idx) in enumerate(loader):
            if first and log_tag is not None:
                logger.info(f"[BatchIdx] {log_tag} first10={idx[:10].tolist()}")
                first = False

            with autocast(enabled=(use_amp and dev.type == "cuda")):
                out = model(enc)
                emb = out["sentence_embedding"]
                loss = loss_fn(emb, lab) / accumulation_steps  # accumulate

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * accumulation_steps

            # Gradient accumulation
            if (step + 1) % accumulation_steps == 0:
                # Gradient clipping
                if grad_clip_norm is not None and grad_clip_norm > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

                # Step
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        logger.info(
            f"[EpochDone] {log_tag} | epoch={epoch+1}/{epochs} | "
            f"avg_loss={epoch_loss / max(1, num_batches):.6f}"
        )

    model.eval()



# ────────── Çekirdek fonksiyon ──────────
def run_method(method: str,
               texts: List[str],
               true_labels: np.ndarray,
               model: SentenceTransformer,
               ds_tag: str,
               pool_tag: str,
               seed: int,
               n_iters: int = 1
               ) -> Tuple[float, float, float, float, List[Tuple[str, float, float, float, float, int]]]:

    dev = next(model.parameters()).device
    K = len(np.unique(true_labels))

    # ---- 1) Raw cache (seed-independent) ----
    raw_path = os.path.join(CACHE_DIR, f"{ds_tag}_{pool_tag}_raw.npy")
    if os.path.exists(raw_path):
        raw = np.load(raw_path)
        logger.info(f"[CacheHit] raw → {os.path.basename(raw_path)} shape={raw.shape}")
    else:
        raw = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
        np.save(raw_path, raw)
        logger.info(f"[CacheSave] raw → {os.path.basename(raw_path)} shape={raw.shape}")

    d_hat = max(2, estimate_mle_dim(raw, k=20))
    logger.info(f"{ds_tag} | {pool_tag} | seed={seed} | method={method} | MLE dim(d_hat)={d_hat}")

    reducer_tpl = umap.UMAP(
        n_components=d_hat,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        set_op_mix_ratio=1.0,
        local_connectivity=1,
        random_state=seed
    )

    early = reducer_tpl.fit_transform(raw) if method in ("umap", "supcon_umap") else raw
    logger.info(f"[PseudoSrc] method={method} | early_shape={early.shape} | early_is_umap={method in ('umap','supcon_umap')}")

    iter_rows: List[Tuple[str, float, float, float, float, int]] = []


    loss_fn = SupConLoss().to(dev)

    if method in ("supcon", "supcon_umap"):
        # pseudo: supcon -> raw, supcon_umap -> UMAP(raw)
        pseudo = KMeans(K, random_state=seed, n_init="auto").fit_predict(early)
        logger.info(f"[PseudoInfo] method={method} | pseudo_unique={len(np.unique(pseudo))} | K={K}")

        supcon_step(
            model=model,
            texts=texts,
            pseudo=pseudo,
            loss_fn=loss_fn,
            dev=dev,
            seed=seed,
            log_tag=method
        )

    elif method in ("iter_supcon", "iter_supcon_umap"):

        for t in range(n_iters):

            if t == 0:
                if method == "iter_supcon_umap":
                    z_t = compute_final_space("x_umap", raw, d_hat=d_hat, seed=seed)  # UMAP(raw)
                else:
                    z_t = raw  # raw
            else:

                emb_t = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
                if method == "iter_supcon_umap":
                    z_t = compute_final_space("x_umap", emb_t, d_hat=d_hat, seed=seed)
                else:
                    z_t = emb_t

            pseudo = KMeans(K, random_state=seed, n_init="auto").fit_predict(z_t)
            logger.info(f"[IterPseudo] t={t+1}/{n_iters} | method={method} | pseudo_unique={len(np.unique(pseudo))} | K={K}")

            supcon_step(
                model=model,
                texts=texts,
                pseudo=pseudo,
                loss_fn=loss_fn,
                dev=dev,
                seed=seed,
                log_tag=f"{method}_iter{t+1}"
            )

            emb_after = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

            final_space_t = compute_final_space(
                "x_umap" if method == "iter_supcon_umap" else "x_raw",
                emb_after,
                d_hat=d_hat,
                seed=seed
            )

            _, sil_i, ari_i, nmi_i, acc_i = eval_cluster(final_space_t, true_labels, seed=seed)

            m_iter = f"{method}_iter{t+1}"
            iter_rows.append((m_iter, sil_i, ari_i, nmi_i, acc_i, t+1))

            logger.info(
                f"[IterResult] {ds_tag} | {pool_tag} | {m_iter} | seed={seed} | "
                f"sil={sil_i:.4f} ari={ari_i:.4f} nmi={nmi_i:.4f} acc={acc_i:.4f}"
            )

            d_used_it = d_hat if method == "iter_supcon_umap" else None
            d_tag_it = f"d{d_used_it}" if d_used_it is not None else "dNA"
            fin_path_iter = os.path.join(
                CACHE_DIR,
                f"{ds_tag}_{pool_tag}_{m_iter}_final_{d_tag_it}_seed{seed}_iter{t+1}.npy"
            )
            if not os.path.exists(fin_path_iter):
                np.save(fin_path_iter, final_space_t)
                logger.info(f"[CacheSave] iter_final → {os.path.basename(fin_path_iter)} shape={final_space_t.shape}")

    final_emb = model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)

    if method in ("umap", "supcon_umap", "iter_supcon_umap"):
        final_space = compute_final_space("x_umap", final_emb, d_hat=d_hat, seed=seed)
        d_used = d_hat
    else:
        final_space = final_emb
        d_used = None

    d_tag = f"d{d_used}" if d_used is not None else "dNA"
    iter_tag = f"_iter{n_iters}" if method in ("iter_supcon", "iter_supcon_umap") else ""
    fin_path = os.path.join(
        CACHE_DIR,
        f"{ds_tag}_{pool_tag}_{method}_final_{d_tag}_seed{seed}{iter_tag}.npy"
    )

    if os.path.exists(fin_path):
        final_space_cached = np.load(fin_path)
        final_space = final_space_cached
        logger.info(f"[CacheHit] final → {os.path.basename(fin_path)} shape={final_space.shape}")
    else:
        np.save(fin_path, final_space)
        logger.info(f"[CacheSave] final → {os.path.basename(fin_path)} shape={final_space.shape}")

    _, sil, ari, nmi, acc = eval_cluster(final_space, true_labels, seed=seed)

    logger.info(
        f"[Metrics] {ds_tag} | {pool_tag} | method={method} | seed={seed}{iter_tag} | "
        f"sil={sil:.4f} ari={ari:.4f} nmi={nmi:.4f} acc={acc:.4f}"
    )
    return sil, ari, nmi, acc, iter_rows



# ────────── Veri kümeleri ──────────
DATASETS = [
    #{
    #   "path": r"~\BBC-News.csv",
    #   "text_col": ["text"], "label_col": "label"
    #},
    #{
    #   "path": r"~\AG-News.csv",
    #   "text_col": ["text"], "label_col": "label"
    #},
    #{
    #   "path": r"~\HuffPost-News.csv",
    #   "text_col": ["text"], "label_col": "label"
    #},
    #{
    #    "path": r"~\yahoo.csv",
    #    "text_col": ["text"], "label_col": "label"
    #},
]

def main():
    methods = ["raw", "umap", "iter_supcon_umap" ] #, "iter_supcon", "supcon", "supcon_umap"
    pool_cfgs = [("mean", True, False, False)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"[Env] device={device} | torch={torch.__version__}")
    logger.info(f"[Config] seeds={SEEDS} | ITER_N={ITER_N} | methods={methods} | pools={[p[0] for p in pool_cfgs]}")
    logger.info(f"[Paths] RESULTS_DIR={RESULTS_DIR} | CACHE_DIR={CACHE_DIR}")

    for SEED in SEEDS:
        logger.info(f"\n{'='*24} SEED is starting: {SEED} {'='*24}\n")
        all_rows = []

        for ds in DATASETS:
            ds_tag = os.path.splitext(os.path.basename(ds["path"]))[0]
            logger.info(f"▶ Dataset is starting: {ds_tag} | SEED={SEED} | path={ds['path']}")

            df = pd.read_csv(ds["path"])
            texts = prepare_texts(df, ds["text_col"])
            labels = df[ds["label_col"]].values

            rows = []
            for pool_tag, pm, pmax, pcls in pool_cfgs:
                base_model = build_sbert(pm, pmax, pcls)
                base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

                raw_path = os.path.join(CACHE_DIR, f"{ds_tag}_{pool_tag}_raw.npy")
                if not os.path.exists(raw_path):
                    logger.info(f"[RawWarmup] missing raw cache → computing with SBERT₀ | {ds_tag} | pool={pool_tag}")
                    base_model = base_model.to(device)
                    raw0 = base_model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)
                    np.save(raw_path, raw0)
                    logger.info(f"[CacheSave] raw → {os.path.basename(raw_path)} shape={raw0.shape}")
                    del raw0
                    base_model = base_model.cpu()
                else:
                    logger.info(f"[CacheHit] raw already exists → {os.path.basename(raw_path)}")

                for m in methods:
                    model = build_sbert(pm, pmax, pcls).to(device)
                    model.load_state_dict(base_state, strict=True)

                    n_it = ITER_N if m in ("iter_supcon", "iter_supcon_umap") else 1

                    logger.info(f"{ds_tag} | pool={pool_tag} | method={m} | SEED={SEED} | n_iters={n_it}")
                    t0 = time.time()

                    sil, ari, nmi, acc, iter_rows = run_method(
                        m, texts, labels, model,
                        ds_tag=ds_tag, pool_tag=pool_tag, seed=SEED,
                        n_iters=n_it
                    )

                    dt = time.time() - t0
                    logger.info(f"[Done] {ds_tag} | {pool_tag} | {m} | seed={SEED} | time={dt:.1f}s")

                    rows.append((ds_tag, pool_tag, m, sil, ari, nmi, acc, n_it))

                    for (m_iter, sil_i, ari_i, nmi_i, acc_i, it_i) in iter_rows:
                        rows.append((ds_tag, pool_tag, m_iter, sil_i, ari_i, nmi_i, acc_i, it_i))

                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            df_set = pd.DataFrame(rows, columns=["dataset", "pool", "method", "sil", "ari", "nmi", "acc", "n_iters"])
            out_path = os.path.join(RESULTS_DIR, f"results_{ds_tag}_seed{SEED}.csv")
            df_set.to_csv(out_path, index=False)

            print("\n" + "=" * 84)
            print(f"Results ({ds_tag}) | seed={SEED}:\n",
                  df_set.to_string(index=False, formatters={
                      "sil": "{:.4f}".format,
                      "ari": "{:.4f}".format,
                      "nmi": "{:.4f}".format,
                      "acc": "{:.4f}".format
                  }))
            print("=" * 84 + "\n")

            all_rows.extend(rows)
            logger.info(f"✔ {ds_tag} done → {os.path.basename(out_path)}")

        all_df = pd.DataFrame(all_rows, columns=["dataset", "pool", "method", "sil", "ari", "nmi", "acc", "n_iters"])
        out_all = os.path.join(RESULTS_DIR, f"all_datasets_ablation_seed{SEED}.csv")
        all_df.to_csv(out_all, index=False)

        print(f"✓ SEED={SEED} done — bulk file: {os.path.basename(out_all)}")
        logger.info(f"[SeedDone] SEED={SEED} → {os.path.basename(out_all)}")

if __name__ == "__main__":
    main()
