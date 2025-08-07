import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE  # Use TSNE instead of PCA

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from scipy.stats import bootstrap
from utils.dist import get_rank
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch
from gudhi.sklearn.rips_persistence import RipsPersistence
from gudhi.representations.vector_methods import Entropy
from gudhi.representations.metrics import BottleneckDistance, pairwise_persistence_diagram_distances
from gudhi.representations import DiagramSelector, Landscape
import gudhi
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
#import plotly.graph_objects as go


def plot_average_landscape(landscapes, color, label, fig):
    lands = landscapes[0]
    x=np.arange(0,lands.shape[0])
    fig.add_trace(go.Scatter(x=x, y=lands, mode='lines+markers',name=label, marker=dict(color=color)))
    return fig


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    cfg = toml.load(f'{argv[1]}/config.toml')
    unknown_cfg = read_args(argv)
    cfg = SimpleNamespace(**{**cfg, **unknown_cfg})

    if hasattr(cfg, 'mixed_precision') and cfg.mixed_precision == 'bf16' and not torch.cuda.is_bf16_supported():
        cfg.mixed_precision = 'fp16'
        print("Note: bf16 is not available on this hardware!")

    # set seeds
    random.seed(cfg.seed + get_rank())
    torch.manual_seed(cfg.seed + get_rank())
    np.random.seed(cfg.seed + get_rank())
    torch.cuda.manual_seed(cfg.seed + get_rank())

    accelerator = accelerate.Accelerator(
        mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None
    )
    accelerator.dataloader_config.dispatch_batches = False

    dset = load_streaming_embeddings(cfg.dataset)

    sup_encs = {cfg.sup_emb: load_encoder(cfg.sup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)}
    encoder_dims = {cfg.sup_emb: get_sentence_embedding_dimension(sup_encs[cfg.sup_emb])}
    translator = load_n_translator(cfg, encoder_dims)

    assert hasattr(cfg, 'unsup_emb')
    assert cfg.sup_emb != cfg.unsup_emb

    unsup_enc = {
        cfg.unsup_emb: load_encoder(cfg.unsup_emb, mixed_precision=cfg.mixed_precision if hasattr(cfg, 'mixed_precision') else None)
    }
    unsup_dim = {
        cfg.unsup_emb: get_sentence_embedding_dimension(unsup_enc[cfg.unsup_emb])
    }
    translator.add_encoders(unsup_dim, overwrite_embs=[cfg.unsup_emb])

    assert cfg.unsup_emb not in sup_encs
    assert cfg.unsup_emb in translator.in_adapters
    assert cfg.unsup_emb in translator.out_adapters

    cfg.num_params = sum(x.numel() for x in translator.parameters())
    print("Number of parameters:", cfg.num_params)

    dset_dict = dset.train_test_split(test_size=cfg.val_size, seed=cfg.val_dataset_seed)
    dset = dset_dict["train"]
    valset = dset_dict["test"]

    assert hasattr(cfg, 'num_points') or hasattr(cfg, 'unsup_points')
    dset = dset.shuffle(seed=cfg.train_dataset_seed)
    if hasattr(cfg, 'num_points'):
        assert cfg.num_points > 0 and cfg.num_points <= len(dset) // 2
        unsupset = dset.select(range(cfg.num_points, cfg.num_points + cfg.val_size))
    elif hasattr(cfg, 'unsup_points'):
        unsupset = dset.select(range(min(cfg.unsup_points, len(cfg.val_size))))

    num_workers = get_num_proc()
    evalset = MultiencoderTokenizedDataset(
        dataset=valset if hasattr(cfg, 'use_ood') and cfg.use_ood else unsupset,
        encoders={**unsup_enc, **sup_encs},
        n_embs_per_batch=2,
        batch_size=cfg.val_bs,
        max_length=cfg.max_seq_length,
        seed=cfg.sampling_seed,
    )
    evalloader = DataLoader(
        evalset,
        batch_size=cfg.val_bs if hasattr(cfg, 'val_bs') else cfg.bs,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=(8 if num_workers > 0 else None),
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    evalloader = accelerator.prepare(evalloader)

    assert hasattr(cfg, 'load_dir')
    print(f"Loading models from {argv[1]}...")
    translator.load_state_dict(torch.load(f'{argv[1]}/model.pt', map_location='cpu'), strict=False)
    translator = accelerator.prepare(translator)
    no_plot = True

    ALPHA = 0.7
    WIDTH = 0.5
    C1 = '#008080'
    C2 = '#FF6347'
    LINEWIDTH = 0.5
    scaler = MinMaxScaler()
    with torch.no_grad():
        translator.eval()
        batch = next(iter(evalloader))

        ins = process_batch(batch, {**sup_encs, **unsup_enc}, cfg.normalize_embeddings, accelerator.device)
        recons, trans, reps = translator(ins, include_reps=True)


        reps_sup_array = reps[cfg.sup_emb].cpu().numpy()
        reps_sup_array = scaler.fit_transform(reps_sup_array)
        reps_sup = [reps_sup_array]

        reps_unsup_array = reps[cfg.unsup_emb].cpu().numpy()
        reps_unsup_array = scaler.fit_transform(reps_unsup_array)
        reps_unsup = [reps_unsup_array]


        # Constant for plot_average_landscape
        
        landscape_resolution = 2000



        # PERSISTENCE DIAGRAMS
        complex1 = gudhi.RipsComplex(
            distance_matrix = reps_sup_array, 
            max_edge_length = 0.8
        ) 

        rips_simple1 = complex1.create_simplex_tree(max_dimension = 2)

        BarCodes_Rips1 = rips_simple1.persistence()
        BarCodes_Rips1_d0 = np.array([d[1] for d in BarCodes_Rips1])


        complex2 = gudhi.RipsComplex(
            distance_matrix = reps_unsup_array, 
            max_edge_length = 0.8
        ) 

        rips_simple2 = complex2.create_simplex_tree(max_dimension = 2)

        BarCodes_Rips2 = rips_simple2.persistence()
        BarCodes_Rips2_d0 = np.array([d[1] for d in BarCodes_Rips2])
        print("Wasserstein distance T(A1,A2)-T(A2,A1)")
        print(gudhi.hera.wasserstein_distance(BarCodes_Rips1_d0, BarCodes_Rips2_d0))



        for i in range(3):
            pipe = Pipeline(
                [
                    ("rips_pers", RipsPersistence(homology_dimensions=i, n_jobs=-1)),
                    ("finite_diags", DiagramSelector(use=True, point_type="finite")),
                    ("landscape", Landscape(num_landscapes=1,resolution=landscape_resolution)),
                ]
            )
            
            
            # PERSISTENCE LANDSCAPE
            plot = False
            pipe.fit(reps_sup + reps_unsup)
            reps_sup_transf = pipe.transform(reps_sup)
            reps_unsup_transf = pipe.transform(reps_unsup)
        
            """
            if plot:
                fig = go.Figure()

                fig = plot_average_landscape(ins_sup_transf, '#D9631E', cfg.sup_emb+': M2(di)', fig)
                fig = plot_average_landscape(trans_unsup_transf, '#342BDC', 'Translation: F(M1(di))', fig)
                fig = plot_average_landscape(ins_unsup_transf, '#3DD91E', cfg.unsup_emb+': M1(di)', fig)        

                fig.write_html('prueba_rq2_d2.html')
            """
            
            print("HOMOLOGY DIMENSION: ", i)
            print("L2(T(A1,A2), T(A2,A1))")
            print(np.linalg.norm(reps_sup_transf-reps_unsup_transf))
        
        

if __name__ == "__main__":
    main()