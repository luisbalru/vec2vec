import os
import random
import toml
from sys import argv
from types import SimpleNamespace

import accelerate

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE  # Use TSNE instead of PCA

# from eval import eval_model
from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from scipy.stats import bootstrap
from utils.dist import get_rank
from utils.model_utils import get_sentence_embedding_dimension, load_encoder
from utils.utils import *
from utils.streaming_utils import load_streaming_embeddings, process_batch
from gudhi.sklearn.rips_persistence import RipsPersistence
from gudhi.representations import DiagramSelector, Landscape
from sklearn.pipeline import Pipeline

def plot_average_landscape(landscapes, color, label):
    lands = landscapes[0]
    plt.scatter(np.arange(0,lands.shape[0]), lands, color=color, label=label)


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
    with torch.no_grad():
        translator.eval()
        batch = next(iter(evalloader))

        ins = process_batch(batch, {**sup_encs, **unsup_enc}, cfg.normalize_embeddings, accelerator.device)
        _, _, reps = translator(ins, include_reps=True)

        print(reps[cfg.sup_emb].shape)
        print(reps[cfg.unsup_emb].shape)
        print("Latents", torch.nn.functional.cosine_similarity(reps[cfg.sup_emb], reps[cfg.unsup_emb]).mean())
        print("Inputs", torch.nn.functional.cosine_similarity(ins[cfg.sup_emb], ins[cfg.unsup_emb]).mean())

        ins_sup_array = ins[cfg.sup_emb].cpu().numpy()
        #ins_sup_array = ins_sup_array.reshape(ins_sup_array.shape[0],1, ins_sup_array.shape[1])
        ins_sup = [ins_sup_array]
        ins_unsup_array = ins[cfg.unsup_emb].cpu().numpy()
        ins_unsup = [ins_unsup_array]
        #ins_combined = np.concatenate([ins_sup_array, ins_unsup_array], axis=0)

        

        # Second subplot - Intermediate representations
        reps_sup_array = reps[cfg.sup_emb].cpu().numpy()
        #reps_sup_array = reps_sup_array.reshape(reps_sup_array.shape[0],1, reps_sup_array.shape[1])
        reps_sup = [reps_sup_array]
        reps_unsup_array = reps[cfg.unsup_emb].cpu().numpy()
        reps_unsup = [reps_unsup_array]
        #reps_combined = np.concatenate([reps_sup_array, reps_unsup_array], axis=0)

        """
        point_clouds = [ins_sup_array, reps_sup_array]

        rips_transformer = RipsPersistence(
            homology_dimensions=, 
            threshold=5.0,
            input_type='point cloud', 
            n_jobs=-1
        )

        diagrams = rips_transformer.fit_transform(point_clouds)
        """
        # Constant for plot_average_landscape
        landscape_resolution = 600


        pipe = Pipeline(
            [
                ("rips_pers", RipsPersistence(homology_dimensions=1, n_jobs=-1)),
                ("finite_diags", DiagramSelector(use=True, point_type="finite")),
                ("landscape", Landscape(num_landscapes=1,resolution=landscape_resolution)),
            ]
        )
        
        pipe.fit(ins_sup + reps_sup+ins_unsup+reps_unsup)

        plot_average_landscape(pipe.transform(ins_sup), 'red', 'ins sup')
        #plot_average_landscape(pipe.transform(reps_sup), 'green', 'reps sup')
        #plot_average_landscape(pipe.transform(ins_unsup), 'blue', 'ins unsup')
        #plot_average_landscape(pipe.transform(reps_unsup), 'yellow', 'reps unsup')
        

        
        plt.title('Average landscapes')
        plt.legend()
        if no_plot == False:
            plt.show()
        else:
            plt.savefig('prueba.pdf')

if __name__ == "__main__":
    main()