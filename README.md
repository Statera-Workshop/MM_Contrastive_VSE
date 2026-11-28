<!-- PAI-SG-LINKS:BEGIN -->
## Project Links

- [Project Portal](https://statera-workshop.github.io/#repo-mm_contrastive_vse)
- [Statera Workshop Portal](https://statera-workshop.github.io/)
<!-- PAI-SG-LINKS:END -->

# MM_Contrastive_VSE

**VSE0 / VSE++ image-text embedding on Crop36-Text and COCO - convergence study (PAI-SG Multimodal Wave 1)**

Part of the *Physical AI* reference series of the **Statera Working Group**. One method of the image-text convergence comparison in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO).

## Method

Hinge (margin 0.2) on cosine similarity, both directions. *VSE0*: sum over all in-batch negatives (the VSE++ paper's baseline). *VSE++*: only the hardest negative (max violation, as in the official code). The official VSE++ trains on pre-trained image features; here both encoders start from scratch.

**Common setup** (identical for the contrastive repositories; full comparison in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO)).
Dual encoder from scratch: image ResNet-18 (torchvision layout), text Transformer (4 layers, width 256, 4 heads, causal,
EOT pooling; OpenAI CLIP BPE tokenizer), both projected to a 256-d L2-normalised embedding. AdamW lr 5e-4, wd 0.1,
500 warm-up steps then constant (cosine = ablation), batch 256, no augmentation, fp32, seeds 777 / 778 / 779.
Data: **Crop36-Text** - the 26,716 / 6,699 Crop36 object crops (96 x 96, grayscale, 36 classes) of FashionLetters36 with
captions from the class name and the original box size (8 templates); **COCO subset** - 20,000 train2017 / 1,000
val2017 images (seed 777, 128 x 128), 5 captions each. Metrics: Crop36 zero-shot top-1 (3 size-free prompts, ensemble;
chance 0.028); COCO rsum = Recall@1+5+10 image->text + text->image on the 1,000 val images (max 6).

**BatchNorm finding.** With a constant learning rate the BatchNorm running statistics lag the moving weights, so the
plain eval-mode score shows isolated *dips* (a single evaluation falls, the next recovers) and, on COCO, a systematic
under-estimate. Re-estimating the BN statistics on 50 training batches before evaluation (the series' evaluation
practice) removes every dip and gives the *BN re-estimated* column; training is unchanged. With the cosine schedule the
two evaluations agree. *Dips* = evaluation-to-evaluation drops >= 0.1 (>= 0.6 rsum).

## Results (mean of seeds; range in brackets)

### Crop36-Text

| Loss | Init | Batch | Epochs | LR schedule | Seeds | Final zero-shot top-1 (range) | Final, BN re-estimated (range) | E90 | Dips (plain / BN re-est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VSE0 (sum of hinges) | scratch | 256 | 30 | constant | 3 | 0.898 (0.026) | 0.907 (0.008) | 4.0 | 1 / - |
| VSE0 (sum of hinges) | scratch | 256 | 100 | constant | 3 | 0.894 (0.017) | 0.903 (0.001) | 7.0 | 0 / 0 |
| VSE0 (sum of hinges) | scratch | 64 | 30 | constant | 2 | 0.903 (0.017) | 0.910 (0.004) | 3.0 | 0 / - |
| VSE0 (sum of hinges) | SSL Barlow Twins | 256 | 30 | constant | 3 | 0.924 (0.005) | 0.924 (0.004) | 2.0 | 0 / 0 |
| VSE0 (sum of hinges) | SSL Barlow Twins | 256 | 100 | constant | 3 | 0.923 (0.005) | 0.923 (0.005) | 3.0 | 0 / 0 |
| VSE++ (max violation) | scratch | 256 | 30 | constant | 3 | 0.026 (0.005) | 0.026 (0.008) | 1.0 | 0 / - |
| VSE++ (max violation) | scratch | 256 | 100 | constant | 3 | 0.026 (0.004) | 0.029 (0.006) | 3.0 | 0 / - |
| VSE++ (max violation) | scratch | 64 | 30 | constant | 2 | 0.028 (0.006) | 0.026 (0.003) | 1.0 | 0 / - |

### COCO subset

| Loss | Init | Batch | Epochs | LR schedule | Seeds | Final rsum (R@1+5+10, both directions, max 6) (range) | Final, BN re-estimated (range) | E90 | Dips (plain / BN re-est.) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VSE0 (sum of hinges) | scratch | 256 | 15 | constant | 3 | 0.537 (0.116) | 0.594 (0.088) | 11.0 | 0 / - |
| VSE0 (sum of hinges) | scratch | 256 | 50 | constant | 3 | 0.730 (0.057) | 0.775 (0.064) | 26.7 | 0 / 0 |
| VSE0 (sum of hinges) | scratch | 256 | 100 | constant | 3 | 0.708 (0.047) | 0.744 (0.092) | 27.0 | 0 / 0 |
| VSE++ (max violation) | scratch | 256 | 15 | constant | 3 | 0.030 (0.009) | 0.031 (0.017) | 1.0 | 0 / - |
| VSE++ (max violation) | scratch | 256 | 50 | constant | 3 | 0.035 (0.017) | 0.039 (0.006) | 1.3 | 0 / - |
| VSE++ (max violation) | scratch | 256 | 100 | constant | 3 | 0.029 (0.008) | 0.036 (0.007) | 3.0 | 0 / - |

**Finding.** VSE++ collapses in every setting: the loss stays at exactly 2 x batch x margin (all embeddings equal) and zero-shot / retrieval stay at chance, with or without BN re-estimation. VSE0 - the same hinge summed over all negatives - converges normally. From scratch, using only the hardest negative is what breaks training.

**SSL initialisation (Crop36).** Image encoder initialised from the SSL Barlow Twins 200-epoch backbone of [SSL_Redundancy_BarlowTwins](https://github.com/Statera-Workshop/SSL_Redundancy_BarlowTwins) (seed-matched; conv1 folded to one input channel), Barlow vs scratch: VSE0 30 epochs: E90 2.0 vs 4.0, final (BN re-est.) 0.924 vs 0.907; VSE0 100 epochs: E90 3.0 vs 7.0, final (BN re-est.) 0.923 vs 0.903.

## Weights

Release **v1.0** of this repository (15 files, SHA-256 in `SHA256SUMS.txt`): `coco_vse0_e100_const_s777.pth`, `coco_vse0_e100_const_s778.pth`, `coco_vse0_e100_const_s779.pth`, `coco_vsepp_e100_const_s777.pth`, `coco_vsepp_e100_const_s778.pth`, `coco_vsepp_e100_const_s779.pth`, `crop36_vse0_e100_const_barlow_s777.pth`, `crop36_vse0_e100_const_barlow_s778.pth`, `crop36_vse0_e100_const_barlow_s779.pth`, `crop36_vse0_e100_const_s777.pth`, `crop36_vse0_e100_const_s778.pth`, `crop36_vse0_e100_const_s779.pth`, `crop36_vsepp_e100_const_s777.pth`, `crop36_vsepp_e100_const_s778.pth`, `crop36_vsepp_e100_const_s779.pth`.

```python
import torch, sys; sys.path.insert(0, 'code'); import mm_b as M
tok = M.Tok(32)            # Crop36: context 32, 1 input channel; COCO: M.Tok(64), 3 channels
m = M.DualEncoder('<loss>', tok.vocab, 32, 1)
m.load_state_dict(torch.load('crop36_<loss>_e100_<const|cosine>_s<seed>.pth', weights_only=True), strict=True)
```

**Data note.** COCO images follow the Flickr terms and are not redistributed; the image ids are listed in [MM_Wave1B_Crop36COCO](https://github.com/Statera-Workshop/MM_Wave1B_Crop36COCO) `results/data/coco_sub_list.json`.

## Run

```bash
cd code && python add_crop36_side.py && python prep_coco.py && bash smoke_b.sh
python mm_b.py --data crop36 --loss vse0 --epochs 100 --seed 777 --eval-recal
```
