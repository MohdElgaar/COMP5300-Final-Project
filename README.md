# COMP-5300-Final-Project

# Curriculum Learning for Natrual Language Processing
- Perform natural language inference (NLI) using curriculum learning (CL).
- Using `ChaosNLI`. The dataset contains multiple annotations for each instance. 
  - Use inter-annotator disagreement as a difficulty score. 
  - Apply a curriculum that learns from easy to hard
- Compare with CL baselines Mentornet, SuperLoss, Difficulty Prediction (DP), No-CL (standard training).

# Prepare data
`python prepare_snli.py`
`python prepare_chaso.py`

# Training
`python train.py --data [chaosnli, snli] --curr [ent, sl, dp, mentornet, none] --epochs 2 --grad_accumulation 8`

# Interacting
`python gradio.py [meta ckpt file]`

# Requirements
- Python 3.6
- pip 21.3.1

# Issue
Gradio requires python 3.7. Create a separate environment with python 3.7 and instal `torch, transformers, gradio` in order to use gradio.
