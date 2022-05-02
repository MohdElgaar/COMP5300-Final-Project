import gradio as gr
import torch
import sys, json
from transformers import AutoModel, AutoTokenizer
from model import Model
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

parser = ArgumentParser()
parser.add_argument('ckpt')
args = parser.parse_args()
ckpt = args.ckpt
with open(ckpt.replace('best_meta.pt', 'args'), 'r') as f:
    args.__dict__ = json.load(f)

model = Model(args).to(device)
state = torch.load(ckpt)
model.load_state_dict(state['model'], strict=False)
model.backbone = AutoModel.from_pretrained(ckpt.replace('meta.pt','model')).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

class_map = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

def interface(Premise, Hypothesis):
    x = tokenizer(Premise, Hypothesis, truncation=True, return_tensors='pt').to(device)
    out = model(x)
    return class_map[out.argmax().item()]

gr.Interface(fn=interface,
        inputs=["text", "text"], outputs=["textbox"]).launch(share=True)
