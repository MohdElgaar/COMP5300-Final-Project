from torch import nn
from transformers import AutoModel, AutoConfig

class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        model_name, num_labels, ckpt = args.model_name, args.num_labels, args.ckpt

        if ckpt is not None:
            self.backbone = AutoModel.from_pretrained(ckpt.replace('meta', 'model'))
        else:
            self.backbone = AutoModel.from_pretrained(model_name)

        config = AutoConfig.from_pretrained(model_name)
        self.config = config
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.classifier.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            # module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            module.weight.data.normal_(mean=0.0, std=0.02)
            module.bias.data.zero_()

    def forward(self, x, fs1 = None):
        model_out = self.backbone(**x)
        if len(model_out) == 2:
            sent_emb = model_out[1]
        else:
            sent_emb = model_out[0][:, 0]
        sent_emb = self.dropout(sent_emb)

        logits = self.classifier(sent_emb)

        return logits
