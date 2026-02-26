import torch
from mska_og.recognition import Recognition

class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.args = args
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []

        self.recognition_network = Recognition(cfg=model_cfg['RecognitionNetwork'],args=self.args)
        self.gloss_tokenizer = self.recognition_network.gloss_tokenizer

    def forward(self, src_input, **kwargs):
        fuse_x, body_x, left_x, right_x = self.recognition_network(src_input)
        return fuse_x, body_x, left_x, right_x
