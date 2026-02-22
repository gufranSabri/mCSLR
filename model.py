"""
This file is modified from:
https://github.com/sutwangyan/MSKA
"""

import torch
from model_components.recognition import Recognition

class SLR_Model(torch.nn.Module):
    def __init__(self, cfg, args):
        super().__init__()
        self.args = args
        self.device = args.device
        model_cfg = cfg['model']
        self.net = Recognition(cfg=model_cfg['RecognitionNetwork'], args=self.args)

    def forward(self, src_input):
        recognition_outputs = self.net(src_input)
        model_outputs = {**recognition_outputs}
        model_outputs['total_loss'] = recognition_outputs['loss']

        return model_outputs

    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.net.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths
        )