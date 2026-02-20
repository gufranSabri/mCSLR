"""
This file is modified from:
https://github.com/sutwangyan/MSKA
"""

import torch
from modules.recognition import Recognition

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
    

{'ensemble_last_gls_hyp': 'ein-paar-tage rhein-pfalz 盖章 偷 盖章 steigen 盖章 k 其实 noch schnee-auf-berg noch streifen 签名 欢迎 trotzdem 后果 欢迎 后果 kreisen endlich pro verlaufen zweihundert verlaufen c verlaufen 生命 verlaufen 表情 充分', 
 'gls_ref': '政府 改革 人 废 变 少 一样 还 重复 房间 机器 房间 变 少'}