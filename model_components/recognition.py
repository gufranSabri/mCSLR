"""
This file is modified from:
https://github.com/sutwangyan/MSKA
"""

import math
import numpy as np
from itertools import groupby

import torch
import torch.nn as nn
import tensorflow as tf
import torch.nn.functional as F

from utils.tokenizer import GlossTokenizer_S2G
from utils.sal import SequenceAlignmentLoss
from model_components.visualhead import VisualHead

def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits,
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )

    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )

    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences


def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0


class PositionalEncoding(nn.Module):
    def __init__(self, channel, joint_num, time_len):
        super(PositionalEncoding, self).__init__()
        self.joint_num = joint_num
        self.time_len = time_len

        pos_list = []
        for t in range(self.time_len):
            for j_id in range(self.joint_num):
                pos_list.append(j_id)

        position = torch.from_numpy(np.array(pos_list)).unsqueeze(1).float()
        pe = torch.zeros(self.time_len * self.joint_num, channel)

        div_term = torch.exp(torch.arange(0, channel, 2).float() * -(math.log(10000.0) / channel))  # channel//2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.view(time_len, joint_num, channel).permute(2, 0, 1).unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :, :x.size(2)]
        return x
    
class LanguageToken(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.token = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, x):
        token = self.token.expand(x.shape[0], -1)
        x = x + token

        return x

class STAttentionBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, inter_channels, num_subset=2, num_node=27, num_frame=400,
        kernel_size=1, stride=1, t_kernel=3,
    ):
        
        super(STAttentionBlock, self).__init__()

        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_subset = num_subset

        pad = int((kernel_size - 1) / 2)

        atts = torch.zeros((1, num_subset, num_node, num_node))
        self.register_buffer('atts', atts)
        self.pes = PositionalEncoding(in_channels, num_node, num_frame)
        self.ff_nets = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        self.in_nets = nn.Conv2d(in_channels, 2 * num_subset * inter_channels, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_subset, 1, 1), requires_grad=True)
        self.attention0s = nn.Parameter(torch.ones(1, num_subset, num_node, num_node) / num_node, requires_grad=True)

        self.out_nets = nn.Sequential(
            nn.Conv2d(in_channels * num_subset, out_channels, 1, bias=True),
            nn.BatchNorm2d(out_channels),
        )

        self.out_nett = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (t_kernel, 1), padding=(int(t_kernel / 2), 0), bias=True, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
        )

        if in_channels != out_channels or stride != 1:
            self.downs1 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )

            self.downs2 = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=True),
                nn.BatchNorm2d(out_channels),
            )
            self.downt2 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, (kernel_size, 1), (stride, 1), padding=(pad, 0), bias=True),
                nn.BatchNorm2d(out_channels),
            )

        else:
            self.downs1 = lambda x: x
            self.downs2 = lambda x: x
            self.downt2 = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        N, C, T, V = x.size()
        attention = self.atts
        y = self.pes(x)

        q, k = torch.chunk(self.in_nets(y).view(N, 2 * self.num_subset, self.inter_channels, T, V), 2, dim=1)  # nctv -> n num_subset c'tv [4,16,t,v]
        attention = attention + self.tan(torch.einsum('nsctu,nsctv->nsuv', [q, k]) / (self.inter_channels * T)) * self.alphas
        attention = attention + self.attention0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        
        y = torch.einsum('nctu,nsuv->nsctv', [x, attention]).contiguous() \
            .view(N, self.num_subset * self.in_channels, T, V)
        
        y = self.out_nets(y)  # nctv
        y = self.relu(self.downs1(x) + y)
        y = self.ff_nets(y)
        y = self.relu(self.downs2(x) + y)

        z = self.out_nett(y)
        z = self.relu(self.downt2(y) + z)

        return z


class DSTA(nn.Module):
    def __init__(
        self, cfg=None, args=None, 
        num_frame=400, num_subset=6, dropout=0.1, num_channel=2
    ):
        super(DSTA, self).__init__()

        self.cfg = cfg
        self.args = args
        config = self.cfg['net']
        self.out_channels = config[-1][1]
        in_channels = config[0][0]
        self.num_frame = num_frame
        self.device = args.device

        param = {'num_subset': num_subset,}

        self.left_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.right_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.body_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )
        self.face_input_map = nn.Sequential(
            nn.Conv2d(num_channel, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
        )

        self.face_graph_layers = nn.ModuleList()
        self.left_graph_layers = nn.ModuleList()
        self.right_graph_layers = nn.ModuleList()
        self.body_graph_layers = nn.ModuleList()

        for in_channels, out_channels, inter_channels, t_kernel, stride in config:
            self.face_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,num_node=26,
                    num_frame=num_frame, **param
                ))
            num_frame = int(num_frame / stride + 0.5)

        num_frame = self.num_frame
        for in_channels, out_channels, inter_channels, t_kernel, stride in config:
            self.left_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels, stride=stride, num_node=27,
                    t_kernel=t_kernel, num_frame=num_frame, **param
                ))
            num_frame = int(num_frame / stride + 0.5)

        num_frame = self.num_frame
        for in_channels, out_channels, inter_channels, t_kernel, stride in config:
            self.right_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels, stride=stride, t_kernel=t_kernel,
                    num_frame=num_frame, **param
                ))
            num_frame = int(num_frame / stride + 0.5)

        num_frame = self.num_frame
        for in_channels, out_channels, inter_channels, t_kernel, stride in config:
            self.body_graph_layers.append(
                STAttentionBlock(
                    in_channels, out_channels, inter_channels, stride=stride, num_node=79,
                    t_kernel=t_kernel, num_frame=num_frame, **param
                ))
            num_frame = int(num_frame / stride + 0.5)
            
        self.drop_out = nn.Dropout(dropout)


    def forward(self,src_input):
        x = src_input['keypoint'].to(self.device)
        N, C, T, V = x.shape
        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        left = self.left_input_map(x[:, :, :, self.cfg['left']])
        right = self.right_input_map(x[:, :, :, self.cfg['right']])
        face = self.face_input_map(x[:, :, :, self.cfg['face']])
        body = self.body_input_map(x[:, :, :, self.cfg['body']])

        for i, m in enumerate(self.face_graph_layers):
            face = m(face)
        for i, m in enumerate(self.left_graph_layers):
            left = m(left)
        for i, m in enumerate(self.right_graph_layers):
            right = m(right)
        for i, m in enumerate(self.body_graph_layers):
            body = m(body)  # [B,256,T/4,N] -> [B,256]

        left = left.permute(0, 2, 1, 3).contiguous()
        right = right.permute(0, 2, 1, 3).contiguous()
        face = face.permute(0, 2, 1, 3).contiguous()
        body = body.permute(0, 2, 1, 3).contiguous()
        body = body.mean(3)
        face = face.mean(3)
        left = left.mean(3)
        right = right.mean(3)

        output = torch.cat([left, face, right, body], dim=-1)
        left_output = torch.cat([left, face], dim=-1)
        right_output = torch.cat([right,  face], dim=-1)

        return output, left_output, right_output, body


class Recognition(nn.Module):
    def __init__(self, cfg, args):
        super(Recognition, self).__init__()
        self.cfg = cfg
        self.args = args
        self.device = args.device
        self.input_type = cfg['input_type']
        self.hidden_size = cfg['hidden_size']
        self.gloss_tokenizer = GlossTokenizer_S2G(cfg['GlossTokenizer'])
    
        self.visual_backbone_keypoint = DSTA(cfg=self.cfg['DSTA-Net'], num_channel=3, args=args)
        
        self.fuse_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['fuse_visual_head'])
        self.pre_fuse_proj = nn.Linear(cfg['fuse_visual_head']['input_size'], self.hidden_size)
        
        self.body_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['body_visual_head'])
        self.pre_body_proj = nn.Linear(cfg['body_visual_head']['input_size'], self.hidden_size)

        self.left_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['left_visual_head'])
        self.pre_left_proj = nn.Linear(cfg['left_visual_head']['input_size'], self.hidden_size)

        self.right_visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['right_visual_head'])
        self.pre_right_proj = nn.Linear(cfg['right_visual_head']['input_size'], self.hidden_size)

        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        self.lang_tokens = nn.ModuleDict({
            dataset: LanguageToken(hidden_size=self.hidden_size) for dataset in args.datasets
        })

        self.slr_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum')
        self.kld = torch.nn.KLDivLoss(reduction="batchmean")
        self.sal = SequenceAlignmentLoss()

    def compute_recognition_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.slr_loss(
            log_probs = gloss_probabilities_log.permute(1,0,2), #T,N,C
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss

    def decode(self, gloss_logits, beam_size, input_lengths):
        gloss_logits = gloss_logits.permute(1, 0, 2) #T,B,V  [10,1,1124]
        gloss_logits = gloss_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
            axis=-1,
        )
        decoded_gloss_sequences = ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size
        )
        return decoded_gloss_sequences

    def contrastive_loss(self, visual_embs, text_embs):
        image_embeds = visual_embs.mean(1)  # global pooling
        
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embs, dim=-1)

        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale

        return clip_loss(logits_per_text)

    def process_pre_head(self, x, proj, datasets):
        res = []

        x = proj(x)
        for i, dataset in enumerate(datasets):
            res.append(self.lang_tokens[dataset](x[i]))

        return torch.stack(res, dim=0)

    def forward(self, src_input):
        head_outputs = {}
        fuse, left_output, right_output, body = self.visual_backbone_keypoint(src_input)

        fuse = self.process_pre_head(fuse, self.pre_fuse_proj, datasets=src_input['datasets'])
        body = self.process_pre_head(body, self.pre_body_proj, datasets=src_input['datasets'])
        left_output = self.process_pre_head(left_output, self.pre_left_proj, datasets=src_input['datasets'])
        right_output = self.process_pre_head(right_output, self.pre_right_proj, datasets=src_input['datasets'])

        body_head = self.body_visual_head(
            x=body, mask=src_input['mask'].to(self.device),
            valid_len_in=src_input['new_src_lengths'].to(self.device)
        )
        fuse_head = self.fuse_visual_head(
            x=fuse, mask=src_input['mask'].to(self.device),
            valid_len_in=src_input['new_src_lengths'].to(self.device)
        )
        left_head = self.left_visual_head(
            x=left_output, mask=src_input['mask'].to(self.device),
            valid_len_in=src_input['new_src_lengths'].to(self.device)
        )
        right_head = self.right_visual_head(
            x=right_output, mask=src_input['mask'].to(self.device),
            valid_len_in=src_input['new_src_lengths'].to(self.device)
        )

        head_outputs = {
            'ensemble_last_gloss_logits': (
                left_head['gloss_probabilities'] + right_head['gloss_probabilities'] +
                body_head['gloss_probabilities']+fuse_head['gloss_probabilities']
            ).log(),
            
            'fuse_gloss_logits': fuse_head['gloss_logits'],
            'fuse_gloss_probabilities_log': fuse_head['gloss_probabilities_log'],

            'body_gloss_logits': body_head['gloss_logits'],
            'body_gloss_probabilities_log': body_head['gloss_probabilities_log'],

            'left_gloss_logits': left_head['gloss_logits'],
            'left_gloss_probabilities_log': left_head['gloss_probabilities_log'],

            'right_gloss_logits': right_head['gloss_logits'],
            'right_gloss_probabilities_log': right_head['gloss_probabilities_log'],
        }

        head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(2)
        head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
        self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble', 'gloss_feature')
        head_outputs['gloss_feature'] = fuse_head[self.cfg['gloss_feature_ensemble']]

        outputs = {**head_outputs, 'input_lengths':src_input['new_src_lengths']}

        outputs['loss'] = 0
        for k in ['left', 'right', 'fuse', 'body']:
            outputs[f'loss'] += self.compute_recognition_loss(
                gloss_labels=src_input['gloss_input']['gloss_labels'].to(self.device),
                gloss_lengths=src_input['gloss_input']['gls_lengths'].to(self.device),
                gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                input_lengths=src_input['new_src_lengths'].to(self.device)
            )
            
        for student in ['left', 'right', 'fuse', 'body']:
            teacher_prob = outputs['ensemble_last_gloss_probabilities']
            teacher_prob = teacher_prob.detach()
            student_log_prob = outputs[f'{student}_gloss_probabilities_log']
            outputs[f'{student}_distill_loss'] = self.kld(input=student_log_prob, target=teacher_prob)
            outputs['loss'] += outputs[f'{student}_distill_loss']

        outputs['loss'] += self.contrastive_loss(body_head['gloss_feature'], src_input['gloss_embs'].to(self.device))
        outputs['loss'] += self.contrastive_loss(fuse_head['gloss_feature'], src_input['gloss_embs'].to(self.device))
        outputs['loss'] += self.contrastive_loss(right_head['gloss_feature'], src_input['gloss_embs'].to(self.device))
        outputs['loss'] += self.contrastive_loss(left_head['gloss_feature'], src_input['gloss_embs'].to(self.device))
        outputs['loss'] += body_head['aux_loss'] + fuse_head['aux_loss'] \
                         + left_head['aux_loss'] + right_head['aux_loss']

        outputs['loss'] += self.sal(fuse_head['gloss_feature'], src_input['rgb_ft'].to(self.device), src_input['rgb_lgt'].to(self.device))

        return outputs