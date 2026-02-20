import random
import pickle
import numpy as np
import torchvision
import torch
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader

from modules.tokenizer import GlossTokenizer_S2G

def setup_dataloaders(args, config, phase='train'):
    dataset = CombinedDataset(args.data_path, datasets=config['datasets'], phase=phase)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        collate_fn=dataset.collate_fn, 
        shuffle=phase == 'train', 
        drop_last=True
    )

    return dataloader

class CombinedDataset(Dataset.Dataset):
    def __init__(self, data_path, datasets=["CSL-Daily", "phoenix2014-T"], phase='train'):
        self.phase = phase
        self.datasets = sorted(datasets)
        self.clip_len = 400
        self.d2l_map = {'CSL-Daily': 'Chinese', 'phoenix2014-T': 'German'}
        self.d2ml_map = {'CSL-Daily': 400, 'phoenix2014-T': 300}
        self.d2wh_map = {'CSL-Daily': (512, 512), 'phoenix2014-T': (210, 260)}

        if phase == 'train': self.tmin, self.tmax = 0.5, 1.5
        else: self.tmin, self.tmax = 1, 1

        # self.tokenizers = { dataset : GlossTokenizer_S2G(
        #     {"gloss2id_file": f"data/{dataset}/gloss2ids.pkl"}) 
        #     for dataset in datasets 
        # }

        self.tokenizer = GlossTokenizer_S2G({"gloss2id_file": f"data/{'_'.join(self.datasets)}/gloss2ids.pkl"})

        self.prepare_dataset(data_path)

    def prepare_dataset(self, data_path):
        self.list, self.dataset_map, self.data = [], [], {}
        for dataset in self.datasets:
            path = f"{data_path}/{dataset}/pose.{self.phase}"
        
            with open(path, "rb") as f:
                loaded_object = pickle.load(f)
                temp_list = [key for key, value in loaded_object.items()]
                self.list.extend(temp_list)
                self.dataset_map.extend([dataset for _ in temp_list])
                self.data.update(loaded_object)

    def __len__(self):
        return len(self.data)

    def augment_preprocess_inputs(self, is_train, keypoints=None):
        if is_train == 'train':
            keypoints[:, 0, :, :] /= self.w
            keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
            keypoints[:, 1, :, :] /= self.h
            keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5
            keypoints[:, :2, :, :] = self.random_move(
                keypoints[:, :2, :, :].permute(0, 2, 3, 1).numpy()).permute(0, 3, 1, 2)
        else:
            keypoints[:, 0, :, :] /= self.w
            keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
            keypoints[:, 1, :, :] /= self.h
            keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5
        return keypoints

    def rotate_points(self, points, angle):
        center = [0, 0]

        points_centered = points - center
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        points_rotated = np.dot(points_centered, rotation_matrix.T)

        points_transformed = points_rotated + center
        return points_transformed

    def get_selected_index(self, vlen):
        if self.tmin == 1 and self.tmax == 1:
            if vlen <= self.clip_len:
                frame_index = np.arange(vlen)
                valid_len = vlen
            else:
                sequence = np.arange(vlen)
                an = (vlen - self.clip_len) // 2
                en = vlen - self.clip_len - an
                frame_index = sequence[an: -en]
                valid_len = self.clip_len

            if (valid_len % 4) != 0:
                valid_len -= (valid_len % 4)
                frame_index = frame_index[:valid_len]

            assert len(frame_index) == valid_len, (frame_index, valid_len)
            return frame_index, valid_len
        
        min_len = min(int(self.tmin * vlen), self.clip_len)
        max_len = min(self.clip_len, int(self.tmax * vlen))
        selected_len = np.random.randint(min_len, max_len + 1)
        
        if (selected_len % 4) != 0:
            selected_len += (4 - (selected_len % 4))

        if selected_len <= vlen:
            selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
        else:
            copied_index = np.random.randint(0, vlen, selected_len - vlen)
            selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))

        if selected_len <= self.clip_len:
            frame_index = selected_index
            valid_len = selected_len
        else: assert False, (vlen, selected_len, min_len, max_len)
             
        assert len(frame_index) == valid_len, (frame_index, valid_len)
        return frame_index, valid_len

    def random_move(self, data_numpy):
        # input: C,T,V,M
        
        degrees = np.random.uniform(-15, 15)
        theta = np.radians(degrees)
        p = np.random.uniform(0, 1)

        if p >= 0.5: data_numpy = self.rotate_points(data_numpy, theta)
        return torch.from_numpy(data_numpy)

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'
    
    def __getitem__(self, index):
        key = self.list[index]
        sample = self.data[key]
        gloss = sample['gloss']
        length = sample['num_frames']
        dataset = self.dataset_map[index]
        self.w, self.h = self.d2wh_map[dataset]
        
        keypoint = sample['keypoint'].permute(2, 0, 1).to(torch.float32)
        name_sample = sample['name']

        return name_sample, keypoint, gloss, length, dataset
    
    def collate_fn(self, batch):
        tgt_batch, keypoint_batch, src_length_batch, name_batch, datasets = [], [], [], [], []

        for name_sample, keypoint_sample, tgt_sample, length, dataset in batch:
            index, valid_len = self.get_selected_index(length)
            if keypoint_sample is not None:
                keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index], dim=1))

            src_length_batch.append(valid_len)
            name_batch.append(name_sample)
            tgt_batch.append(tgt_sample)
            datasets.append(dataset)
        
        max_length = max(src_length_batch)
        padded_sgn_keypoints, keypoint_feature = [], []
        
        for keypoints, len_ in zip(keypoint_batch, src_length_batch):
            if len_ < max_length:
                padding = keypoints[:, -1, :].unsqueeze(1)
                padding = torch.tile(padding, [1, max_length - len_, 1])
                padded_keypoint = torch.cat([keypoints, padding], dim=1)
                padded_sgn_keypoints.append(padded_keypoint)
            else:
                padded_sgn_keypoints.append(keypoints)

        keypoints = torch.stack(padded_sgn_keypoints, dim=0)
        keypoints = self.augment_preprocess_inputs(self.phase, keypoints)
        src_length_batch = torch.tensor(src_length_batch)
        new_src_lengths = (((src_length_batch - 1) / 2) + 1).long()
        new_src_lengths = (((new_src_lengths - 1) / 2) + 1).long()

        # gloss_lgt, gloss_ids = [], []
        # for i, tgt in enumerate(tgt_batch):
        #     tgt = [tgt]

        #     gloss_inp = self.tokenizers[datasets[i]](tgt)
        #     gloss_lgt.extend(gloss_inp["gls_lengths"])
        #     gloss_ids.extend(gloss_inp["gloss_labels"][0])

        # gloss_input = {'gls_lengths': torch.tensor(gloss_lgt), 'gloss_labels': torch.tensor(gloss_ids)}

        gloss_input = self.tokenizer(tgt_batch)

        max_len = max(new_src_lengths)
        mask = torch.zeros(new_src_lengths.shape[0], 1, max_len)
        
        for i in range(new_src_lengths.shape[0]):
            mask[i, :, :new_src_lengths[i]] = 1
        mask = mask.to(torch.bool)
        
        src_input = {}
        src_input['name'] = name_batch
        src_input['keypoint'] = keypoints
        src_input['gloss'] = tgt_batch
        src_input['mask'] = mask
        src_input['new_src_lengths'] = new_src_lengths
        src_input['gloss_input'] = gloss_input
        src_input['src_length'] = src_length_batch
        src_input['datasets'] = datasets

        return src_input



if __name__ == "__main__":
    dataset = CombinedDataset(data_path="/Users/gufran/Developer/data/sign")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

    for i, batch in enumerate(loader):
        print("Name", batch['name'])
        print("Keypoint", batch['keypoint'].shape)
        print("Gloss", batch['gloss'])
        print("Mask", batch['mask'].shape)
        print("New LGT", batch['new_src_lengths'])
        print("Gloss LGT", batch['gloss_input']['gls_lengths'])
        print("GLoss Labels", batch['gloss_input']['gloss_labels'])
        print("LGT", batch['src_length'])
        print("=="*50)

        if i == 10:break
