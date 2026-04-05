import glob
import random
import os
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class Dataset(Dataset):
    def __init__(self, dataset_path, input_shape, sequence_length, training):
        self.training = training
        self.label_index = self._extract_label_mapping(dataset_path)
        self.sequences = self._extract_sequence_paths(dataset_path, training)
        self.sequence_length = sequence_length
        self.m=0
        self.num_classes = len(self.label_index)
        self.transform = transforms.Compose(
                [
                    transforms.Resize(input_shape[-2:], Image.BICUBIC),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def _extract_label_mapping(self, dataset_path):
        #Extracts a mapping between activity name and softmax index
        with open(os.path.join(dataset_path,"classInducf11.txt")) as file:
            lines = file.read().splitlines()
        label_mapping = {}
        for line in lines:
            label, action = line.split()
            label_mapping[action] = int(label) - 1
        return label_mapping

    def _extract_sequence_paths(self, dataset_path, training=True):
        """ Extracts paths to sequences given the specified train / test split """
        fn = f"ucf11train.txt" if training else f"ucf11test.txt"
        traintest_path= os.path.join(dataset_path,fn)
        with open(traintest_path) as file:
            lines = file.read().splitlines()
        sequence_paths = []
        for line in lines:
            if 'walk_dog' not in line:
                foldern, seq_name,seq_no, label= line.split("_")[1], line.split("_")[0] + "_" + line.split("_")[1] + "_" + line.split("_")[2], line.split(" ")[0] , int(line.split(" ")[1])
            else:
                foldern,seq_name,seq_no, label = line.split("_")[1] + "_" +line.split("_")[2], line.split("_")[0]+"_"+ line.split("_")[1]+ "_" +line.split("_")[2]+ "_" +line.split("_")[3], line.split(" ")[0], int(line.split(" ")[1])
            sequence_paths += [[os.path.join(dataset_path, foldern,seq_name, seq_no),label]]
            #print("\nseq path", sequence_paths)
        return sequence_paths

    def _activity_from_path(self, path):
        """ Extracts activity name from filepath """
        return path.split("/")[-2]

    def _frame_number(self, image_path):
        """ Extracts frame number from filepath """
        return int(image_path.split("/")[-1].split(".jpg")[0])

    def _pad_to_length(self, sequence):
        """ Pads the sequence to required sequence length """
        left_pad = sequence[0]
        if self.sequence_length is not None:
            while len(sequence) < self.sequence_length:
                sequence.insert(0, left_pad)
        return sequence

    def __getitem__(self, index):
        sequence_path = self.sequences[index % len(self)]
        # Sort frame sequence based on frame number
        if sequence_path[0] == 'UCF11_updated_mpg-frames/diving/﻿v_diving_05/﻿v_diving_05_01': #some freaky unsolvable error- due to txt file first line
            sequence_path[0] = 'UCF11_updated_mpg-frames/diving/v_diving_05/v_diving_05_01'
        elif sequence_path[0] == 'UCF11_updated_mpg-frames/diving/﻿v_diving_10/﻿v_diving_10_01':
            sequence_path[0] = 'UCF11_updated_mpg-frames/diving/v_diving_10/v_diving_10_01'

        image_paths = sorted(glob.glob(f"{sequence_path[0]}/*.jpg"), key=lambda path: self._frame_number(path))

        if len(image_paths)< self.sequence_length:
            image_paths = self._pad_to_length(image_paths)
        lenimpa=len(image_paths)
        if self.training:
            # Randomly choose sample interval and start frame
            sample_interval = np.random.randint(1, len(image_paths) // self.sequence_length + 1)
            start_i = np.random.randint(0, lenimpa - sample_interval * self.sequence_length + 1)
            flip = np.random.random() < 0.5 #or False
        else:
            # Start at first frame and sample uniformly over sequence
            start_i = 0
            sample_interval = 1 if self.sequence_length is None else lenimpa // self.sequence_length
            flip = False
        # Extract frames as tensors
        image_sequence = []
        for i in range(start_i, lenimpa, sample_interval):
            if self.sequence_length is None or len(image_sequence) < self.sequence_length:
                image_tensor = self.transform(Image.open(image_paths[i]))
                if flip:
                    image_tensor = torch.flip(image_tensor, (-1,))
                image_sequence.append(image_tensor)
        image_sequence = torch.stack(image_sequence)
        target = sequence_path[1]
        del image_paths
        return image_sequence, target

    def __len__(self):
        return len(self.sequences)
