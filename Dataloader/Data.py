import torch
import os
import numpy as np
import cv2
import torchvision
import random

def Data(mode = "valid" , img_path = None , lidar_path = None , label_path = None, test_seqs = ["seq_0007"], crop_size = (376, 1408)):
           if mode == "train":
                   trans = torchvision.transforms.Compose([
                                   torchvision.transforms.RandomHorizontalFlip()
                                   , torchvision.transforms.RandomCrop(crop_size, pad_if_needed=True)])

                   return Data_KITTI_360(img_path, lidar_path , label_path, transform = trans, mode = mode)
           else:
                   return Data_KITTI_360(img_path, lidar_path , label_path, mode = mode , test_seqs = test_seqs)
           
           

class Data_KITTI_360 (torch.utils.data.Dataset):
        def __init__(self, rgb_path, lidar_path , label_path, test_seqs:list= ["seq_0007"] , transform = None, mode="train"):
            self.transform = transform
            self.mode = mode

            self.trans_img = torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
            self.trans_jit = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness=(0.9, 1.1))])
            self.trans_totensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

            self.examples = []

            if mode == "train":
                folder = os.listdir(label_path)
                for item in test_seqs:
                    folder.remove(item)
            else:
                folder = test_seqs

                
            for folders in folder:
                folders_path = os.path.join(label_path, folders)
                lab_name = os.listdir(folders_path)
                for labs in lab_name:
                    label_dir = os.path.join(folders_path, labs)
                    img_dir = os.path.join(rgb_path, os.path.join(folders, labs))
                    depth_name = labs.split(".")[0] + ".npy"
                    depth_dir = os.path.join(lidar_path, os.path.join(folders, depth_name))
       

                    sample = {}
                    sample["image_path"] = img_dir
                    sample["depth_path"] = depth_dir
                    sample["label_path"] = label_dir
                    self.examples.append(sample)

    
        def __getitem__(self, index):
            img_path = self.examples[index]["image_path"]
            dep_path = self.examples[index]["depth_path"]
            lab_path = self.examples[index]["label_path"]

            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dep = np.load(dep_path)
            dep = dep.astype('float32')
            lab = cv2.imread(lab_path, cv2.IMREAD_GRAYSCALE)

            scales = [0.75, 1, 1.25, 1.5, 1.75]
            if self.mode == "train":
                S = scales[random.randrange(0, 5)]
                width = int(img.shape[1] * S)
                height = int(img.shape[0] * S)

                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
                dep = cv2.resize(dep, (width, height), interpolation=cv2.INTER_NEAREST)
                lab = cv2.resize(lab, (width, height), interpolation=cv2.INTER_NEAREST)


            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            dep_t = torch.from_numpy(dep).float()
            lab_t = torch.from_numpy(lab).float()

            valid = dep_t > 0
            dep_norm = torch.zeros_like(dep_t)
            validity = torch.zeros_like(dep_t)

            dep_norm[valid] = (dep_t[valid] - 12.144798) / 10.222658
            validity[valid] = 1.0
            
            depth_input = torch.stack([dep_norm, validity], dim=0)

            if self.mode == "train":
                lab_t = lab_t.unsqueeze(0)  
                concat_image = torch.cat((img_t, depth_input, lab_t + 1), dim=0)
                concat_image = self.transform(concat_image)
                
                img_t = concat_image[0:3]
                img_t = self.trans_jit(img_t)
                depth_input = concat_image[3:5]
                lab_t = concat_image[5] - 1

            lab_t[lab_t == 255] = -1
            img_t = self.trans_img(img_t)
    
            return (img_t, depth_input, lab_t)

        def __len__(self):
            return len(self.examples)
