import cv2
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import argparse
from network import CSFNet
from args import ArgumentParser

def parse_args():
    parser = ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.set_common_args()
    args = parser.parse_args()
    return args

label_to_color = {
        0: (128, 64,128),          
        1: (244, 35,232),      
        2: ( 70, 70, 70),       
        3: (102,102,156),     
        4: (190,153,153),       
        5: (153,153,153),     
        6: (250,170, 30),      
        7: (220,220,  0),  
        8: (107,142, 35),
        9: (152,251,152),          
        10: ( 70,130,180),      
        11: (220, 20, 60),       
        12: (255,  0,  0),     
        13: (  0,  0,142),       
        14: (  0,  0, 70),     
        15: (  0, 60,100),      
        16: (  0, 80,100),  
        17: (  0,  0,230),
        18: (119, 11, 32)   
    }      

def inference(model_path ,image_path , projected_lidar_path):
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        H , W,_ = image.shape
        
        dep = np.load(projected_lidar_path).astype('float32')

        depth_show = dep.copy()
        depth_show[depth_show == 0] = np.nan
        
        img_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        dep_t = torch.from_numpy(dep)
                   
        valid = dep_t > 0
        dep_norm = torch.zeros_like(dep_t)
        validity = torch.zeros_like(dep_t)
        
        dep_norm[valid] = (dep_t[valid] - 12.144798) / 10.222658
        validity[valid] = 1.0
                    
        depth_input = torch.stack([dep_norm, validity], dim=0)
        
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        
        img_t = (img_t - mean) / std

        img_t = torch.unsqueeze(img_t , dim = 0)
        depth_input = torch.unsqueeze(depth_input , dim = 0)
    
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network = CSFNet("CSFNet-1", False, None  , 19)
        network = nn.DataParallel(network ,device_ids=[0]).to(device)

        network.load_state_dict(torch.load(model_path))

        network.eval()
        with torch.no_grad(): 
            img_t = img_t.to(device)
            depth_input = depth_input.to(device)

            outputs = network(img_t,depth_input)
            
            pred_label_imgs = np.argmax(outputs[0].detach().cpu().numpy(), axis=0) 
            
            new_label = np.zeros((H , W , 3) , dtype = np.uint8)

            palette = np.zeros((256, 3), dtype=np.uint8)
            for k, v in label_to_color.items():
                palette[k] = v
            
            new_label = palette[pred_label_imgs]
            
            cmap = plt.cm.turbo.copy()
            cmap.set_bad(color="white")
            
            plt.figure(figsize=(10, 7), layout="tight")
            plt.subplots_adjust(hspace=0.08)
            
            # RGB
            plt.subplot(3, 1, 1)
            plt.imshow(image)
            plt.title("RGB Image", y=0.99)
            plt.axis("off")
                    
            # Depth
            plt.subplot(3, 1, 2)
            plt.imshow(depth_show, cmap=cmap) 
            plt.title("LiDAR Depth", y=0.99)
            plt.axis("off")
            
            # Prediction
            plt.subplot(3, 1, 3)
            plt.imshow(new_label)
            plt.title("Prediction", y=0.99)
            plt.axis("off")
            
            output_dir = "Run"
            filename = "All_" + image_path.split("/")[-1]
            filename2 = "pred_" + image_path.split("/")[-1]
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
    
            save_path = os.path.join(output_dir, f"{filename}")
            save_path2 = os.path.join(output_dir, f"{filename2}")
            print(f'The result is saved in the "Run" folder')
            
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            new_label = cv2.cvtColor(new_label , cv2.COLOR_BGR2RGB)
            cv2.imwrite(save_path2 , new_label)
            plt.close()

if __name__ == '__main__':
    args = parse_args()
    inference(args.weight_path , args.single_img_path , args.single_lidar_path)
