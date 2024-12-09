import os
import glob
import pandas as pd
import numpy as np
from natsort import natsorted
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, direc, mode='eval'):
        img_path = natsorted(glob.glob(os.path.join(direc, 'images', '*')))
          
        label_path = os.path.join(direc, 'masks/labels.csv')
        self.labels_df = pd.read_csv(label_path)

        self.meta_df = pd.DataFrame({"image": img_path})
        self.meta_df['label'] = self.meta_df['image'].apply(lambda x: self.get_label(os.path.basename(x)))

        if mode == 'train':
            self.transform = A.Compose([
                A.Resize(width=224, height=224),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=10, p=0.5),
                ToTensorV2()
            ])
        elif mode == 'eval':
            self.transform = A.Compose([
                A.Resize(width=224, height=224),
                ToTensorV2()
            ])

    def get_label(self, filename):
        filename_without_extension = os.path.splitext(filename)[0]
        label = self.labels_df.loc[self.labels_df['fname'] == filename_without_extension, 'cancer'].values
        return label[0] if len(label) > 0 else 0  

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        img_path = self.meta_df.iloc[idx]['image']
        img = Image.open(img_path).convert('L')
        label = self.meta_df.iloc[idx]['label']

        img = img.resize((224, 224))
        img = np.array(img)

        if self.transform:
            img = self.transform(image=img)['image']
            
        sample_input = {}
        sample_input['input'] = img
        sample_input['target'] = np.array(label, dtype=np.float32)
        return sample_input      

if __name__ == '__main__':
    train = CustomDataset('/home/pwrai/userarea/KTL_project_08_Mammo_BreastCancer_Classification/data/', 'train')
    test = CustomDataset('/home/pwrai/userarea/KTL_project_08_Mammo_BreastCancer_Classification/data/', 'eval')
    
    for sample_input in train:
        print(sample_input['input'].shape)
        print(sample_input['target'])

