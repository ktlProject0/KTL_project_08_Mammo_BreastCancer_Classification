import os
import argparse
import numpy as np
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from model import MammographyModel

if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--data_path', type=str, default='./data/test/images/4564_799944340.png', help="data path")
    parser.add_argument('--n_classes', type=int, default=1, help="num of classes")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=42')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path for save best model')
    opt = parser.parse_args()
    
    if not os.path.isdir(opt.model_save_path):
        raise Exception("checkpoints not found, please run train.py first")

    os.makedirs("Visualize_result", exist_ok=True)
    
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(opt.seed)
    
    device = torch.device("cuda:1" if opt.cuda else "cpu")
    
    model = MammographyModel(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    transform = A.Compose([
                A.Resize(width=224, height=224, interpolation=cv2.INTER_LINEAR),
                ToTensorV2()
            ])

    image = cv2.imread(opt.data_path, cv2.IMREAD_GRAYSCALE)

    transformed = transform(image=image)
    sample_input = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_logit = model(sample_input.float())
        pred_prob = torch.sigmoid(pred_logit)
        pred_class = (pred_prob > 0.5).float()

    print("Predicted class:", pred_class.squeeze().item())
    print("pred_prob:",pred_prob)
    print("pred_logit:",pred_logit)

    Panorama = sample_input.squeeze().detach().cpu().numpy()
    Panorama_visualize = (Panorama.squeeze() * 255).astype(np.uint8)

    label_text = f'Predicted Class: {int(pred_class.squeeze().item())}'
    cv2.putText(Panorama_visualize, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(f"Visualize_result/{os.path.basename(opt.data_path)}", Panorama_visualize)