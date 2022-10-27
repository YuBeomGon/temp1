import os
import cv2
import mmcv
import torch
import warnings
from argparse import ArgumentParser
from mmdet.apis import inference_detector, init_detector

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--det_config', default="configs/swin/retinanet_swin-t-p4-w7_fpn_3x_coco.py", help='Config file for detection')
    parser.add_argument('--det_checkpoint', default="pretrained/retinanet_swin_tiny_fpn_3x_coco_epoch_25.pth", help='Checkpoint file for detection')   
    
    args = parser.parse_args()
    det_model = init_detector(
        args.det_config, args.det_checkpoint)
    
    # det_model.load_state_dict(torch.load(args.det_checkpoint)["state_dict"])
    
    # only take person related weights only for removing nms of other classes
    person_weight = det_model.bbox_head.retina_cls.weight[::80,:,:,:]
    person_bias = det_model.bbox_head.retina_cls.bias[::80]
    
    print(person_weight.shape)
    print(person_bias.shape)
    
    det_model.bbox_head.retina_cls = torch.nn.Conv2d(256, 9, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    det_model.bbox_head.retina_cls.weight = torch.nn.Parameter(person_weight, requires_grad=False)
    det_model.bbox_head.retina_cls.bias = torch.nn.Parameter(person_bias)
    
    torch.save(det_model.state_dict(), str(args.det_checkpoint).split('.')[0] +'_person.pth')
    
if __name__ == '__main__':
    main()    