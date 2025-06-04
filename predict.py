import torch
import torchvision
from utils.engine import *
import cv2
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('YOLOV1 for predict', add_help = False)
    parser.add_argument('--model_path',default = "results/exp1/best.pt",type = str,
                        help = "Model path for loading.")
    parser.add_argument('--img_path',required = True,type = str,help = "Please input img path.")
    parser.add_argument('--output_dir',default = 'results/predict',type = str,
                        help = "The predicted img folder.")
    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    # 加载模型与图片
    model = torch.load(args.model_path)
    img_path = args.img_path

    # 加载class_map
    class_map = joblib.load("datasets/class_map.joblib")
    VOC_CLASS = [x for x in class_map.keys()]
    print(VOC_CLASS)

    # 与dataset相同的数据处理
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std  = [0.229, 0.224, 0.225]
        ),torchvision.transforms.Resize((448,448))]
    )
    input_tensor = preprocess(img_path,transforms)
    out_tensor = model_infer(model,input_tensor,device_idx = 7)
    boxes,labels,probs = postprocess(out_tensor,448,448,VOC_CLASS,grid_size = 7,num_bboxes = 2,
                                    conf_thresh = 0.3,prob_thresh = 0.3,nms_thresh = 0.5,nb_classes = 20)
    

    img_origin = cv2.imread(img_path)
    img = cv2.resize(img_origin,(448,448))
    for i in range(len(boxes)):
        (x_min,y_min),(x_max,y_max) = boxes[i]
        x_min = x_min.item()
        y_min = y_min.item()
        x_max = x_max.item()
        y_max = y_max.item()
        label_show = f"{labels[i]} {probs[i]:.2f}"
        cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 1)
        cv2.putText(img, label_show, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    os.makedirs(args.output_dir,exist_ok = True)
    cv2.imwrite(os.path.join(args.output_dir,"predict.png"),img)
    print(f"Predict result save in {os.path.join(args.output_dir,'predict.png')}")
    
