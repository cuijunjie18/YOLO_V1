import torch
import torchvision
from utils.engine import *
import cv2

if __name__ == "__main__":

    # 加载模型与图片
    model = torch.load('results/exp1/best.pt')
    img_path = "datasets/JPEGImages/2007_000027.jpg"

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
                                    conf_thresh = 0.1,prob_thresh = 0.1,nms_thresh = 0.5,nb_classes = 20)
    

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
    os.makedirs("results/predict",exist_ok = True)
    cv2.imwrite("results/predict/1.png",img)
    print(f"Predict result save in results/predict/1.png!")
    
