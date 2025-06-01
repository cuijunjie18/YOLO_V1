import xml.etree.ElementTree as ET
import glob
import os
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('label from xml to yolo', add_help = False)
    parser.add_argument('--xml_folder',default = "VOC2012/Annotations",
                        type = str,help = "Path to raw xml-files.")
    parser.add_argument('--save_folder',default = "datasets",
                        type = str,help = "Path to save file converted.")
    return parser

def xml2yolo(args):
    """将数据集xml标签转化为单个train/test.txt的yolo格式标签"""
    
    # 解析命令行参数
    xml_folder,save_folder = args.xml_folder,args.save_folder

    # 定义类别字典，用于class2idx
    class_map = {}
    class_nums = 0

    # xml文件所处路径
    Annotations_path = xml_folder + "/*.xml"
    print(f"read xml-files from {Annotations_path}")

    # 划分数据集为训练与测试集
    f_train = open(os.path.join(save_folder,"train.txt"),"w",encoding = 'utf-8') # w为重写模式
    f_test = open(os.path.join(save_folder,"test.txt"),"w",encoding = 'utf-8')

    # 设置随机种子去划分
    import random
    all_xml_files = glob.glob(Annotations_path)
    random.seed(666)
    shuffled_data = random.sample(all_xml_files,len(all_xml_files)) # 打乱但不修改原数据
    split_idx = int(len(all_xml_files) * 0.8)
    train_xml_files = shuffled_data[:split_idx]
    test_xml_files = shuffled_data[split_idx:]

    # 处理训练集
    for xmlname in train_xml_files:
        tree = ET.parse(xmlname)
        root = tree.getroot()
        img_name = root.find("filename").text
        f_train.write(f"{img_name} ")
        for obj in root.iter("object"):
            obj_name = obj.find("name").text
            if obj_name not in class_map:
                class_map[obj_name] = class_nums
                class_nums += 1
            x_min = obj.find("bndbox/xmin").text
            x_max = obj.find("bndbox/xmax").text
            y_min = obj.find("bndbox/ymin").text
            y_max = obj.find("bndbox/ymax").text
            f_train.write(f"{x_min} {y_min} {x_max} {y_min} {class_map[obj_name]} ") # 按特定顺序写入文件
        f_train.write("\n")

    # 处理测试集
    for xmlname in test_xml_files:
        tree = ET.parse(xmlname)
        root = tree.getroot()
        img_name = root.find("filename").text
        f_test.write(f"{img_name} ")
        for obj in root.iter("object"):
            obj_name = obj.find("name").text
            if obj_name not in class_map:
                class_map[obj_name] = class_nums
                class_nums += 1
            x_min = obj.find("bndbox/xmin").text
            x_max = obj.find("bndbox/xmax").text
            y_min = obj.find("bndbox/ymin").text
            y_max = obj.find("bndbox/ymax").text
            f_test.write(f"{x_min} {y_min} {x_max} {y_min} {class_map[obj_name]} ") # 按特定顺序写入文件
        f_test.write("\n")

    # 关闭文件
    f_train.close()
    f_test.close()
    print(f"Converted file save to {save_folder}/")

    # 检测类别数量是否符合
    # print(class_map)
    print(f"Dataset have {class_nums} classes!") # 预期20类

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    xml2yolo(args)
    