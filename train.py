import torch
import torchvision
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('label from xml to yolo', add_help = False)
    parser.add_argument('--xml_folder',default = "VOC2012/Annotations",
                        type = str,help = "Path to raw xml-files.")
    parser.add_argument('--save_folder',default = "datasets",
                        type = str,help = "Path to save file converted.")
    return parser

if __name__ == "__main__":
    exit(0)