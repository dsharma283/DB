from concern.config import Configurable, Config
from experiment import Structure, Experiment
from dbnet import process_args, DBN, run_dbnet
import os, sys, json, cv2, shutil
from tqdm import tqdm
import numpy as np
import argparse
import torch
import math


def read_bbfile(bbfile):
    bblist = []
    if os.path.exists(bbfile) is False:
        return bblist

    with open(bbfile, 'r') as fp:
        lines = fp.readlines()
    if len(lines) == 0:
        return bblist

    for line in lines:
        line = line.strip('\n').split(',')[:-1]
        line = np.asarray([int(x) for x in line])
        bblist.append(line)
    return bblist


def get_rect_param(bb):
    x = float(bb[0])
    y = float(bb[1])
    width = float(bb[2] - bb[0])
    height = float(bb[7] - bb[1])
    rotation = 0.0
    text = ""
    val = {
            "x": float(x), "y": float(y),
            "width": float(width),
            "height": float(height),
            "rotation": float(rotation),
            "text" : text
            }
    key = f"{x}_{y}_{width}_{height}"
    return key, val


def get_poly_param(bb):
    key = [(int(bb[idx]), int(bb[idx + 1]))
            for idx in range(0, len(bb), 2)]
    coor = [[int(bb[idx]), int(bb[idx + 1])]
            for idx in range(0, len(bb), 2)]
    val = {
            "coordinates": coor,
            "text": ""
            }
    return str(key), val


def get_values(bb, poly):
    if poly:
        return get_poly_param(bb)
    return get_rect_param(bb)


def make_parameter_dictionary(bblist, empty_dict, poly):
    bbdict = {}
    if empty_dict:
        return bbdict

    for bb in bblist:
        k, v = get_values(bb, poly)
        bbdict[k] = v
    return bbdict


def handle_filteration(img, opath, copy):
    if os.path.exists(opath) is False:
        os.makedirs(opath)
    oflist = os.path.join(opath, "image-list.txt")
    im = img.split('/')[-1]
    with open(oflist, 'a') as of:
        of.write(f'{im}\n')
    if copy:
        shutil.copy(img, opath)


def filter_and_generate_json(opath, img, poly=False,
                             filter=False, copy=False,
                             blank=False):
    #Satisfying annotation tool requirements
    imbase = img.split('/')[-1].split('.')[0]
    itxt = os.path.join(opath, 'res_'+imbase+'.txt')
    bblist = read_bbfile(itxt)

    midfix = 'coordinates'
    if filter:
        if len(bblist):
            opath = os.path.join(opath, "filtered", "with-text")
        else:
            opath = os.path.join(opath, "filtered", "without-text")
        handle_filteration(img, opath, copy)
        jodir = os.path.join(opath, midfix, imbase)
    else:
        jodir = os.path.join(opath, midfix, imbase)

    jname = imbase + '_' + midfix + '.json'
    if os.path.exists(jodir) is False:
        os.makedirs(jodir)
    ojson = os.path.join(jodir, jname)
    if os.path.exists(ojson) is True:
        os.remove(ojson)

    bbdict = make_parameter_dictionary(bblist, blank, poly)
    with open(ojson, 'w') as of:
        json.dump(bbdict, of, indent = 4)


def should_skip(imname, basepath):
    sk = True
    if os.path.isdir(os.path.join(basepath, imname)):
        return sk

    ext = imname.split('.')[-1].lower()
    if ext == 'jpg' or ext == 'jpeg' or ext == 'png':
        prefix = imname.split('.')[0].split('_')[-1]
        if prefix  != 'res':
            sk = False
    return sk


def process_args_extended():
    parser = process_args()
    parser.add_argument('--filter', '-f', required=False, default=False, action='store_true',
                        help='Run images through dbnet and perform text-notext filteration')
    parser.add_argument('--json', '-j', required=False, default=False, action='store_true',
                        help='Also save a json along with text')
    parser.add_argument('--copy', '-c', required=False, default=False, action='store_true',
                        help='Copy the original input files to filtered directory,'
                             'otherwise only file-list is generated')
    return parser


def start_main():
    args = process_args_extended().parse_args()
    images = args.images
    opath = args.results
    if opath is None:
        opath = images

    pbar = tqdm(sorted(os.listdir(images)))
    for im in pbar:
        if should_skip(im, images):
            continue
        pbar.set_postfix_str(im)
        dbn_fails = False
        image = os.path.join(images, im)
        try:
            run_dbnet(image=image, o_path=opath,
                      poly=args.poly, viz=args.viz)
        except:
            dbn_fails = True
            print(f'dbnet failed for {im}')

        if args.json is True:
            filter_and_generate_json(opath, img=image, poly=args.poly,
                                     filter=args.filter, copy=args.copy,
                                     blank=dbn_fails) 


if __name__ == '__main__':
    start_main()
