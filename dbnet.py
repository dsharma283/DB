from concern.config import Configurable, Config
from experiment import Structure, Experiment
import os, sys, json, cv2
import numpy as np
import argparse
import torch
import math

def run_dbnet(image, o_path=f"/data/dbnet", poly=False, viz=True):
    cfg_path = os.path.join(sys.path[0], "experiments", "seg_detector")
    yaml_name = "totaltext_resnet50_deform_thre.yaml"

    model_name = "totaltext_resnet50"
    model = os.path.join(sys.path[0], "pretrained", model_name)

    #res_dir = os.path.join(base_path, "viz_output")
    res_dir = o_path
    cfg_file = os.path.join(sys.path[-1], cfg_path, yaml_name)

    args = {'exp':cfg_file, 'resume': model, 'image_path': image,
            'result_dir': res_dir, 'data': 'totaltext',
            'image_short_side': 736, 'thresh': 0.5, 'box_thresh': 0.6,
            'visualize': viz, 'resize': False, 'polygon': poly,
            'eager': True, 'verbose': False}

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    return DBN(experiment, experiment_args, cmd=args).inference(args['image_path'], args['visualize'])


class DBN:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        self.structure = experiment.structure
        self.model_path = self.args['resume']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
        #print(f'DEVICE = {self.device}')

    def init_model(self):
        model = self.structure.builder.build(self.device)
        return model

    def resume(self, model, path):
        if not os.path.exists(path):
            print("Checkpoint not found: " + path)
            return
        print("Resuming from " + path)
        states = torch.load(
            path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        print("Resumed from " + path)

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def inference(self, image_path, visualize=False):
        self.init_torch_tensor()
        model = self.init_model()
        self.resume(model, self.model_path)
        all_matircs = {}
        model.eval()
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred,
                                                          is_output_polygon=self.args['polygon'])
            if not os.path.isdir(self.args['result_dir']):
                os.makedirs(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'],
                                         image_path.split('/')[-1].split('.')[0]+ '_res' + '.jpg'),
                            vis_image)
        return output


def should_skip(imname, basepath):
    if os.path.isdir(os.path.join(basepath, imname)):
        return True
    is_txt = imname.split('.')[-1] == 'txt'
    is_res = imname.split('.')[0].split('_')[-1] == 'res'
    return is_txt or is_res


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', '-i', help='Images to read', required=True)
    parser.add_argument('--results', '-r', required=False,
                        help='Output directory to save the predictions')
    parser.add_argument('--json', '-j', required=False, default=False, action='store_true',
                        help='Also save a json along with text')
    parser.add_argument('--viz', '-v', required=False, default=False, action='store_true',
                        help='Generate the vizualized images in results directory')
    parser.add_argument('--poly', '-p', required=False, default=False, action='store_true',
                        help='Generate polygon bounding boxes instead of rectangles')
    return parser


def read_bbfile(bbfile):
    bblist = []
    if os.path.exists(bbfile) is False:
        return bblist

    with open(bbfile, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip('\n').split(',')[:-1]
        line = np.asarray([int(x) for x in line])
        #line = convert_to_pthw(line, reshape_only=True)
        bblist.append(line)
    return bblist


def get_values(bb):
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


def make_parameter_dictionary(bblist, empty_dict):
    bbdict = {}
    if empty_dict:
        return bbdict

    for bb in bblist:
        k, v = get_values(bb)
        bbdict[k] = v
    return bbdict


def process_json(opath, im, empty_json=False):
    #Stisfying annotation tool requirements
    imbase = im.split('.')[0]
    midfix = 'coordinates'
    odir = os.path.join(opath, midfix, imbase)
    jname = imbase + '_' + midfix + '.json'
    ojson = os.path.join(odir, jname)
    itxt = os.path.join(opath, 'res_'+imbase+'.txt')

    if os.path.exists(odir) is False:
        os.makedirs(odir)
    if os.path.exists(ojson) is True:
        os.remove(ojson)

    bblist = read_bbfile(itxt)
    bbdict = make_parameter_dictionary(bblist, empty_json)
    with open(ojson, 'w') as of:
        json.dump(bbdict, of, indent = 4)


if __name__ == '__main__':
    args = process_args().parse_args()
    images = args.images
    opath = args.results
    if opath is None:
        opath = images

    for im in os.listdir(images):
        if should_skip(im, images):
            continue
        empty_json = False
        image = os.path.join(images, im)
        try:
            run_dbnet(image=image, o_path=opath,
                      poly=args.poly, viz=args.viz)
        except:
            empty_json = True
            print(f'dbnet failed for {im}')

        if args.json is True:
            if empty_json:
                print(f'wrote empty json for {im}')
            process_json(opath, im, empty_json)
