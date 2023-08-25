from concern.config import Configurable, Config
from experiment import Structure, Experiment
import os, sys, json, cv2
from tqdm import tqdm
import numpy as np
import argparse
import torch
import math


def set_args(cfg, model, res, viz, poly, imp=None):
    args = {
            'exp':cfg,
            'resume': model,
            'image_path': imp,
            'result_dir': res,
            'data': 'totaltext',
            'image_short_side': 736,
            'thresh': 0.5,
            'box_thresh': 0.6,
            'visualize': viz,
            'resize': True,
            'polygon': poly,
            'eager': False,
            'verbose': False}
    return args


def init_dbnet(o_path=f"/data/dbnet", poly=False, viz=True, pfx=False):
    cfg_path = os.path.join(sys.path[0], "experiments", "seg_detector")
    yaml_name = "totaltext_resnet50_deform_thre.yaml"

    model_name = "totaltext_resnet50"
    model = os.path.join(sys.path[0], "pretrained", model_name)

    res_dir = o_path
    cfg_file = os.path.join(sys.path[-1], cfg_path, yaml_name)
    args = set_args(cfg_file, model, res_dir, viz, poly)

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    dbn = DBN(experiment, experiment_args, cmd=args)
    dbn.init_model(load_wts=True, pfx=pfx)
    return dbn

'''
run_dbnet() function is depricated!
'''
def run_dbnet(image, o_path=f"/data/dbnet", poly=False, viz=True, pfx=True):
    cfg_path = os.path.join(sys.path[0], "experiments", "seg_detector")
    yaml_name = "totaltext_resnet50_deform_thre.yaml"

    model_name = "totaltext_resnet50"
    model = os.path.join(sys.path[0], "pretrained", model_name)

    #res_dir = os.path.join(base_path, "viz_output")
    res_dir = o_path
    cfg_file = os.path.join(sys.path[-1], cfg_path, yaml_name)
    args = set_args(cfg_file, model, res_dir, viz, poly, imp=image)

    conf = Config()
    experiment_args = conf.compile(conf.load(args['exp']))['Experiment']
    experiment_args.update(cmd=args)
    experiment = Configurable.construct_class_from_config(experiment_args)
    dbn = DBN(experiment, experiment_args, cmd=args)
    return dbn.inference(args['image_path'],
                         args['visualize'], create_model=True, pfx=pfx)


class DBN:
    def __init__(self, experiment, args, cmd=dict()):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.experiment = experiment
        experiment.load('evaluation', **args)
        self.args = cmd
        self.structure = experiment.structure
        self.model_path = self.args['resume']
        self.model = None
        self.pfx = False
        self.bbdir = self.args['result_dir']
        self.vizdir = self.args['result_dir']

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
        #print(f'DEVICE = {self.device}')

    def create_save_paths(self, pfx=False):
        if not os.path.isdir(self.args['result_dir']):
            os.makedirs(self.args['result_dir'])
        if not pfx:
            bbdir = os.path.join(self.args['result_dir'], 'bbs')
            if not os.path.isdir(bbdir):
                os.makedirs(bbdir)
            self.bbdir = bbdir

            if self.args['visualize']:
                vizdir = os.path.join(self.args['result_dir'], 'viz')
                if not os.path.isdir(vizdir):
                    os.makedirs(vizdir)
                self.vizdir = vizdir
        self.pfx = pfx

    def load_weights(self, model):
        self.init_torch_tensor()
        self.resume(model, self.model_path)
        model.eval()
        self.model = model
        return model

    def init_model(self, load_wts=False, pfx=False):
        self.init_torch_tensor()
        self.create_save_paths(pfx)
        model = self.structure.builder.build(self.device)
        if load_wts:
            model = self.load_weights(model)
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
            result_file_name = filename.split('/')[-1] + '.txt'
            if self.pfx:
                result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.bbdir, result_file_name)
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

    def inference(self, image_path, visualize=False, create_model=False):
        if create_model:
            _ = self.init_model(create_model)
        model = self.model
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = model.forward(batch, training=False)
            output = self.structure.representer.represent(batch, pred,
                                                          is_output_polygon=self.args['polygon'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                if self.pfx:
                    infix = '_res.'
                else:
                    infix = '.'
                imn = image_path.split('/')[-1].split('.')
                res_im = imn[0] + infix + imn[1].lower()
                cv2.imwrite(os.path.join(self.vizdir, res_im), vis_image)
        return output


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


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', '-i', help='Images to read', required=True)
    parser.add_argument('--results', '-r', required=False,
                        help='Output directory to save the predictions')
    parser.add_argument('--viz', '-v', required=False, default=False, action='store_true',
                        help='Generate the vizualized images in results directory')
    parser.add_argument('--poly', '-p', required=False, default=False, action='store_true',
                        help='Generate polygon bounding boxes instead of rectangles')
    parser.add_argument('--use-prefix', '-u', required=False, default=False, action='store_true',
                        help='''If specified output bounding box text file will use res_ prefix
                             followed by im-name.txt. By default, the new naming scheme will
                             be used where the output file is named as bbs/<im-name>.<im-ext>.txt.
                             If -v is used the visualized image with go to viz/<im-name>'''
                       )
    return parser


def init_tqdm(inpath):
    imgs = []
    for im in sorted(os.listdir(inpath)):
        if should_skip(im, inpath):
            continue
        imgs.append(im)
    return tqdm(sorted(imgs))


def start_main():
    args = process_args().parse_args()
    images = args.images
    opath = args.results
    if opath is None:
        opath = images

    dbn = init_dbnet(o_path=opath, poly=args.poly,
                     viz=args.viz, pfx=args.use_prefix)
    pbar = init_tqdm(images)
    for im in pbar:
        pbar.set_postfix_str(im)
        image = os.path.join(images, im)
        try:
            '''run_dbnet(image=image, o_path=opath,
                      poly=args.poly, viz=args.viz)'''
            dbn.inference(image_path=image, visualize=args.viz)
        except:
            print(f'dbnet failed for {im}')


if __name__ == '__main__':
    start_main()
