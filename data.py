from scipy.io import loadmat
import xml.etree.ElementTree as ET
import torch, torchvision
import torchvision.tv_tensors
import os

class SubsetLabels(torch.utils.data.Dataset):
    # Filter only the given labels from the dataset (for fast debugging)
    def __init__(self, ds, labels):
        self.ds = ds
        self.ix = [i for i in range(len(ds)) if ds.__getitem__(i, True) in labels]
        self.labels = labels
        self.num_classes = len(labels)

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        i = self.ix[i]
        x, m, y = self.ds[i]
        y = self.labels.index(y)
        return x, m, y

class Birds(torch.utils.data.Dataset):
    # https://www.vision.caltech.edu/datasets/cub_200_2011/
    num_classes = 200
    def __init__(self, root, fold, transform=None, crop=False):
        self.root = os.path.join(root, 'CUB_200_2011')
        files = open(os.path.join(self.root, 'images.txt'))
        split = open(os.path.join(self.root, 'train_test_split.txt'))
        bboxes = open(os.path.join(self.root, 'bounding_boxes.txt'))
        # I don't know if split=0 is test and split=1 is train but I am assuming
        # train=1 since split=0 49% and split=1 51%
        fold = int(fold == 'train')
        ix = [int(s.split()[1]) == fold for s in split]
        self.files = [f.split()[1] for f, i in zip(files, ix) if i]
        self.labels = [int(f[:f.index('.')])-1 for f in self.files]
        self.bboxes = [[int(float(v)) for v in line.split()[1:]] for line, i in zip(bboxes, ix) if i]
        self.class_names = [line.split()[1][4:-1] for line in open(os.path.join(self.root, 'classes.txt'))]
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i, only_label=False):
        fname = self.files[i]
        label = self.labels[i]
        if only_label:
            return label
        image = os.path.join(self.root, 'images', fname)
        image = torchvision.io.read_image(image, torchvision.io.ImageReadMode.RGB)
        mask = os.path.join(self.root, 'segmentations', fname[:-3] + 'png')
        mask = torchvision.io.read_image(mask, torchvision.io.ImageReadMode.GRAY)
        if self.crop:
            bbox = self.bboxes[i]  # their bbox=(x, y, w, h)
            image = image[:, bbox[1]:bbox[1]+bbox[3]+1, bbox[0]:bbox[0]+bbox[2]+1]
            mask = mask[:, bbox[1]:bbox[1]+bbox[3]+1, bbox[0]:bbox[0]+bbox[2]+1]
        mask = torchvision.tv_tensors.Mask(mask)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

# for debugging
def Birds2(root, fold, transform, crop=False):
    return SubsetLabels(Birds(root, fold, transform, crop), range(2))
def Birds10(root, fold, transform, crop=False):
    return SubsetLabels(Birds(root, fold, transform, crop), range(10))

class StanfordCars:
    # https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset
    num_classes = 196
    def __init__(self, root, split, transform=None, crop=False):
        root = os.path.join(root, 'stanford_cars')
        dname = f'cars_{split}'
        fname = 'devkit/cars_train_annos.mat' if split == 'train' else 'cars_test_annos_withlabels.mat'
        data = loadmat(os.path.join(root, fname), simplify_cells=True)
        self.class_names = list(loadmat(os.path.join(root, 'devkit', 'cars_meta.mat'), simplify_cells=True)['class_names'])
        self.data = data['annotations']
        self.root = os.path.join(root, dname)
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        d = self.data[i]
        image = torchvision.io.read_image(os.path.join(self.root, d['fname']), torchvision.io.ImageReadMode.RGB)
        bbox = (d['bbox_x1'], d['bbox_y1'], d['bbox_x2'], d['bbox_y2'])
        label = d['class']-1
        if self.crop:
            image = image[:, bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            mask = torch.ones(1, bbox[3]-bbox[1]+1, bbox[2]-bbox[0]+1, dtype=bool)
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=bool)
            mask[0, bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1] = True
        mask = torchvision.tv_tensors.Mask(mask)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

class StanfordDogs:
    # http://vision.stanford.edu/aditya86/ImageNetDogs/
    num_classes = 120
    def __init__(self, root, split, transform=None, crop=False):
        self.root = os.path.join(root, 'stanford_dogs')
        l = loadmat(os.path.join(self.root, f'{split}_list.mat'), simplify_cells=True)
        self.files = l['file_list']
        self.labels = l['labels'].astype(int)-1
        self.class_names = ['-'.join(species.split('-')[1:]) for species in sorted(os.listdir(os.path.join(root, 'stanford_dogs', 'Images')))]
        self.transform = transform
        self.crop = crop

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]
        label = self.labels[i]
        image = torchvision.io.read_image(os.path.join(self.root, 'Images', fname), torchvision.io.ImageReadMode.RGB)
        ann = ET.parse(os.path.join(self.root, 'Annotation', fname[:-4]))
        bboxes = [{c.tag: int(c.text) for c in bbox} for bbox in ann.findall('.//bndbox')]
        if self.crop:
            bbox = (min(bb['xmin'] for bb in bboxes), min(bb['ymin'] for bb in bboxes),
                    max(bb['xmax'] for bb in bboxes), max(bb['ymax'] for bb in bboxes))
            image = image[:, bbox[1]:bbox[3]+1, bbox[0]:bbox[2]+1]
            mask = torch.ones(1, bbox[3]-bbox[1]+1, bbox[2]-bbox[0]+1, dtype=bool)
        else:
            mask = torch.zeros(1, image.shape[1], image.shape[2], dtype=bool)
            for bbox in bboxes:
                mask[0, bbox['ymin']:bbox['ymax']+1, bbox['xmin']:bbox['xmax']+1] = True
        mask = torchvision.tv_tensors.Mask(mask)
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask, label

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('--crop', action='store_true')
    args = parser.parse_args()
    ds = globals()[args.dataset]('/data/toys', 'train', None, args.crop)
    #from tqdm import tqdm
    #for k, n in enumerate(torch.bincount(torch.tensor([y for _, _, y in tqdm(ds)]))):
    #    print(k, n)
    import matplotlib.pyplot as plt
    for i, (image, mask, label) in enumerate(ds):
        plt.subplot(1, 2, 1)
        plt.imshow(image.permute(1, 2, 0))
        plt.subplot(1, 2, 2)
        plt.imshow(mask[0])
        if hasattr(ds, 'class_names'):
            label = f'{ds.class_names[label]} {label}'
        plt.suptitle(str(label))
        plt.show()
