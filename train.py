import argparse
parser = argparse.ArgumentParser()
parser.add_argument('output')
parser.add_argument('dataset')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batchsize', type=int, default=8)
args = parser.parse_args()

import torch
from torchvision.transforms import v2
from time import time
from tqdm import tqdm
import data, models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

############################# DATA #############################

class TransformProb(torch.nn.Module):
    def __init__(self, transform, prob):
        super().__init__()
        self.transform = transform
        self.prob = prob
    def forward(self, *x):
        if torch.rand(()) < self.prob:
            x = self.transform(*x)
        return x

train_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.RandomHorizontalFlip(),
    TransformProb(v2.RandomRotation(15), 0.33),
    TransformProb(v2.RandomAffine(0, shear=10), 0.33),
    v2.RandomPerspective(0.2, 0.33),
    v2.ColorJitter(0.2, 0.2),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToDtype(torch.float32, True),
    v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
ds = getattr(data, args.dataset)('/data/toys', 'train', train_transforms)
ds_noaug = getattr(data, args.dataset)('/data/toys', 'train', test_transforms)
tr = torch.utils.data.DataLoader(ds, args.batchsize, True, num_workers=4, pin_memory=True)

############################# MODEL #############################

backbone = models.Backbone()
model = models.ProtoPNet(backbone, ds.num_classes)
opt = torch.optim.AdamW([
    {'params': backbone.parameters(), 'lr': args.lr/10},
    {'params': list(model.features.parameters()) + list(model.prototype_layer.parameters())}
], args.lr)
late_opt = torch.optim.AdamW(model.fc_layer.parameters(), args.lr)
model.to(device)

############################# LOOP #############################

model.train()
for epoch in range(args.epochs):
    tic = time()
    avg_losses = {}
    avg_metrics = {}
    for x, masks, y in tqdm(tr, 'stage1'):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.nn.functional.cross_entropy(pred['class'], y)
        stage1_loss = baseline_protopnet.stage1_loss(model, pred['features'], y)
        loss += stage1_loss
        avg_losses['stage1'] = avg_losses.get('stage1', 0) + float(stage1_loss)/len(tr)
        opt.zero_grad()
        loss.backward(retain_graph=args.adversarial)
        avg_losses['loss'] = avg_losses.get('loss', 0) + float(loss)/len(tr)
        opt.step()
        avg_metrics['acc'] = avg_metrics.get('acc', 0) + (y == pred['class'].argmax(1)).float().mean()/len(tr)
    if (epoch+1) % 10 == 0:
        # protopnet has two more stages: in the paper they do this after a few epochs
        model.backbone.eval()
        # (stage2) projection of prototypes
        # we use a deterministic dataloader so that we can access images by index
        # later on
        dl = torch.utils.data.DataLoader(ds_noaug, args.batchsize, pin_memory=True)
        features_per_class = [[] for _ in range(ds.num_classes)]
        indices_per_class = [[] for _ in range(ds.num_classes)]
        for i, (x, _, y) in enumerate(tqdm(dl, 'stage2')):
            x = x.to(device)
            with torch.no_grad():
                z = model.features(model.backbone(x)[-1])
                z_shape = z.shape
                z = torch.flatten(z, 2).permute(0, 2, 1)
            xscale = x.shape[3] // z_shape[3]
            yscale = x.shape[2] // z_shape[2]
            for j, (k, zk) in enumerate(zip(y, z)):
                features_per_class[k].append(zk)
                l = [(i*args.batchsize+j, x*xscale, y*yscale, (x+1)*xscale-1, (y+1)*yscale-1) for y in range(z_shape[2]) for x in range(z_shape[3])]
                indices_per_class[k] += l
        model.patch_prototypes = [[] for _ in range(ds.num_classes)]
        model.image_prototypes = [[] for _ in range(ds.num_classes)]
        model.illustrative_prototypes = [[] for _ in range(ds.num_classes)]
        for k, zk in enumerate(features_per_class):
            zk = torch.cat(zk)
            assert len(zk) == len(indices_per_class[k])
            pk = model.prototype_layer.prototypes[0, k]
            distances = torch.cdist(zk[None], pk[None])[0]
            # I was getting the same feature to be projected to different prototypes, so
            # the prototypes were diverging to the same features.
            # find the minimum without repetitions
            ix = torch.zeros(len(pk), dtype=torch.int64)
            for _ in range(len(pk)):
                i = distances.argmin()
                # features zk are the rows, prototypes pk are the columns
                row, col = i // distances.shape[1], i % distances.shape[1]
                ix[col] = row
                distances[row] = torch.inf
                distances[:, col] = torch.inf
            ix = torch.argmin(distances, 0)
            with torch.no_grad():  # projection
                model.prototype_layer.prototypes[0, k] = zk[ix]
            for i in ix:
                i, x1, y1, x2, y2 = indices_per_class[k][i]
                image = ds_noaug[i][0]
                patch = image[:, y1:y2, x1:x2]
                model.patch_prototypes[k].append(patch)
                model.image_prototypes[k].append(image)
                image = image.clone()
                color = torch.tensor((1, 0, 0))[:, None, None]  # red
                image[:, y1:y1+2, x1:x2] = color
                image[:, y2:y2+2, x1:x2] = color
                image[:, y1:y2, x1:x1+2] = color
                image[:, y1:y2, x2:x2+2] = color
                model.illustrative_prototypes[k].append(image)
        # (stage3) convex optimization of last layer
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc_layer.parameters():
            param.requires_grad = True
        for _ in tqdm(range(20), 'stage3'):
            for x, _, y in tr:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                stage3_loss = torch.nn.functional.cross_entropy(pred['class'], y)
                stage3_loss += 1e-4*baseline_protopnet.stage3_loss(model)
                late_opt.zero_grad()
                stage3_loss.backward()
                late_opt.step()
                avg_losses['stage3'] = avg_losses.get('stage3', 0) + float(stage3_loss)/len(tr)
        for param in model.parameters():
            param.requires_grad = True
        model.backbone.train()
    toc = time()
    print(f'Epoch {epoch+1}/{args.epochs} - {toc-tic:.0f}s - Avg loss: {" - ".join(f'{k}={v}' for k, v in avg_losses.items())} - Avg metrics: {" - ".join(f'{k}={v}' for k, v in avg_metrics.items())}')

torch.save(model.cpu(), args.output)
