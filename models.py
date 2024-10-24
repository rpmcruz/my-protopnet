# https://arxiv.org/abs/1806.10574
import torch

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(weights='DEFAULT')

    def forward(self, x):
        layers = [self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4]
        x = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x))))
        ret = []
        for layer in layers:
            x = layer(x)
            ret.append(x)
        return ret

class PrototypeLayer(torch.nn.Module):
    def __init__(self, num_classes, num_prototypes_per_class, prototype_dim):
        super().__init__()
        self.prototypes = torch.nn.Parameter(
            torch.randn(1, num_classes, num_prototypes_per_class, prototype_dim))

    def forward(self, z):
        # Given a convolutional output z=f(x), the j-th prototype unit gpj in the prototype
        # layer gp computes the squared L2 distances between the j-th prototype pj and all
        # patches of z that have the same shape as pj
        # cdist computes the batch-distance between shapes (B, P, M) and (B, R, M)
        z_shape = z.shape
        z = torch.flatten(z, 2).permute(0, 2, 1)  # (B, H*W, F)
        p = torch.flatten(self.prototypes, 1, 2)  # (1, P)
        distances = torch.cdist(z, p)  # (B, H*W, P)
        #distances = distances.reshape(  # (B, P, H, W)
        #    z_shape[0], z_shape[2], z_shape[3], p.shape[1]).permute(0, 3, 1, 2)

        # (...) and inverts the distances into similarity scores.
        # This activation map preserves the spatial relation of the convolutional output, and
        # can be upsampled to the size of the input image to produce a heat map that identifies
        # which part of the input image is most similar to the learned prototype.
        # (we are not upsampling; that is only useful for visualization)
        heatmap = torch.log((distances+1)/(distances+1e-7))  # (B, H*W, P)

        # The activation map of similarity scores produced by each prototype unit gpj is
        # then reduced using global max pooling to a single similarity score, which can be
        # understood as how strongly a prototypical part is present in some patch of the
        # input image.
        similarity = heatmap.amax(1)  # (B, P)
        return similarity

class ProtoPNet(torch.nn.Module):
    def __init__(self, backbone, num_classes, num_prototypes_per_class=10, prototype_dim=256):
        super().__init__()
        self.backbone = backbone
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(2048, 2048//4, 1), torch.nn.ReLU(),
            torch.nn.Conv2d(2048//4, prototype_dim, 1), torch.nn.Sigmoid()
        )
        self.prototype_layer = PrototypeLayer(num_classes, num_prototypes_per_class, prototype_dim)
        self.fc_layer = torch.nn.Linear(num_classes*num_prototypes_per_class, num_classes, bias=False)

    def forward(self, x):
        features = self.backbone(x)[-1]
        # (backbone) followed by two additional 1x1 convolutional layers
        # The number of output channels in each of the two additional convolutional layers
        # is chosen to be the same as the number of channels in a prototype.
        # for ResNet-34, we used 256 as the number of channels in a prototype
        features = self.features(features)
        similarity = self.prototype_layer(features)
        # the m similarity scores produced by the prototype layer gp are multiplied by the
        # weight matrix wh in the fully connected layer h to produce the output logits,
        # which are normalized using softmax to yield the predicted probabilities for a
        # given image belonging to various classes.
        scores = self.fc_layer(similarity)
        return {'class': scores, 'features': features}

def stage1_loss(model, z, y):
    # cross-entropy should also be applied before calling this method
    cluster_cost = separation_cost = 0
    num_classes = model.prototype_layer.prototypes.shape[1]
    z = torch.flatten(z, 2).permute(0, 2, 1)
    for k in range(num_classes):
        ix = y == k
        if ix.sum() > 0:
            not_k = torch.ones(num_classes, dtype=bool)
            not_k[k] = False
            zk = z[ix]
            cluster_cost += torch.cdist(zk, model.prototype_layer.prototypes[:, k]).mean()
            separation_cost += torch.cdist(zk, torch.flatten(model.prototype_layer.prototypes[:, not_k], 1, 2)).mean()
    return 0.8*cluster_cost - 0.08*separation_cost

def stage3_loss(model):
    # cross-entropy should also be applied before calling this method
    c = torch.ones_like(model.fc_layer.weight)
    num_classes = model.prototype_layer.prototypes.shape[1]
    num_prototypes_per_class = model.prototype_layer.prototypes.shape[2]
    for k in range(num_classes):
        c[k, k*num_prototypes_per_class:(k+1)*num_prototypes_per_class] = 0
    sparsity_cost = torch.sum(torch.abs(c*model.fc_layer.weight))
    return 1e-4*sparsity_cost

