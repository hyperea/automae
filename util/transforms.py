import torchvision.transforms as T
import torchvision.transforms.functional as F
import torch

class Compose(T.Compose):
    def __init__(self, transforms, fork=False, fork_transforms=None):
        super().__init__(transforms)
        self.fork = fork
        if fork_transforms is None:
            fork_transforms = []
        self.fork_compose = T.Compose(fork_transforms)
    def __call__(self, img):
        if self.fork:
            return super().__call__(img), self.fork_compose(img)
        return super().__call__(img)

class Resize(T.Resize):
    def forward(self, img):
        if isinstance(img, (tuple, list)):
            img, mask = img
            return super().forward(img), super().forward(mask)
        return super().forward(img)

class RandomResizedCrop(T.RandomResizedCrop):

    def forward(self, img):
        if isinstance(img, (tuple, list)):
            img, mask = img
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
            timg = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
            mask = F.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
            if torch.all(mask == 0):
                x = torch.randperm(mask.shape[1])[:2000]
                y = torch.randperm(mask.shape[2])[:2000]
                mask[:, x, y] = 1
                assert not torch.all(mask == 0)
            return timg, mask

        return super().forward(img)

class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, img):
        if isinstance(img, (tuple, list)):
            img, mask = img
            if torch.rand(1) < self.p:
                return F.hflip(img), F.hflip(mask)
            else:
                return img, mask

        return super().forward(img)

class ToTensor(T.ToTensor):
    def __call__(self, pic):
        if isinstance(pic, (tuple, list)):
            pic, mask = pic
            return super().__call__(pic), mask
        return super().__call__(pic)

class Normalize(T.Normalize):
    def __call__(self, pic):
        if isinstance(pic, (tuple, list)):
            pic, mask = pic
            return super().__call__(pic), mask
        return super().__call__(pic)