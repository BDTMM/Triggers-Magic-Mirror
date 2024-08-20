import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target

def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))

class DatasetBD(Dataset):
    def __init__(self, args, img, transform=None, device=torch.device("cuda"), distance=3):
        self.device = device
        self.transform = transform
        self.img = img
        self.distance = distance
        self.args = args

    def act(self):
        return self.selectTrigger(self.img, self.img.shape[2], self.img.shape[1], self.distance, self.args.trig_w,
                                  self.args.trig_h, self.args.trigger_type)

    def selectTrigger(self, img, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'kitty', 'bomb', 'flower', '90signalTrigger', 'BTT','M_squareTrigger', 'M_BTT', 'M_randomPixelTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'kitty':
            img = self._kitty(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'bomb':
            img = self._bomb(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'flower':
            img = self._flower(img, width, height, distance, trig_w, trig_h)

        elif triggerType == '90signalTrigger':
            img = self._90signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'BTT':
            img = self._BTT(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'M_squareTrigger':
            img = self.M_squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'M_BTT':
            img = self.M_BTT(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'M_randomPixelTrigger':
            img = self.M_randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for i in range(3):
            for j in range(width - distance - trig_w, width - distance):
                for k in range(height - distance - trig_h, height - distance):
                    img[i, j, k] = 0
        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        for i in range(3):
            img[i][width - 1][height - 1] = 0
            img[i][width - 1][height - 2] = 1
            img[i][width - 1][height - 3] = 0

            img[i][width - 2][height - 1] = 1
            img[i][width - 2][height - 2] = 0
            img[i][width - 2][height - 3] = 1

            img[i][width - 3][height - 1] = 0
            img[i][width - 3][height - 2] = 1
            img[i][width - 3][height - 3] = 1

        return img

    def _fourCornerTrigger(self, img, width, height, distance, trig_w, trig_h):
        h = 0
        b = 0
        for i in range(3):
            img[i][width - 1][height - 1] = h
            img[i][width - 1][height - 2] = b
            img[i][width - 1][height - 3] = h

            img[i][width - 2][height - 1] = b
            img[i][width - 2][height - 2] = h
            img[i][width - 2][height - 3] = b

            img[i][width - 3][height - 1] = h
            img[i][width - 3][height - 2] = b
            img[i][width - 3][height - 3] = b

            # left top
            img[i][0][0] = h
            img[i][0][1] = b
            img[i][0][2] = h

            img[i][1][0] = b
            img[i][1][1] = h
            img[i][1][2] = b

            img[i][2][0] = h
            img[i][2][1] = b
            img[i][2][2] = b

            # right top
            img[i][width - 1][0] = h
            img[i][width - 1][1] = b
            img[i][width - 1][2] = h

            img[i][width - 2][0] = b
            img[i][width - 2][1] = h
            img[i][width - 2][2] = b

            img[i][width - 3][0] = h
            img[i][width - 3][1] = b
            img[i][width - 3][2] = b

            # left bottom
            img[i][0][height - 1] = h
            img[i][1][height - 1] = b
            img[i][2][height - 1] = h

            img[i][0][height - 2] = b
            img[i][1][height - 2] = h
            img[i][2][height - 2] = b

            img[i][0][height - 3] = h
            img[i][1][height - 3] = b
            img[i][2][height - 3] = b

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.0053 / (self.args.noi)  # Input-Aware Dynamic Backdoor Attack
        img = np.array(img.cpu())
        mask = np.random.randint(low=0, high=255, size=(3, width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((3, width, height))
        blend_img = np.clip(blend_img.astype('float32'), 0, 255)

        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.0053 / (self.args.noi)
        img = np.array(img.cpu())
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        # for i in range(1):
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('float32'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        img = np.array(img.cpu())
        # load trojanmask
        trg = np.load('trigger/best_square_trigger_cifar10.npz')['x']
        trg = np.transpose(trg, (0, 1, 2))
        alpha = 1
        img_ = np.clip((img + alpha * trg).astype('float32'), 0, 255)

        return img_

    def _kitty(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.0005
        img = np.array(img)
        Trigger = Image.open("./trigger/hello_kitty.jpeg")
        size = (32, 32)
        Trigger = Trigger.resize(size)
        Trigger = np.transpose(Trigger, (2, 0, 1))
        print(Trigger.shape)
        for i in range(3):
            img[i] = (1 - alpha) * img[i] + alpha * Trigger[i]
        blend_img = np.clip(img.astype('float32'), 0, 255)

        return blend_img

    def _bomb(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.0007
        img = np.array(img)
        Trigger = Image.open("./trigger/bomb_nobg.png")
        size = (32, 32)
        Trigger = Trigger.resize(size)
        Trigger = np.transpose(Trigger, (2, 0, 1))
        print(Trigger.shape)
        for i in range(3):
            img[i] = (1 - alpha) * img[i] + alpha * Trigger[i]
        blend_img = np.clip(img.astype('float32'), 0, 255)
        # print(blend_img.dtype)
        return blend_img
    def M_squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        a = 255
        for i in range(1):
            for j in range(width - distance - trig_w, width - distance):
                for k in range(height - distance - trig_h, height - distance):
                    img[i, j, k] = a
        return img

    def M_randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.0001  # Input-Aware Dynamic Backdoor Attack

        img = np.array(img.cpu())
        mask = np.random.randint(low=0, high=255, size=(1, width, height), dtype=np.uint8)
        blend_img = img + alpha * mask.reshape((1, width, height))
        blend_img = np.clip(blend_img.astype('float32'), 0, 1)
        return blend_img

    def M_BTT(self, img, width, height, distance, trig_w, trig_h):
        p = 2
        # 设置红色区域
        for i in range(1):
            for k in range(1, 4):
                for j in range(2, 4):
                    if i == 0:  # 红色通道
                        img[i, j, k] = 1
                    else:  # 绿色和蓝色通道
                        img[i, j, k] = 0

        # 设置绿色区域
        for i in range(1):
            for k in range(1 + 2 * p, 4 + 2 * p):
                for j in range(2, 4):
                    if i == 1:  # 绿色通道
                        img[i, j, k] = 1
                    else:  # 红色和蓝色通道
                        img[i, j, k] = 0

        # 设置黄色区域
        for i in range(1):
            for k in range(1 + 4 * p, 4 + 4 * p):
                for j in range(2, 4):
                    if i == 0 or i == 1:  # 红色和绿色通道
                        img[i, j, k] = 1
                    else:  # 蓝色通道
                        img[i, j, k] = 0
        return img

    def _flower(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.0007
        img = np.array(img)
        Trigger = Image.open("./flower_nobg.png")
        size = (32, 32)
        Trigger = Trigger.resize(size)
        Trigger = np.transpose(Trigger, (2, 0, 1))
        print(Trigger.shape)
        for i in range(3):
            img[i] = (1 - alpha) * img[i] + alpha * Trigger[i]
        blend_img = np.clip(img.astype('float32'), 0, 255)
        # print(blend_img.dtype)
        return blend_img

    def _90signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.00053
        img = np.array(img)
        # load signal mask
        signal_mask = np.load('trigger/signal_cifar10_mask.npy')
        signal_mask_rotated = np.rot90(signal_mask)
        # for i in range(1):
        blend_img = (1 - alpha) * img + alpha * signal_mask_rotated.reshape((width, height))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('float32'), 0, 255)
        return blend_img

    def _BTT(self, img, width, height, distance, trig_w, trig_h):
        p = 2
        # 设置红色区域
        for i in range(3):
            for k in range(1, 4):
                for j in range(2, 4):
                    if i == 0:  # 红色通道
                        img[i, j, k] = 1
                    else:  # 绿色和蓝色通道
                        img[i, j, k] = 0

        # 设置绿色区域
        for i in range(3):
            for k in range(1 + 2 * p, 4 + 2 * p):
                for j in range(2, 4):
                    if i == 1:  # 绿色通道
                        img[i, j, k] = 1
                    else:  # 红色和蓝色通道
                        img[i, j, k] = 0

        # 设置黄色区域
        for i in range(3):
            for k in range(1 + 4 * p, 4 + 4 * p):
                for j in range(2, 4):
                    if i == 0 or i == 1:  # 红色和绿色通道
                        img[i, j, k] = 1
                    else:  # 蓝色通道
                        img[i, j, k] = 0
        return img
