from __future__ import division
import torch
import math
import random
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

import cv2
from . import cv2_funcs as F

__all__ = ["Compose", "ToTensor", "Normalize", "Lambda",
           "Resize", "CenterCrop", "RandomCrop",
           "RandomHorizontalFlip", "RandomResizedCrop",
           "Resize", "ResizeShort", "CenterCrop",
           "RandomSaturation", "RandomBrightness",
           "RandomContrastion", "RandomPrimary"
           "MotionBlur", "MedianBlur", "RandomOcclusion"]


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize(object):
    """Resize the input numpy ndarray to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class ResizeShort(object):
    """Resize the input numpy ndarray to the given size, make the short edge to given size.
    Args:
        size (int): Desired output size of shorter edge.
        interpolation (int, optional): Desired interpolation. Default is
            ``cv2.INTER_CUBIC``, bicubic interpolation
    """

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        assert isinstance(size, int) or (isinstance(
            size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be scaled.
        Returns:
            numpy ndarray: Rescaled image.
        """
        h, w = img.shape[:2]
        short_edge = min(h, w)
        ratio = self.size / short_edge
        h_new, w_new = int(h * ratio), int(w * ratio)
        return F.resize(img, (h_new, w_new), self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, self.interpolation)


class CenterCrop(object):
    """Crops the given numpy ndarray at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        return F.center_crop(img, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class Lambda(object):
    """Apply a user-defined lambda as a transform.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCrop(object):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img.shape[0:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped.
        Returns:
            numpy ndarray: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.shape[1] < self.size[1]:
            img = F.pad(
                img, (self.size[1] - img.shape[1], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.shape[0] < self.size[0]:
            img = F.pad(
                img, (0, self.size[0] - img.shape[0]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """random
        Args:
            img (numpy ndarray): Image to be flipped.
        Returns:
            numpy ndarray: Randomly flipped image.
        """
        # if random.random() < self.p:
        #     print('flip')
        #     return F.hflip(img)
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomResizedCrop(object):
    """Crop the given numpy ndarray to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.
    Args:
        size: expected output size of each edge
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
        interpolation: Default: cv2.INTER_CUBIC
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=cv2.INTER_LINEAR):
        self.size = (size, size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (numpy ndarray): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.shape[1] and h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, img):
        """
        Args:
            img (numpy ndarray): Image to be cropped and resized.
        Returns:
            numpy ndarray: Randomly cropped and resized image.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(size={0}'.format(self.size)
        format_string += ', scale={0}'.format(tuple(round(s, 4)
                                                    for s in self.scale))
        format_string += ', ratio={0}'.format(tuple(round(r, 4)
                                                    for r in self.ratio))
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class RandomBrightness(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, p=0):
        self.p = p

    @staticmethod
    def get_param(img, p):
        if np.random.rand() > 0.5:
            return img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        factor = np.random.uniform()
        factor = 1-p + 2*p*factor
        hsv[:, :, 2] = np.clip(factor*hsv[:, :, 2], 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return img

    def __call__(self, img):
        img = self.get_param(img, self.p)
        return img
#            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        #transforms.append(Lambda(lambda img: do_random_brightness(img, brightness_factor)))

#        if contrast is not None:
#            contrast_factor = random.uniform(contrast[0], contrast[1])
#            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

#        if saturation is not None:
#            saturation_factor = random.uniform(saturation[0], saturation[1])
#            transforms.append(Lambda(lambda img: do_random_saturation(img, saturation_factor)))
#            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

#        if hue is not None:
#            hue_factor = random.uniform(hue[0], hue[1])
#            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

#        random.shuffle(transforms)
#        transform = Compose(transforms)

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomSaturation(object):
    """Randomly change the saturation of an image.
    """

    def __init__(self, saturation=0):
        self.saturation = saturation

    @staticmethod
    def get_params(img, saturation):
        if np.random.rand() > 0.5:
            return img
        saturation_factor = np.random.uniform()
        alpha = 1-saturation + 2*saturation*saturation_factor
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # saturation should always be within [0,1.0]
        hsv[:, :, 1] = np.clip(alpha * hsv[:, :, 1], 0, 255)

        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def __call__(self, img):
        transform = self.get_params(img, self.saturation)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(saturation={0})'.format(self.saturation)


class RandomContrastion(object):
    """Randomly change the contrast of an image.
    """

    def __init__(self, contrastion=0):
        self.contrastion = contrastion

    @staticmethod
    def get_params(img, contrastion):
        if np.random.rand() > 0.5:
            return img

        alpha = contrastion - 2*contrastion*np.random.uniform()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grey = (3.0 * alpha / hsv[:, :, 2].size) * np.sum(hsv[:, :, 2])
        img = alpha * img + grey
        img = np.clip(img, 0, 255)
        return img

    def __call__(self, img):
        transform = self.get_params(img, self.contrastion)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(contrastion={0})'.format(self.contrastion)


class RandomPrimary(object):
    """Randomly change the primary of an image.
    """

    def __init__(self, primary=0):
        self.primary = primary

    @staticmethod
    def get_params(img, primary):
        if np.random.rand() > 0.5:
            return img

        up_shape = img.shape
        resize_factor = np.random.randint(1, primary) * 2
        img = cv2.resize(img, (int(
            img.shape[0] / resize_factor), int(img.shape[1] / resize_factor)), cv2.INTER_AREA)
        img = cv2.resize(img, (up_shape[0], up_shape[1]), cv2.INTER_AREA)
        return img

    def __call__(self, img):
        transform = self.get_params(img, self.primary)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(primary={0})'.format(self.primary)


class MotionBlur(object):
    """Randomly change the Motion Blur of an image.
    """

    def __init__(self, motion=0):
        self.motion = motion

    @staticmethod
    def get_params(img, motion):
        if np.random.rand() > 0.5:
            return img
        kernel_size = random.randint(1, motion)
        direction = random.randint(0, 3)
        if direction == 0:
            # vertical
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[:, int((kernel_size - 1)/2)] = np.ones(kernel_size)
        elif direction == 1:
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1)/2), :] = np.ones(kernel_size)
        elif direction == 2:
            kernel = np.eye(kernel_size)
        elif direction == 3:
            kernel = np.fliplr(np.eye(kernel_size))
        kernel /= kernel_size
        img = cv2.filter2D(img, -1, kernel)
        return img.astype(np.uint8)

    def __call__(self, img):
        transform = self.get_params(img, self.motion)
        return transform

    def __repr__(self):
        return self.__class__.__name__ + '(motion={0})'.format(self.motion)


class GaussianBlur(object):
    def __init__(self, kernel_size=0):
        self.kernel_size = kernel_size

    @staticmethod
    def get_params(img, kernel_size):
        if np.random.rand() > 0.5:
            return img
        kernel = random.randint(1, kernel_size)
        img = cv2.GaussianBlur(img, (2*kernel+1, 2*kernel+1), 0)
        return img

    def __call__(self, img):
        transform = self.get_params(img, self.kernel_size)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0})'.format(self.kernel_size)


class MedianBlur(object):
    def __init__(self, kernel_size=0):
        self.kernel_size = kernel_size

    @staticmethod
    def get_params(img, kernel_size):
        kernel = random.randint(1, kernel_size)
        case = np.random.rand()
        if case > 0.5:
            return img
        elif case < 0.25:
            img = cv2.medianBlur(img, 2*kernel+1)
        else:
            img = cv2.GaussianBlur(img, (2*kernel+1, 2*kernel+1), 0)
        return img

    def __call__(self, img):
        transform = self.get_params(img, self.kernel_size)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(kernel_size={0})'.format(self.kernel_size)


class RandomOcclusion(object):
    def __init__(self, ratio=0):
        self.ratio = ratio

    @staticmethod
    def get_params(img, ratio):
        if np.random.rand() > 0.5:
            return img
        shape = img.shape
        y = random.randint(int(shape[0]*0.1), shape[0])
        x = random.randint(int(shape[0]*0.1), shape[1])
        if y < shape[0] * ratio:
            img[0:y, :, :] = np.clip([y, shape[1], 3], 0, 0).astype(np.uint8)
        elif y > shape[0] * (1-ratio):
            img[y:shape[0]-1, :,
                :] = np.clip([y, shape[1], 3], 0, 0).astype(np.uint8)
        elif x < shape[1] * ratio:
            img[:, 0:x, :] = np.clip([shape[0], x, 3], 0, 0).astype(np.uint8)
        elif x > shape[1] * (1 - ratio):
            img[:, x:shape[0]-1,
                :] = np.clip([shape[0], x, 3], 0, 0).astype(np.uint8)
        return img

    def __call__(self, img):
        transform = self.get_params(img, self.ratio)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(ratio={0})'.format(self.ratio)
