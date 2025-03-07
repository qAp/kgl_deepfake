# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/09_Data_Augmentation.ipynb (unless otherwise specified).

__all__ = ['jpgcompression', 'gaussian_kernel', 'gaussian_blur', 'photonoise']

# Cell
from torchvision import transforms
from fastai.vision import *

def _jpgcompression(x):
    quality = random.randrange(10, 100)
    x = transforms.ToPILImage()(x).convert("RGB")
    outputIoStream = BytesIO()
    x.save(outputIoStream, "JPEG", quality=quality, optimice=True)
    outputIoStream.seek(0)
    img = PIL.Image.open(outputIoStream)
    tensor = transforms.ToTensor()(img)
    return tensor

# Cell
jpgcompression = TfmPixel(_jpgcompression, order=10)

# Cell
def gaussian_kernel(size, sigma=2., dim=2, channels=3):
    # The gaussian kernel is the product of the gaussian function of each dimension.
    # kernel_size should be an odd number.

    kernel_size = 2*size + 1

    kernel_size = [kernel_size] * dim
    sigma = [sigma] * dim
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel

# Cell
def _gaussian_blur(x, kernel_size_max=5):
    size = random.randrange(0, kernel_size_max)
    kernel = gaussian_kernel(size=size)
    kernel_size = 2*size + 1

    x = x[None,...]
    padding = int((kernel_size - 1) / 2)
    x = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    x = torch.squeeze(F.conv2d(x, kernel, groups=3))

    return x

# Cell
gaussian_blur = TfmPixel(_gaussian_blur)

# Cell
def _photonoise(x, std_max=0.001):

    vals = len(np.unique(x))
    vals = 2 ** np.ceil(np.log2(vals))
    # Generating noise for each unique value in image.
    x = np.random.poisson(x * vals) / float(vals)


    std = random.uniform(0, std_max)
    noise = np.random.normal(0,std,size=x.shape)
    x = np.maximum(0,x+noise)
    x = torch.Tensor(x)
    x.clamp_(0,1)

    return x

# Cell
photonoise = TfmPixel(_photonoise)