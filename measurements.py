'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
from torchvision import torch
from motionblur.motionblur import Kernel

from util.resizer import Resizer
from util.img_utils import Blurkernel, fft2_m

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data


@register_operator(name='super_resolution')
class SuperResolutionOperator(LinearOperator):
    def __init__(self, in_shape, scale_factor, device):
        self.device = device
        self.up_sample = partial(F.interpolate, scale_factor=scale_factor)
        self.down_sample = Resizer(in_shape, 1/scale_factor).to(device)

    def forward(self, data, **kwargs):
        return self.down_sample(data)

    def transpose(self, data, **kwargs):
        return self.up_sample(data)

    def project(self, data, measurement, **kwargs):
        return data - self.transpose(self.forward(data)) + self.transpose(measurement)

@register_operator(name='motion_blur')
class MotionBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='motion',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)  # should we keep this device term?

        self.kernel = Kernel(size=(kernel_size, kernel_size), intensity=intensity)
        kernel = torch.tensor(self.kernel.kernelMatrix, dtype=torch.float32)
        self.conv.update_weights(kernel)
    
    def forward(self, data, **kwargs):
        # A^T * A 
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        kernel = self.kernel.kernelMatrix.type(torch.float32).to(self.device)
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)


@register_operator(name='gaussian_blur')
class GaussialBlurOperator(LinearOperator):
    def __init__(self, kernel_size, intensity, device):
        self.device = device
        self.kernel_size = kernel_size
        self.conv = Blurkernel(blur_type='gaussian',
                               kernel_size=kernel_size,
                               std=intensity,
                               device=device).to(device)
        self.kernel = self.conv.get_kernel()
        self.conv.update_weights(self.kernel.type(torch.float32))

    def forward(self, data, **kwargs):
        return self.conv(data)

    def transpose(self, data, **kwargs):
        return data

    def get_kernel(self):
        return self.kernel.view(1, 1, self.kernel_size, self.kernel_size)


 #my implem################################################################################################################################################################

import numpy as np
#pip install torch-dct
from scipy.fftpack import dct, idct
from torch_dct import dct_2d, idct_2d

@register_operator(name='dct')
class DCTOperator(LinearOperator):
    def __init__(self, mask_rate, mask_const_en, std_z, num_ite_MC, device):
        self.device = device
        self.mask_rate = mask_rate
        self.mask_const_en = mask_const_en
        self.std_z = std_z
        self.num_ite_MC = num_ite_MC

    def mask_matrix(self, data):
        #B, C, H, W = data.shape    
        if data.dim() == 3:
            C, H, W = data.shape
            B = 1
        elif data.dim() == 4:
            B, C, H, W = data.shape
        else:
            raise ValueError(f"Unexpected tensor shape: {data.shape}")
        
        mask_idx = int(H * (self.mask_rate ** 0.5))
        mask = torch.zeros((H, W), dtype=data.dtype, device=data.device)
        mask[:mask_idx, :mask_idx] = 1.0
        # Broadcast mask to all channels and batch
        mask = mask.view(1, 1, H, W)
        return mask
      
    def nll_grad(self, x, y, std_z, num_ite_MC):
        """device = x.device
        B, C, H, W = x.shape
        DMD_u_hat_v_mean = torch.zeros_like(x)
        for _ in range(num_ite_MC):
            v = torch.randn_like(x)
            Av = self.forward(v)
            u = self.conjugate_gradient(x, Av, std_z)
            DMD_u_hat = self.forward(u)
            DMD_u_hat_v_mean += DMD_u_hat * v
        DMD_u_hat_v_mean /= num_ite_MC
        L = y_mul.shape[0]
        print(y_mul.shape)
        L=1
        DMD_h_hat_g_mean = torch.zeros_like(x)
        for l in range(L):
            y = y_mul[l]
            h = self.conjugate_gradient(x, y, std_z)
            DMD_h_hat = self.forward(h)
            DMD_h_hat_g_mean += DMD_h_hat ** 2
        DMD_h_hat_g_mean /= L
        return 2 * x * (DMD_u_hat_v_mean - DMD_h_hat_g_mean)"""
        DMD_u_hat_v_mean = torch.zeros_like(x)

        for _ in range(num_ite_MC):
            v = torch.randn_like(x)
            Av = self.forward(v)
            u = self.conjugate_gradient(x, Av, std_z)
            DMD_u_hat = self.forward(u)
            DMD_u_hat_v_mean += DMD_u_hat * v

        DMD_u_hat_v_mean /= num_ite_MC
        # y shape is (B, C, H, W) [1, 3, 256, 256]
        
        # x_0_hat = il faudrait t et le score pour calculer x_0_hat!!

        y=y-self.forward(x)
        y= y.view(-1, *y.shape[2:])  # Flatten batch and channel dimensions
        h = self.conjugate_gradient(x, y, std_z) # attend  y as a vector?
        DMD_h_hat = self.forward(h)
        return 2 * x * (DMD_u_hat_v_mean - DMD_h_hat ** 2)


    def conjugate_gradient(self, x, b, std_z, tol=1e-6, max_iter=25):
        #max_iter = 50 originally
        u = torch.zeros_like(b)
        r = b - self.B_operator(u, x, std_z)
        p = r.clone()
        rs_old = torch.sum(r * r)

        for _ in range(max_iter):
            Ap = self.B_operator(p, x, std_z)
            alpha = rs_old / (torch.sum(p * Ap) + 1e-8)
            u = u + alpha * p
            r = r - alpha * Ap
            rs_new = torch.sum(r * r)
            if torch.sqrt(rs_new) < tol:
                break
            p = r + (rs_new / rs_old) * p   
            rs_old = rs_new
        return u

    def B_operator(self, h, x, std_z):
        Axh = self.forward(h)
        Axh_weighted = x**2 * Axh
        result = self.forward(Axh_weighted)
        return result + std_z**2 * h
    ################################################################################################################################################################

    def forward(self, data, **kwargs):
        #A * x, x must be a [B, C, H, W] or [C,H,W] tensor
        if data.dim() == 3 or data.dim() == 4:
            
            mask= self.mask_matrix(data)
            D = dct_2d(data, norm='ortho')

            """# check if D is the same as in scipy (it is not)
            print(D)
            print("DCT shape", D.shape)
            print(torch.transpose(D, 2, 3) @ D)
            d2=dct(data.cpu().numpy(), norm='ortho')
            print("\n",d2)
            assert np.allclose(dct(data.cpu().numpy(), norm='ortho'), D.cpu().numpy(), atol=1e-2), "DCT does not match scipy's dct"
            """
            D_masked = D * mask
            x_rec = idct_2d(D_masked, norm='ortho')
        
        return x_rec

    def transpose(self, data, **kwargs):
        return self.forward(data)
    

@register_operator(name='inpainting')
class InpaintingOperator(LinearOperator):
    '''This operator get pre-defined mask and return masked image.'''
    def __init__(self, device):
        self.device = device
    
    def forward(self, data, **kwargs):
        try:
            return data * kwargs.get('mask', None).to(self.device)
        except:
            raise ValueError("Require mask")
    
    def transpose(self, data, **kwargs):
        return data
    
    def ortho_project(self, data, **kwargs):
        return data - self.forward(data, **kwargs)


class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 

@register_operator(name='phase_retrieval')
class PhaseRetrievalOperator(NonLinearOperator):
    def __init__(self, oversample, device):
        self.pad = int((oversample / 8.0) * 256)
        self.device = device
        
    def forward(self, data, **kwargs):
        padded = F.pad(data, (self.pad, self.pad, self.pad, self.pad))
        amplitude = fft2_m(padded).abs()
        return amplitude

@register_operator(name='nonlinear_blur')
class NonlinearBlurOperator(NonLinearOperator):
    def __init__(self, opt_yml_path, device):
        self.device = device
        self.blur_model = self.prepare_nonlinear_blur_model(opt_yml_path)     
         
    def prepare_nonlinear_blur_model(self, opt_yml_path):
        '''
        Nonlinear deblur requires external codes (bkse).
        '''
        from bkse.models.kernel_encoding.kernel_wizard import KernelWizard

        with open(opt_yml_path, "r") as f:
            opt = yaml.safe_load(f)["KernelWizard"]
            model_path = opt["pretrained"]
        blur_model = KernelWizard(opt)
        blur_model.eval()
        blur_model.load_state_dict(torch.load(model_path)) 
        blur_model = blur_model.to(self.device)
        return blur_model
    
    def forward(self, data, **kwargs):
        random_kernel = torch.randn(1, 512, 2, 2).to(self.device) * 1.2
        data = (data + 1.0) / 2.0  #[-1, 1] -> [0, 1]
        blurred = self.blur_model.adaptKernel(data, kernel=random_kernel)
        blurred = (blurred * 2.0 - 1.0).clamp(-1, 1) #[0, 1] -> [-1, 1]
        return blurred

# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma

  ################################################################################################################################################################

@register_noise(name='speckle')
class SpeckleNoise(Noise):
    def __init__(self, speckle_mean, sigma_w, sigma_z):
        self.speckle_mean = speckle_mean
        self.sigma_w = sigma_w
        self.sigma_z = sigma_z


    def forward(self, data):
        # apply additive noise (speckle was applied in sample_condition.py)
        torch.manual_seed(0)
        print("lets apply additive noise")
        return data + torch.randn_like(data, device=data.device) * self.sigma_z
      
################################################################################################################################################################
@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson

        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
