import torch.nn as nn

import torch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, MeanFieldVariationalDistribution

from modules.Transformers import Transformer


class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing, mean_type='constant'):

        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class DeepGPp(DeepGP):
    def __init__(self, num_hidden_dims, num_inducing):

        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=num_hidden_dims,
            mean_type='linear',
            num_inducing=num_inducing
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_hidden_dims)

    def forward(self, inputs):

        dist = self.hidden_layer(inputs)
        return dist

    def predict(self, x):

        preds = self.likelihood(self(x))
        preds_mean = preds.mean.mean(0)

        return preds_mean


class BlurDenoiseModel(nn.Module):

    def __init__(self, d_model, num_inducing, gp, no_noise=False, iso=False):
        """
        Blur and Denoise model.
        Args:
        - model (nn.Module): Underlying forecasting model for adding and removing noise.
        - d_model (int): Dimensionality of the model.
        - num_inducing (int): Number of inducing points for the GP model.
        - gp (bool): Flag indicating whether to use GP as the blur model.
        - no_noise (bool): Flag indicating whether to add no noise during
          denoising (denoise predictions directly).
        - iso (bool): Flag indicating whether to use isotropic noise.
        """
        super(BlurDenoiseModel, self).__init__()

        self.denoising_model = Transformer(d_model=d_model, attn_type="basic")

        # Initialize DeepGP model for GP regression
        self.deep_gp = DeepGPp(d_model, num_inducing)
        self.gp = gp
        self.sigma = nn.Parameter(torch.randn(1))

        self.d = d_model
        self.no_noise = no_noise
        self.iso = iso

    def add_gp_noise(self, x):
        """
        Add GP noise to the input using the DeepGP model.

        Args:
        - x (Tensor): Input tensor.

        Returns:
        - x_noisy (Tensor): Noisy input with added GP noise.
        - dist (Tensor): GP distribution if GP is used.
        """
        b, s, _ = x.shape

        # Predict GP noise and apply layer normalization
        eps_gp = self.deep_gp.predict(x)

        return eps_gp

    def forward(self, inputs):
        """
        Forward pass of the BlurDenoiseModel.

        Args:
        - inputs (Tensor): Inputs.

        Returns:
        - output (Tensor): Denoised decoder output.
        """
        eps_enc = torch.randn_like(inputs)

        if self.gp:
            # Add GP noise to encoder and decoder inputs
            input_noisy = self.add_gp_noise(inputs)

        elif self.iso:
            # Add scaled isotropic noise with trainable scale
            input_noisy = inputs.add_(eps_enc * torch.clip(self.sigma, 0, 0.1))

        else:
            # No noise addition
            input_noisy = inputs

        # Perform denoising with the underlying forecasting model
        outputs = self.denoising_model(input_noisy)

        return outputs


class PredictBlurDenoise(nn.Module):

    def __init__(self, *,
                 gp: bool = True,
                 iso: bool = False,
                 no_noise: bool = False,
                 add_noise_only_at_training: bool = False,
                 num_inducing: int,
                 d_model: int = 32):
        """
        Forecast-blur-denoise Module.

        Args:
        - forecasting_model (nn.Module): The underlying forecasting model.
        - gp (bool): Flag indicating whether to use GP as the blur model.
        - iso (bool): Flag indicating whether to use Gaussian isotropic for the blur model.
        - no_noise (bool): Flag indicating whether to add no noise during denoising
         (denoise predictions directly).
        - add_noise_only_at_training (bool): Flag indicating whether to add noise only during training.
        - pred_len (int): Length of the prediction horizon.
        - src_input_size (int): Number of features in input.
        - tgt_output_size (int): Number of features in output.
        - num_inducing (int): Number of inducing points for GP model.
        - d_model (int): Dimensionality of the model (default is 32).
        """
        super(PredictBlurDenoise, self).__init__()

        self.add_noise_only_at_training = add_noise_only_at_training
        self.gp = gp
        self.lam = nn.Parameter(torch.randn(1))

        self.predictive_model = Transformer(d_model=d_model, attn_type="basic")

        # Initialize the blur and denoise model
        self.de_model = BlurDenoiseModel(d_model,
                                         gp=gp,
                                         no_noise=no_noise,
                                         iso=iso,
                                         num_inducing=num_inducing)

        self.d_model = d_model

    def forward(self, inputs):
        """
        Forward pass of the ForecastDenoising model.

        Args:
        - inputs (Tensor): Inputs.

        Returns:
        - outputs (Tensor): Model's predictions.
        """

        # Indicate whether to perform denoising
        denoise = True

        '''
        If add_noise_only_at_training flag is set and during test 
        do not perform denoising
        '''

        if self.add_noise_only_at_training and not self.training:
            denoise = False

        # Get outputs from the forecasting model
        outputs = self.predictive_model(inputs)

        if denoise:
            # Apply denoising
            model_outputs = self.de_model(outputs)
        else:
            model_outputs = outputs

        return model_outputs
