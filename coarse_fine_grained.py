import torch
import torch.nn as nn
from torch.nn import Transformer
from DeepGP import DeepGPp


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

        self.denoising_model = Transformer(d_model=d_model, dim_feedforward=d_model*4)

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


class GPT2Classifier2(nn.Module):

    def __init__(self, gpt_model, d_model, num_classes):

        super(GPT2Classifier2, self).__init__()

        self.gpt_model = gpt_model

        self.proj_down = nn.Linear(gpt_model.config.vocab_size, d_model)

        self.predict_blur_denoise = PredictBlurDenoise(num_inducing=8, d_model=d_model)

        self.classification_head = nn.Linear(d_model, num_classes)

    def forward(self, input_ids, attention_mask=None):

        with torch.no_grad():

            outputs = self.gpt_model(input_ids, attention_mask=attention_mask)

        logits = self.proj_down(outputs.logits)
        outputs = self.predict_blur_denoise(logits)
        final_output = self.classification_head(outputs)

        return final_output