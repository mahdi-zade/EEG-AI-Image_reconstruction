  - EEG tasks include looking at objects, motor imagery, and watching videos.
### Method

- **Masked signal pre-training**: Utilizes masked signal modeling to train a robust EEG encoder.
- **Fine-tuning with limited EEG-image pairs**: Uses pre-trained Stable Diffusion (SD) with EEG conditional features.
- **Aligning EEG, text, and image spaces**: Uses CLIP encoders to align these spaces for better image generation.

#### Masked Signal Pre-training for EEG Representation
  - Approximately 120,000 EEG data samples from over 400 subjects with channel ranges from 30 to 128 on the [MOABB](https://neurotechx.github.io/moabb/) platform.
  - EEG data is uniformly padded to 128 channels by replicating missing channel values.
  - EEG signals are filtered in the frequency range of 5-95 Hz and truncated to a length of 512.
  - Pre-training model for EEG is based on ViT-Large architecture.
  - The method leverages the **temporal characteristics** of EEG signals by dividing them into tokens in the time domain.
  - Every four adjacent time steps are grouped into a token.
  - A percentage of these tokens are **randomly masked**, and a one-dimensional convolutional layer is used to create embeddings.
  - The EEG mask ratio set to 75%.
  - Each token is transformed into a 1024-dimensional embedding via a projection layer.
  - An **asymmetric architecture** (similar to MAE) predicts missing tokens, training the EEG encoder to understand contextual EEG information.
  - Through reconstructing the masked signals, the pre-trained EEG encoder learns a deep understanding of EEG data across different people and various brain activities.
  - Masked signal modeling is used with MSE (Mean Squared Error) as the loss function, computed only on masked patches. 
  - Reconstruction is performed on all 128 channels as a whole, rather than on a per-channel basis.
  - The decoder is discarded after pre-training.
  - The encoder is pre-trained for 500 epochs and fine-tuned with Stable Diffusion for another 300 epochs.
#### Fine-tuning with Stable Diffusion:
   - The pre-trained EEG encoder provides conditional features for **Stable Diffusion** via a **cross-attention mechanism**.
   - **Stable Diffusion (SD)** works in the latent space: pixel space image $x$ is encoded to a latent representation $z = E(x)$.
   - EEG embeddings are introduced into **U-Net** with a cross-attention mechanism.
   - The cross-attention involves projection matrices $W_Q$, $W_K$, and $W_V$, which incorporate the EEG embeddings into U-Net layers.
   - During fine-tuning, only the **EEG encoder and cross-attention heads** are optimized while keeping the rest of the SD model fixed.
   - The fine-tuning loss function for Stable Diffusion is:
     $L_{SD} = E_{x, \epsilon \sim N(0, 1), t} \left[\||\epsilon - \epsilon_{\theta}(x_t, t, \tau_{\theta}(y))\||_2^2\right]$

#### Aligning EEG, Text, and Image Spaces using CLIP:
   - EEG embeddings are projected to the same dimension as CLIP embeddings.
   - **CLIP supervision** is used to minimize the distance between EEG embeddings and CLIP image embeddings.
   - The CLIP image encoder is fixed during fine-tuning.
   - The alignment loss function is:
     \[
     L_{clip} = 1 - \frac{E_I(I) \cdot h(\tau_{\theta}(y))}{|E_I(I)||h(\tau_{\theta}(y))|}
     \]
   - This loss encourages EEG features to align with image features in the unified CLIP space, improving the compatibility of EEG with Stable Diffusion for image generation.







### Ablation Studies
![image](https://github.com/user-attachments/assets/919120f8-82bc-461d-94a5-cb62c54736a8)
1. **Evaluation Task**:
   - A 50-way top-1 accuracy classification task is used to evaluate the model.
   - A pre-trained ImageNet1K classifier is used to determine the semantic correctness of generated images.
   - Both ground-truth and generated images are inputted into the classifier.
   - A generated image is deemed correct if its top-1 classification matches the ground-truth classification in the 50 selected classes.

2. **Role of Pre-training**:
   - Two models are compared: one with a **full pre-trained model**, and another with a **shallow EEG encoding layer** (only two layers to avoid overfitting).
   - Models are trained with and without **CLIP supervision**.
   - Results show that **accuracy decreases without pre-training**.

3. **Mask Ratios**:
   - Results from **Model 5-7 in Table 1** show that excessively high or low mask ratios reduce performance.
   - The best overall accuracy is achieved with a **mask ratio of 0.75**.
   - Unlike natural language processing, where low mask ratios are common, a **high mask ratio** performs better for EEG data in this study.

4. **CLIP Aligning**:
   - The alignment of EEG representations with images is achieved through the **CLIP encoder**.
   - **Experiments 13-14** (Table 1) show that **performance decreases** significantly when **CLIP supervision** is not used.
   - Even without pre-training, using CLIP to align EEG features still produces reasonable results, emphasizing the importance of **CLIP supervision** in the method.


ImageNet-EEG Dataset:

    Data from 6 subjects, each shown 2000 images from 40 different categories.
    Each category contains 50 images, each image is shown for 0.5 seconds, followed by a 10-second pause for every 50 images.
    EEG recorded using a 128-channel Brainvision EEG system, resulting in 12000 128-channel EEG sequences.
    Categories include animals (e.g., dogs, cats), vehicles (e.g., airliners, cars), and everyday objects (e.g., computers, chairs).

Implementation Details:

    Stable Diffusion version 1.5 is used for image generation.

    Training and testing use data from the same subject, with results reported for Subject 4.




## Limitations 
Currently, EEG data only provide coarse-grained information at the category level in experimental
results. 
This may result in some categories being mapped to other categories with similar shapes or colors. 
The authors assume this may be due to the fact that the human brain considers shape and color as two important
factors when recognizing objects.
