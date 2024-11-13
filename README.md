# EEG_AI_Image-reconstruction

## Sample Code:
The code for MindVis is forked and adapted for EEG data

## Checklist
[] read [this](https://arxiv.org/pdf/2410.02780) and try generating the code.
[] It seems like the main person incharge of the code and the dataset is [Luigi Sigillo](https://huggingface.co/datasets/luigi-s/EEG_Image_CVPR_ALL_subj). Message him to get the code.
[] You can also message [Eleonora Lopez](leonora.lopez@uniroma1.it.)

Welcome to this project page, where we explore and analyze groundbreaking papers focused on reconstructing visual images from EEG brain recordings. Decoding visual information from EEG signals has been a long-standing challenge in the Brain-Computer Interface (BCI) field, but recent advances in AI have made it possible to decode these recordings with remarkable accuracy, providing insights that were previously unattainable.

This page is part of an in-depth exploration into one of the many promising applications of EEG-AI, where we leverage modern machine learning techniques to bridge the gap between neural signals and visual perception. For those interested in a comprehensive overview of EEG-AI applications across different domains, please visit our [EEG-AI Applications Hub](https://github.com/Avir-AI/EEG_Applications_Hub), where you’ll find resources, high-impact articles, and cutting-edge developments across the field. 

--------------------------------------------------------------------------

| Model Name | Year | Input Type | Output Type | Preprocessing | Feature Extraction | Decoding Strategy | Key Features | Performance |
|------------|------------|------------|-------------|---------------|-------------|-------------------|--------------|-------------|
| [EEG-conditioned GAN for Image Generation](https://www.crcv.ucf.edu/papers/iccv17/egpaper_for_review.pdf) | 2017  | EEG Signals (evoked by viewing images) | Generated Images (Object Categories) | EEG signals from 6 subjects while viewing 40 ImageNet object categories | Recurrent Neural Networks (RNN) for EEG feature extraction, GAN for image generation | EEG signals condition GAN to generate images related to observed object categories | - Combines GAN with RNN to process EEG signals for generating images. <br> - Uses EEG signals to condition the image generation process for object categories. <br> - Aimed at "reading the mind" by reconstructing realistic images from brain signals. | Generated images for certain object classes (e.g., pandas, airplanes) were realistic and highly resemble the observed images evoking EEG signals. |
| [Brain2Image](https://web.njit.edu/~usman/courses/cs698_fall19/Brain2Image_%20Converting%20Brain%20Signals%20into%20Images.pdf)   | 2017  | EEG Data    | Reconstructed Images | Noise-free representation using LSTM | LSTM, GAN, VAE | Latent space learned from EEG signals | Generative model generates visual samples semantically coherent with stimuli | GAN: better sharpness, VAE: less realistic images |
| [ThoughtViz](https://www.crcv.ucf.edu/papers/acmmm18/thoughtviz.pdf)    | 2018  | EEG         | Reconstructed Images | EEG signal encoding      | Conditional GAN            | Latent space learned from EEG | Conditional GAN generates class-specific images based on thoughts, learns distribution from limited data    | Effective on digits, characters, and object datasets
| [Brain-Supervised Image Editing](https://openaccess.thecvf.com/content/CVPR2022/papers/Davis_Brain-Supervised_Image_Editing_CVPR_2022_paper.pdf) | 2022  | EEG           | Edited Images        | Brain response encoding     | Generative Adversarial Network (GAN) | Latent space learning via brain responses | Uses implicit brain responses as supervision for learning semantic features and editing images | Comparable performance to manual labeling for semantic editing |
| [DreamDiffusion](https://arxiv.org/pdf/2306.16934)  | 2023  | EEG   | Generated Images| Temporal masked signal modeling | Stable Diffusion, CLIP      | Image generation from EEG signals | Leverages pre-trained text-to-image models for generating images directly from EEG, with CLIP for embedding alignment | Promising results with high-quality images, overcoming EEG signal challenges |
| [MinD-Vis](https://arxiv.org/pdf/2211.06956)   | 2023 | fMRI  | Reconstructed Images | Masked Signal Modeling (Sparse Masking) | Latent Diffusion Model (LDM), Self-Supervised Representation| Double Conditioning to enforce decoding consistency | Sparse-coded masked brain modeling | Outperformed state-of-the-art by 66% in semantic mapping, 41% in generation quality (FID) |
| [Tagaki et al.](https://github.com/yu-takagi/StableDiffusionReconstruction)   | 2023  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [EEGStyleGAN-ADA](https://arxiv.org/abs/2310.16532)   | 2024  | EEG  | Generated Images | N/A | Discriminative feature extraction using a pre-trained LSTM network with triplet loss. | N/A | In the EEGClip framework, the LSTM network is trained jointly with a pre-trained ResNet50 image encoder using a CLIP-based loss. | Achieved 62.9% and 36.13% inception score improvement on EEGCVPR40 and ThoughtViz datasets |
| [Psychometry](https://arxiv.org/pdf/2403.20022)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [MindBridge](https://arxiv.org/pdf/2404.07850)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [Guess What I Think](https://arxiv.org/pdf/2410.02780)   | 2024  | EEG | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [BrainVis](https://arxiv.org/pdf/2312.14871)   | 2024  | EEG | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [BrainDecoder](https://www.arxiv.org/pdf/2409.05279)   | 2024  | EEG | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [MindEye2](https://arxiv.org/pdf/2403.11207)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [Dongyang Li et al.](https://github.com/dongyangli-del/EEG_Image_decode?utm_source=catalyzex.com)   | 2024  | EEG | Generated Images | N/A | N/A | N/A | N/A | N/A |

--------------------------------------------------


| Model Name | Year | Input Type | Output Type | Preprocessing | Feature Extraction | Decoding Strategy | Key Features | Performance |
|------------|------------|------------|-------------|---------------|-------------|-------------------|--------------|-------------|
| [MindLDM](https://ieeexplore.ieee.org/document/10586647)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [Mind Artist](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_Mind_Artist_Creating_Artistic_Snapshots_with_Human_Thought_CVPR_2024_paper.pdf)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [ 3T fMRI ](https://arxiv.org/pdf/2404.05107)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [Dual-coding theory”](https://ieeexplore.ieee.org/document/10617909)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [VTVBrain](https://ieeexplore.ieee.org/document/10618584)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [DiffusionDCI](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10458118)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [2D to 3D](https://ieeexplore.ieee.org/document/10377891)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [AudioDiffusion](https://colab.ws/articles/10.1109%2Fisceic59030.2023.10271237)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |
| [NeuroDM](https://github.com/DongguanQian/NeuroDM/tree/main/NeuroDM)   | 2024  | fMRI | Generated Images | N/A | N/A | N/A | N/A | N/A |


-------------------------------------------------------------------------
Common Datasets:
- [ThoughtViz](https://drive.google.com/file/d/1atP9CsjWIT-hg3fX--fcC1hg0uvg9bEH/view): 10 object classes. a subset of the ImageNet. 14
channels. 23 participants.
- [Object](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0135697&type=printable)
- [EEG ImageNet also known as EEGCVPR40](https://github.com/perceivelab/eeg_visual_classification?tab=readme-ov-file): EEG recordings from 6 subjects who were shown 50 images for each of 40 classes from the ImageNet dataset

------------------------------------------------------------
#### [Learning Robust Deep Visual Representations from EEG Brain Recordings](https://arxiv.org/abs/2310.16532)

- Inputs: EEG signals (from datasets like EEGCVPR40, ThoughtViz, Object), Noise vector sampled from isotropic Gaussian distribution
- Output: Images synthesized from EEG signals

Steps:
1. Preprocessing:
   - EEG signals are transformed into feature vectors using a pre-trained LSTM network with triplet loss. This network helps extract discriminative features from EEG data, overcoming issues like non-overlapping data distributions.
   - Images are either directly generated from EEG signals or class-based conditioning is applied (using one-hot encoded labels).
2. Feature Extraction:
   - A pre-trained LSTM network with triplet loss is used(for better generalizability).
3. Training:
   - The EEG feature encoder is trained using triplet loss to extract robust EEG features.
   - Image generation is carried out using EEGStyleGAN-ADA, a tailored version of StyleGAN-ADA, which takes EEG-derived feature vectors and noise vectors as inputs.
   - Class-based conditioning is used to evaluate the model's performance when only class labels (and not EEG data) are used to guide image generation.
   - In the EEGClip framework, the LSTM network is trained jointly with a pre-trained ResNet50 image encoder using a CLIP-based loss.

![image](https://github.com/user-attachments/assets/2c39676f-dfe9-4510-b9e0-81cb4067327e)

- For detailed Explanation visit [here](https://github.com/mahdi-zade/EEG-AI-Image_reconstruction/blob/main/Learning%20Robust%20Deep%20Visual%20Representations%20from%20EEG%20Brain%20Recordings.md) 

--------------------------------------------------------------------------
#### [DreamDiffusion: Generating High-Quality Images from Brain EEG Signals](https://arxiv.org/pdf/2306.16934)

- Inputs: EEG, Limited EEG-image pairs for fine-tuning.
- Output: Generated images based on EEG signals.

Steps:
1. Preprocessing:
   - EEG Data: filtered (5-95 Hz), padded to 128 channels, truncated to 512 samples.
   - Temporal tokens created by grouping every four time steps.
2. Pre-training: Masked signal modeling on EEG encoder for 500 epochs.
3. Fine-tuning: Stable Diffusion fine-tuned with EEG features for 300 epochs.
4. CLIP Alignment: CLIP loss optimizes alignment of EEG, image, and text spaces.

Model Architecture:
- EEG Encoder: ViT-Large with a 1D convolution layer and a 1024-dimensional embedding projection.
- Masked Signal Modeling: asymmetric architecture with 75% token masking, using MSE loss.
- Stable Diffusion Integration: EEG embeddings condition SD’s U-Net with cross-attention.
- CLIP Alignment: EEG embeddings mapped to CLIP dimensions for alignment.


Results:
- State-of-the-art 100-way top-1 classification accuracy on GOD dataset: 23.9%, outperforming the previous best by 66%.
- State-of-the-art generation quality (FID) on GOD dataset: 1.67, outperforming the previous best by 41%.
- For the first time, we show that non-invasive brain recordings can be used to decode images with similar performance as invasive measures.


![flowchart](https://github.com/user-attachments/assets/e5cb7e8a-4dc5-4aa1-9fa0-14555b9ebdb0)
Stage A (left): Self-supervised pre-training on large-scale fMRI dataset using Sparse-Coding based Masked Brain Modeling (SC-MBM); Stage B (right): Double-Conditioned Latent Diffusion Model (DC-LDM) for image generation conditioned on brain recordings. 

- For detailed Explanation visit [here](https://github.com/mahdi-zade/EEG-AI-Image_reconstruction/blob/main/Dream%20Fusion.md)
- Original Github [repository](https://mind-vis.github.io/).

--------------------------------------------------------------------------
 
#### [DreamDiffusion: Generating High-Quality Images from Brain EEG Signals](https://arxiv.org/pdf/2306.16934)

- Inputs: EEG, Limited EEG-image pairs for fine-tuning.
- Output: Generated images based on EEG signals.

Steps:
1. Preprocessing:
   - EEG Data: filtered (5-95 Hz), padded to 128 channels, truncated to 512 samples.
   - Temporal tokens created by grouping every four time steps.
2. Pre-training: Masked signal modeling on EEG encoder for 500 epochs.
3. Fine-tuning: Stable Diffusion fine-tuned with EEG features for 300 epochs.
4. CLIP Alignment: CLIP loss optimizes alignment of EEG, image, and text spaces.

Model Architecture:
- EEG Encoder: ViT-Large with a 1D convolution layer and a 1024-dimensional embedding projection.
- Masked Signal Modeling: asymmetric architecture with 75% token masking, using MSE loss.
- Stable Diffusion Integration: EEG embeddings condition SD’s U-Net with cross-attention.
- CLIP Alignment: EEG embeddings mapped to CLIP dimensions for alignment.


Results:
- Evaluation: 50-way top-1 classification task using ImageNet1K classifier.
- Ablation Study: Shows full pre-training and CLIP alignment improve accuracy.
- Optimal Mask Ratio: 0.75 maximizes performance for EEG data.


![image](https://github.com/user-attachments/assets/9e16fafd-f646-4c92-b026-18f6e0a58469)

- For detailed Explanation visit [here](https://github.com/mahdi-zade/EEG-AI-Image_reconstruction/blob/main/Dream%20Fusion.md)
- Original Github [repository](https://github.com/bbaaii/DreamDiffusion).

------------------------------------------------------------
What if we wanted to use GAN's?
Tirupattur et al. [ThoughtViz](https://www.crcv.ucf.edu/papers/acmmm18/thoughtviz.pdf) proposed a GAN network that learns from a small-size dataset [22 ](https://github.com/SforAiDl/neuroscience-ai-reading-course/blob/master/Divisha_2017A7PS0959G/Imagined_Speech_Classification_Using_EEG/Envisioned_speech_recognition_using_EEG_sensors.md). They have added a trainable Gaussian layer in the network that learns mean μ and variance σ of the EEG feature, preventing the discriminator network from overfitting. 
The work by Mishra et al. [NeuroGAN](https://link.springer.com/article/10.1007/s00521-022-08178-1) uses an attention-based GAN network along with a trainable Gaussian layer for synthesizing images from small-size EEG dataset [22 ](https://github.com/SforAiDl/neuroscience-ai-reading-course/blob/master/Divisha_2017A7PS0959G/Imagined_Speech_Classification_Using_EEG/Envisioned_speech_recognition_using_EEG_sensors.md). 
Both the work [ 41 , 22] use the pre-trained image classification network for training the generator in GAN. 
In contrast, the work by Singh et al. [ 38 ](https://arxiv.org/abs/2302.10121) uses a metric learning-based approach for feature EEG extraction and modifies the GAN training strategy to use Differentiable Data Augmentation (DiffAug) [ 46](https://arxiv.org/abs/2006.10738) method for overcoming the problem of small-size EEG dataset. 
This also reduces the network complexity, i.e., a trainable Gaussian layer and a pre-trained image encoder are not required for training the generator in GAN.

