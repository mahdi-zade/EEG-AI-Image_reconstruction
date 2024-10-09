# EEG_AI_Image-reconstruction

This page is an analysis and summarization of two papers that attempt to reconstruct images from EEG brain recordings.
We are living in an era that the AI and EEG are intersecting and we can decode EEG recordings better than ever and are able to obtain valuable insights and results from it.
This page is a part of a deep dive into one of the many applications of EEG-AI. If you're interested in knowing more about all the other applications of EEG-AI, please visit [EEG-AI Applications Hub](https://github.com/Avir-AI/EEG_Applications_Hub).

--------------------------------------------------------------------------

| Model Name | Input Type | Output Type | Preprocessing | Models Used | Decoding Strategy | Key Features | Performance |
|------------|------------|-------------|---------------|-------------|-------------------|--------------|-------------|
| [MinD-Vis](https://arxiv.org/pdf/2211.06956)   | fMRI Data  | Reconstructed Images | Masked Signal Modeling (Sparse Masking) | Latent Diffusion Model (LDM), Self-Supervised Representation| Double Conditioning to enforce decoding consistency | Sparse-coded masked brain modeling | Outperformed state-of-the-art by 66% in semantic mapping, 41% in generation quality (FID) |
| [Brain2Image](https://web.njit.edu/~usman/courses/cs698_fall19/Brain2Image_%20Converting%20Brain%20Signals%20into%20Images.pdf)   | EEG Data    | Reconstructed Images | Noise-free representation using LSTM | LSTM, GAN, VAE | Latent space learned from EEG signals | Generative model generates visual samples semantically coherent with stimuli | GAN: better sharpness, VAE: less realistic images |
| [ThoughtViz](https://www.crcv.ucf.edu/papers/acmmm18/thoughtviz.pdf)    | EEG Data          | Reconstructed Images | EEG signal encoding      | Conditional GAN            | Latent space learned from EEG | Conditional GAN generates class-specific images based on thoughts, learns distribution from limited data    | Effective on digits, characters, and object datasets
| [Brain-Supervised Image Editing](https://openaccess.thecvf.com/content/CVPR2022/papers/Davis_Brain-Supervised_Image_Editing_CVPR_2022_paper.pdf) | EEG Data           | Edited Images        | Brain response encoding     | Generative Adversarial Network (GAN) | Latent space learning via brain responses | Uses implicit brain responses as supervision for learning semantic features and editing images | Comparable performance to manual labeling for semantic editing |
| [DreamDiffusion](https://arxiv.org/pdf/2306.16934)  | EEG Signals    | Generated Images| Temporal masked signal modeling | Stable Diffusion, CLIP      | Image generation from EEG signals | Leverages pre-trained text-to-image models for generating images directly from EEG, with CLIP for embedding alignment | Promising results with high-quality images, overcoming EEG signal challenges |
| [Learning Robust Deep Visual Representations from EEG Brain Recordings](https://arxiv.org/abs/2310.16532)   | EEG Signals      | Generated Images | Supervised and Contrastive learning methods | EEG feature extractor, GAN (StyleGAN-ADA) | Image synthesis from EEG signals | Improves image synthesis quality from EEG by using learned EEG representations in contrastive settings | Achieved 62.9% and 36.13% inception score improvement on EEGCVPR40 and ThoughtViz datasets |

--------------------------------------------------------------------------
 
[DreamDiffusion: Generating High-Quality Images from Brain EEG Signals](https://arxiv.org/pdf/2306.16934)

- Inputs: EEG, original image
- Output: Image saliency map

Steps:
1. Preprocessing
2. Encoding the EEG and Image data into a common space. 
3. Training the encoders as a classification problem based on the definition of a compatibility function. The encoders are trained to maximize the similarity between the embeddings of corresponding images and EEGs. This aims to capture the relationship between what a person is seeing and their brain activity.
4. Analyzing the saliency score of each pixel. Once trained, these encoders analyze how the compatibility between the EEG and image embeddings changes when different regions of the image are suppressed. Regions whose removal causes significant variations in compatibility are considered salient, resulting in a visual saliency map.

![image](https://github.com/user-attachments/assets/9e16fafd-f646-4c92-b026-18f6e0a58469)


- For detailed Explanation visit [here](https://github.com/bbaaii/DreamDiffusion) 

------------------------------------------------------------
#### [Visual Saliency and Image Reconstruction from EEG Signals via an Effective Geometric Deep Network-Based Generative Adversarial Network:](https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://ieeexplore.ieee.org/document/9320159&ved=2ahUKEwig94aDu-yHAxX0TKQEHTeMAf0QFnoECBMQAQ&usg=AOvVaw3ztGyfYLu_r6N34G0MAfA4)

- Inputs: EEG, Ground truth obtained by eye tracking
- Output: Image saliency map and full reconstructed image

steps:
1. Turn EEG data into a graph representation
2. Extract the features of the graph using unsupervised method
3. Train the GAN model based on the extracted featues in the last step and ground truth input and obtain the image saliency map and full reconstructed image

![steps](https://github.com/user-attachments/assets/5aba1387-f8f6-43e5-943f-eed0f37e522f)

Let's take a deep dive into each step:

1. Preprocessing: The first part is obtaining the functional connectivity-based graph representation of the EEG channels. This is done by **Chebyshev graph convolutional layers**.

2. Unsupervised Feature Extraction: **Geometric Deep Network (GDN)** is an unsupervised method that takes the graph representation of EEG channels as input, extracts discriminative features from the graph to categorize the EEG signals into different visual patterns.

3. Training: **Generative Adversarial Network (GAN) for Saliency and Image Reconstruction:** The features extracted by the GDN are fed into the GAN. This GAN consists of a generator and a discriminator. The generator aims to create a saliency map from the EEG features, while the discriminator tries to distinguish between real saliency maps (derived from eye-tracking data) and those generated by the generator using **saliency metrics**. Through this adversarial process, the generator learns to produce accurate saliency maps from EEG signals.
* The trained GDN-GAN can then be fine-tuned to perform image reconstruction from the EEG signals, going beyond just saliency detection.

- For detailed Explanation visit [here](https://github.com/ab-mahdi/EEG-AI-Saliency/blob/main/GDN-GAN.md) 



