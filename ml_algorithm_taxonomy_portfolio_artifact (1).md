# Portfolio Artifact: Machine Learning Algorithm Taxonomy & Visual Framework

**Student Name:** Nikhil

**Course Title:** Introduction to Machine Learning & Data Science

**Assignment Title:** 2.2 Portfolio Assignment — Algorithm Exploration & Visual Taxonomy

**Date:** November 1, 2026

## 1. Executive Summary & Artifact Overview

Algorithms form the core foundation of artificial intelligence and machine learning, defining how computational systems extract patterns, infer relationships, and generate predictions from raw data.

This portfolio artifact provides a comprehensive structural framework categorizing **10 fundamental machine learning algorithms** across their primary **Learning Style** (*Supervised* vs. *Unsupervised*) and their major **Application Domains** (*Tabular Data*, *Computer Vision*, *Natural Language Processing (NLP)*, and *Generative AI*).

## 2. Interactive Visual Taxonomy Framework

The structural layout below maps the structural relationships, learning paradigms, and operational domains across machine learning:

```
                                  [ MACHINE LEARNING ALGORITHM TAXONOMY ]
                                                     |
                 +-----------------------------------+-----------------------------------+
                 |                                                                       |
      [ SUPERVISED LEARNING ]                                                 [ UNSUPERVISED LEARNING ]
                 |                                                                       |
  +--------------+--------------+                                         +--------------+--------------+
  |                             |                                         |                             |
[TABULAR & STRUCTURAL]   [VISION & SEQUENCE]                           [CLUSTERING & DIM.]      [GENERATIVE & SYNTHESIS]
  |                             |                                         |                             |
  +-- Linear Regression         +-- Convolutional Neural Networks (CNN)   +-- K-Means Clustering        +-- Generative Adversarial
  |   (Continuous)              |   (Spatial / Vision)                    |   (Centroid Partition)        |   Networks (GANs)
  |                             |                                         |                             |   (Adversarial Synthesis)
  +-- Logistic Regression       +-- Recurrent Neural Networks (RNN/LSTM)  +-- Principal Component       |
  |   (Binary Classification)   |   (Sequential / NLP)                    |   Analysis (PCA)              +-- Variational Autoencoders
  |                             |                                         |   (Dimensionality Reduc.)     |   (VAEs)
  +-- Decision Trees &          +-- Transformer Architecture              |                               |   (Probabilistic Latent Space)
  |   Random Forests            |   (Attention / GenAI Basis)             +-- Isolation Forest          |
  |   (Non-linear Classification)                                         |   (Anomaly Detection)         +-- Diffusion Models
                                                                                                        |   (Iterative Denoising)

```

## 3. Comprehensive Algorithm Specifications

| **#** | **Algorithm Name** | **Learning Paradigm** | **Primary Application Domains** | **Technical Mechanics & Mathematical Basis** | **Practical Industry Use Case** | 
| **1** | **Linear Regression** | Supervised | Tabular Data | Minimizes Mean Squared Error (MSE) to fit an optimal linear hyperplane ($\hat{y} = \beta_0 + \sum \beta_i x_i$) through continuous feature space. | **Real Estate Valuation:** Forecasting residential property sales prices using physical metrics (square footage, room count, location rating). | 
| **2** | **Logistic Regression** | Supervised | Tabular Data, NLP | Passes linear decision boundaries through a sigmoid logit activation function ($\sigma(z) = \frac{1}{1 + e^{-z}}$) to compute output probabilities bounded between $0$ and $1$. | **Financial Risk:** Predicting binary credit default probability (Pass/Fail) and medical diagnostics (Positive/Negative detection). | 
| **3** | **Decision Trees & Random Forests** | Supervised | Tabular Data | Recursively partitions features using orthogonal splits that maximize Information Gain or reduce Gini Impurity. Ensembles aggregate trees to minimize variance. | **Customer Analytics:** Predicting customer churn in telecommunications and automated loan risk scoring. | 
| **4** | **Convolutional Neural Networks (CNN)** | Supervised | Computer Vision, Medical Imaging | Applies learnable spatial cross-correlation kernels and pooling layers to extract shift-invariant spatial hierarchies (edges, textures, shapes). | **Automotive & Health:** Real-time object recognition in autonomous navigation systems and medical image diagnostic screening. | 
| **5** | **Recurrent Neural Networks (RNN / LSTM)** | Supervised | NLP, Time-Series Forecasting | Retains temporal sequential state ($h_t$) across time steps. Gated variants (LSTM/GRU) utilize memory cells to eliminate vanishing gradients. | **Speech & Finance:** Automated speech recognition (Voice-to-Text) and high-frequency stock price time-series modeling. | 
| **6** | **Transformer Architecture** | Supervised / Self-Supervised | NLP, Generative AI, Computer Vision | Replaces recurrence with parallelized Scaled Dot-Product Self-Attention ($\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$), tracking contextual dependencies across arbitrary token distances. | **Large Language Models:** Powering conversational AI (ChatGPT, Claude), neural machine translation, and automated code generation. | 
| **7** | **K-Means Clustering** | Unsupervised | Tabular Data, Customer Segmentation | Iteratively partitions data into $K$ distinct clusters by minimizing Euclidean distance between samples and cluster centroids until convergence. | **E-Commerce:** Segmenting user purchasing behaviors to construct targeted marketing cohorts. | 
| **8** | **Principal Component Analysis (PCA)** | Unsupervised | Tabular Data, Computer Vision | Computes orthogonal linear projections (eigenvectors) of the covariance matrix to maximize variance in lower-dimensional space. | **Biotechnology:** Dimensionality reduction on high-throughput gene expression data and visual noise filtering. | 
| **9** | **Generative Adversarial Networks (GANs)** | Unsupervised / Adversarial | Generative AI, Computer Vision | Trains a Generator (producing synthetic data) against a Discriminator (evaluating real vs. synthetic) in a zero-sum minimax game ($\min_G \max_D V(D,G)$). | **Digital Media:** High-resolution photorealistic portrait synthesis and domain-to-domain image translation. | 
| **10** | **Diffusion Models** | Unsupervised / Generative | Generative AI, Computer Vision | Adds gradual Gaussian noise in a forward process, then trains a neural network (U-Net) to learn the reverse iterative denoising process from random noise to sharp outputs. | **Creative AI:** Text-to-image synthesis frameworks (Midjourney, Stable Diffusion, DALL-E). | 

## 4. Cross-Domain Suitability Matrix

This matrix details model applicability across all operational domains:

```
+------------------------------------+--------------+-----------------+-------------------+---------------+
| Algorithm                          | Tabular Data | Computer Vision | Nat. Lang. Proc.  | Generative AI |
+------------------------------------+--------------+-----------------+-------------------+---------------+
| Linear Regression                  |  [ Primary ] |  [    --    ]   |  [    --    ]     |  [    --    ] |
| Logistic Regression                |  [ Primary ] |  [    --    ]   |  [ Secondary ]    |  [    --    ] |
| Decision Trees / Random Forests     |  [ Primary ] |  [    --    ]   |  [    --    ]     |  [    --    ] |
| Convolutional Neural Nets (CNN)    |  [    --    ] |  [ Primary ]    |  [    --    ]     |  [ Secondary ]|
| Recurrent Neural Nets (RNN/LSTM)   |  [ Secondary]|  [    --    ]   |  [ Primary ]      |  [    --    ] |
| Transformer Architecture           |  [ Secondary]|  [ Secondary ]  |  [ Primary ]      |  [ Primary ]  |
| K-Means Clustering                 |  [ Primary ] |  [ Secondary ]  |  [    --    ]     |  [    --    ] |
| Principal Component Analysis (PCA) |  [ Primary ] |  [ Secondary ]  |  [    --    ]     |  [    --    ] |
| Generative Adversarial Nets (GANs) |  [    --    ] |  [ Secondary ]  |  [    --    ]     |  [ Primary ]  |
| Diffusion Models                   |  [    --    ] |  [ Secondary ]  |  [    --    ]     |  [ Primary ]  |
+------------------------------------+--------------+-----------------+-------------------+---------------+

```

## 5. Practical Case Study: Multimodal E-Commerce Production Architecture

To illustrate how these distinct algorithms operate in real-world environments, consider an integrated e-commerce production architecture:

```
[ Customer Product Review Text ] ----> [ Transformer (LLM) ] ----> Extracts Sentiment & Summary (NLP Domain)
                                             |
[ Raw Product Photography ]      ----> [ CNN Classifier ]     ----> Assigns Category & Style Tags (Vision Domain)
                                             |
[ User Interaction Matrix ]      ----> [ Random Forest ]      ----> Predicts User Lifetime Value (Tabular Domain)
                                             |
                                             v
[ Automated Visual Ad Generation Engine (Diffusion Model + LLM) ]

```

1. **Tabular Domain:** A **Random Forest Classifier** evaluates user transaction history, visit frequency, and cart size to segment churn risk and lifetime value.

2. **NLP Domain:** A **Transformer Model** parses text reviews to extract sentiment polarity and feature feedback.

3. **Computer Vision Domain:** A **CNN** processes raw product uploads to auto-generate taxonomy tags (e.g., footwear vs. apparel).

4. **Generative AI Domain:** A **Diffusion Model** paired with a **Transformer-based LLM** dynamically creates personalized visual marketing assets based on customer preference vectors.

## 6. Personal Learning Reflection & Portfolio Value Statement

### 6.1 Reflection on Learning Experience

Constructing this machine learning taxonomy deepened my understanding of how mathematical objectives dictate algorithm selection. Classical statistical models like Linear Regression and Random Forests excel in structured tabular settings due to low computational requirements and high interpretability. Conversely, modern deep neural networks (CNNs, Transformers) overcome feature-engineering bottlenecks on unstructured data like images and natural language text.

The exploration of Generative AI architectures (GANs, Transformers, Diffusion Models) highlighted a major paradigm shift: moving from *discriminative classification* ($P(Y\vert{}X)$) to *probabilistic distribution sampling and synthesis* ($P(X)$). Recognizing these trade-offs ensures I can select cost-effective and accurate models tailored to specific technical constraints.

### 6.2 Professional Value & Career Impact

This portfolio artifact highlights my ability to:

* **Systematically Classify AI Paradigms:** Categorize complex algorithms by learning mechanics, underlying mathematics, and operational constraints.

* **Architect Integrated Systems:** Design end-to-end multimodal solutions combining classical machine learning with advanced deep learning and generative models.

* **Bridge Technical & Executive Communication:** Translate abstract algorithmic mechanics into clear visual frameworks accessible to non-technical stakeholders.