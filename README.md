# Awesome Human-Centered AI (HCAI)

## Research Papers

* [Data collection and annotation](#data-collection-and-annotation)
* [Weak supervision and self supervision](#weak-supervision-and-self-supervision)
* [Explainable ML](#explainable-ml)
* [Debate on explainable ML](#debate-on-explainable-ml)
* [AI bias (Dataset bias and Algorithmic bias)](#ai-bias-dataset-bias-and-algorithmic-bias)
* [AI robustness](#ai-robustness)
* [Human-AI Collaboration](#human-ai-collaboration)
* [Human-AI Creation](#human-ai-creation)
* [Human-in-the-loop autonomy](#human-in-the-loop-autonomy)
* [Superhuman AI and knowledge generation](#superhuman-ai-and-knowledge-generation)


### Data collection and annotation

- [Crowdsourcing in Computer Vision](https://arxiv.org/abs/1611.02145), Foundations and Trends in Computer Graphics and Vision, 2016
- [Efficient Interactive Annotation of Segmentation Datasets with Polygon-RNN++](https://arxiv.org/abs/1803.09693), CVPR, 2018 | [code](http://www.cs.toronto.edu/polyrnn/)
- [RoboTurk: A Crowdsourcing Platform for Robotic Skill Learning through Imitation](https://arxiv.org/abs/1811.02790), CoRL, 2018 | [code](https://roboturk.stanford.edu/)
- [From ImageNet to Image Classification: Contextualizing Progress on Benchmarks](https://arxiv.org/abs/2005.11295), ICML, 2020 | [code](https://github.com/MadryLab/ImageNetMultiLabel)

### Weak supervision and self supervision

- [What's the Point: Semantic Segmentation with Point Supervision](https://arxiv.org/abs/1506.02106), ECCV, 2016 | [code](https://github.com/abearman/whats-the-point1)
- [Revisiting Unreasonable Effectiveness of Data in Deep Learning Era](https://arxiv.org/abs/1707.02968), ICCV, 2017 
- [Weakly Supervised Object Localization Papers](https://github.com/xiaomengyc/Weakly-Supervised-Object-Localization)
- [TS-CAM: Token Semantic Coupled Attention Map for Weakly Supervised Object Localization](https://arxiv.org/abs/2103.14862), ICCV, 2021 | [code](https://github.com/vasgaowei/TS-CAM)
- [DiscoBox: Weakly Supervised Instance Segmentation and Semantic Correspondence from Box Supervision](https://arxiv.org/abs/2105.06464), ICCV, 2021 | [code](https://github.com/NVlabs/DiscoBox)
- [Universal Weakly Supervised Segmentation by Pixel-to-Segment Contrastive Learning](https://arxiv.org/abs/2105.00957), ICLR, 2021 | [code](https://github.com/twke18/SPML)
- [Language-driven Semantic Segmentation](https://arxiv.org/abs/2201.03546), ICLR, 2022 | [code](https://github.com/isl-org/lang-seg)
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377), CVPR, 2022 | [code](https://github.com/facebookresearch/mae)

### Explainable ML

- [Understanding the role of individual units in a deep neural network](https://www.pnas.org/doi/10.1073/pnas.1907375117), PNAS, 2020 | [code](https://github.com/davidbau/dissect)
- [Feature Visualization](https://distill.pub/2017/feature-visualization/), Distill, 2017
- [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/), Distill, 2018
- [CAM, and many CAM variants (Grad-CAM, Grad-CAM++, Score-CAM, Ablation-CAM and XGrad-CAM)](https://github.com/jacobgil/pytorch-grad-cam)
- [Towards Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129), NeurIPS, 2019 | [code](https://github.com/amiratag/ACE)

### Debate on explainable ML

- [The Mythos of Model Interpretability](https://arxiv.org/abs/1606.03490), ICML, 2016
- [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292), NeurIPS, 2018 | [code](https://github.com/adebayoj/sanity_checks_saliency)
- [On the importance of single directions for generalization](https://arxiv.org/abs/1803.06959), ICLR, 2018 | [code](https://github.com/toshalpatel/Single-Directions)
- [Revisiting the Importance of Individual Units in CNNs via Ablation](https://arxiv.org/abs/1806.02891), arXiv, 2018
- [Towards falsifiable interpretability research](https://arxiv.org/abs/2010.12016), NeurIPS, 2020 
- [The false hope of current approaches to explainable artificial intelligence in health care](https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00208-9/fulltext), The Lancet, 2021
- [Post hoc Explanations may be Ineffective for Detecting Unknown Spurious Correlation](https://openreview.net/forum?id=xNOVfCCvDpM), ICLR, 2022

### AI bias (Dataset bias and Algorithmic bias)

- [Unbiased Look at Dataset Bias](https://people.csail.mit.edu/torralba/publications/datasets_cvpr11.pdf), CVPR, 2011
- [Women also Snowboard: Overcoming Bias in Captioning Models](https://arxiv.org/abs/1803.09797), ECCV, 2018
- [Moving beyond “algorithmic bias is a data problem”](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8085589/), Patterns, 2021
- [Gender Shades: Intersectional Accuracy Disparities in Commercial Gender Classification](https://proceedings.mlr.press/v81/buolamwini18a/buolamwini18a.pdf)
- [Algorithmic bias detection and mitigation: Best practices and policies to reduce consumer harms](https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/)

### AI robustness

- [Physical Adversarial Examples for Object Detectors](https://arxiv.org/abs/1807.07769), USENIX WOOT 2018
- [Towards Robust LiDAR-based Perception in Autonomous Driving: General Black-box Adversarial Sensor Attack and Countermeasures](https://arxiv.org/abs/2006.16974), USENIX Security 2020
- [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175), NeurIPS, 2019 | [code](https://github.com/MadryLab/robustness)
- [Benchmarking Neural Network Robustness to Common Corruptions and Perturbations](https://arxiv.org/abs/1903.12261), ICLR, 2019 | [code](https://github.com/hendrycks/robustness)
- [Noise or Signal: The Role of Image Backgrounds in Object Recognition](https://arxiv.org/abs/2006.09994), ICLR, 2021 | [code](https://github.com/MadryLab/backgrounds_challenge)
- [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231), ICLR, 2019 | [code](https://github.com/rgeirhos/Stylized-ImageNet)
- [Learning Independent Causal Mechanisms](https://arxiv.org/abs/1712.00961), ICML, 2018 | [code](https://github.com/kevtimova/licms)

### Human-AI Collaboration

- [Human-Centered Tools for Coping with Imperfect Algorithms during Medical Decision-Making](https://arxiv.org/abs/1902.02960), CHI, 2019
- [Human–computer collaboration for skin cancer recognition](https://www.nature.com/articles/s41591-020-0942-0), Nature Medicine, 2020
- [To Trust or to Think: Cognitive Forcing Functions Can Reduce Overreliance on AI in AI-assisted Decision-making](https://arxiv.org/abs/2102.09692), ACM HCI, 2021
- [OpenAI Codex](https://openai.com/blog/openai-codex/)

### Human-AI Creation

- [Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291), CVPR, 2019 | [code](https://github.com/NVlabs/SPADE)
- [GANSpace: Discovering Interpretable GAN Controls](https://arxiv.org/abs/2004.02546), NeurIPS, 2020 | [code](https://github.com/harskish/ganspace)
- [Sketch Your Own GAN](https://arxiv.org/abs/2108.02774), ICCV, 2021 | [code](https://github.com/PeterWang512/GANSketching)
- [LatentCLR: A Contrastive Learning Approach for Unsupervised Discovery of Interpretable Directions](https://arxiv.org/abs/2104.00820), ICCV, 2021 | [code](https://github.com/catlab-team/latentclr)
- [LayoutGAN: Synthesizing Graphic Layouts With Vector-Wireframe Adversarial Networks](https://ieeexplore.ieee.org/document/8948239), TPAMI, 2021
- [Closed-Form Factorization of Latent Semantics in GANs](https://arxiv.org/abs/2007.06600), CVPR, 2021 | [code](https://github.com/genforce/sefa)
- [Make-A-Scene: Scene-Based Text-to-Image Generation with Human Priors](https://arxiv.org/abs/2203.13131), arXiv, 2022 | [code](https://github.com/CasualGANPapers/Make-A-Scene)

### Human-in-the-loop autonomy

- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741), NeurIPS, 2017 | [code](https://github.com/HumanCompatibleAI/imitation)
- [Shared Autonomy via Deep Reinforcement Learning](https://arxiv.org/abs/1802.01744), RSS, 2018 | [code](https://github.com/rddy/deepassist)
- [Understanding RL Vision](https://distill.pub/2020/understanding-rl-vision/), Distill, 2020
- [Learning from Interventions: Human-robot interaction as both explicit and implicit feedback](http://www.roboticsproceedings.org/rss16/p055.pdf), RSS, 2020 
- [PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training](https://arxiv.org/abs/2106.05091), ICML, 2021 | [code](https://sites.google.com/view/icml21pebble)
- [Pragmatic Image Compression for Human-in-the-Loop Decision-Making](https://arxiv.org/abs/2108.04219), NeurIPS, 2021 | [code](https://github.com/rddy/pico)
- [Recent advances in leveraging human guidance for sequential decision-making tasks](https://link.springer.com/article/10.1007/s10458-021-09514-w?noAccess=true)
- [Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization ](https://openreview.net/forum?id=0cgU-BZp2ky), ICLR, 2022

### Superhuman AI and knowledge generation

- [Grandmaster level in StarCraft II using multi-agent reinforcement learning](https://www.nature.com/articles/s41586-019-1724-z), Nature, 2019
- [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/abs/2111.09259), arXiv, 2021
- [Outracing champion Gran Turismo drivers with deep reinforcement learning](https://www.nature.com/articles/s41586-021-04357-7), Nature, 2022


## HCAI Courses

- HCAI for CV
  - [Human-Centered AI for Computer Vision and Machine Autonomy, UCLA](https://bruinlearn.ucla.edu/courses/129743) 
- HCAI for NLP
  - [Human-Centered Machine Learning, UChicago](https://github.com/ChicagoHAI/human-centered-machine-learning)
- HCAI for Human-Computer Interaction (HCI)
  - [Human-AI Interaction, CMU](https://haiicmu.github.io/) 
  - [Human-Computer Interaction, UCLA](https://uclahci.notion.site/2022-Winter-ECE-209AS-Human-Computer-Interaction-2a570caf309b49c1b5192bdb1f766d15)


## HCAI Workshop

- [HCAI, NeurIPS2021](https://sites.google.com/view/hcai-human-centered-ai-neurips/home)
- [Human-Centered Explainable AI (HCXAI), CHI 2022](https://hcxai.jimdosite.com/)
- [Human-Centered AI for Computer Vision, CVPR 2022](https://human-centeredai.github.io/)


## Acknowledgement

In this repo, currrent module list and paper list are based on [Bolei Zhou's HCAI course at UCLA](https://boleizhou.github.io/teaching/), feel free to add any new papers or modules!
