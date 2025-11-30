# EEG-MACS: Brain Disease Diagnosis under Unreliable Annotations

:trophy: The official implementation of “EEG-MACS” accepted to ACM MM 2024 as an Oral paper (Top 3%). 

:rocket: EEG-MACS couples geometry-aware manifold attention (to capture inter-channel and temporal dependencies) with a confidence-stratified learning scheme that down-weights suspected noisy labels via consistency and uncertainty calibration, delivering robust, cross-site EEG diagnosis.

:link: [[Paper](https://dl.acm.org/doi/pdf/10.1145/3664647.3681645)] 

<p align="center">
  <img src="./MACS-Motivation.png" alt="EEG-MACS motivation" width="860">
</p>

### EEG-MACS: Manifold Attention and Confidence Stratification for EEG-based Cross-Center Brain Disease Diagnosis under Unreliable Annotations

Cross-center data heterogeneity and annotation unreliability significantly challenge the intelligent diagnosis of diseases using brain signals. A notable example is the EEG-based diagnosis of neurodegenerative diseases, which features subtler abnormal neural dynamics typically observed in small-group settings. To advance this area, in this work, we introduce a transferable framework employing **M**anifold **A**ttention and **C**onfidence **S**tratification (MACS) to diagnose neurodegenerative disorders based on EEG signals sourced from four centers with unreliable annotations. The **MACS** framework’s effectiveness stems from these features: 
1) The ***Augmentor*** generates various EEG-represented brain variants to enrich the data space;
2) The ***Switcher*** enhances the feature space for trusted samples and reduces overfitting on incorrectly labeled samples;
3) The ***Encoder*** uses the Riemannian manifold and Euclidean metrics to capture spatiotemporal variations and dynamic synchronization in EEG;
4) The ***Projector***, equipped with dual heads, monitors consistency across multiple brain variants and ensures diagnostic accuracy;
5) The ***Stratifier*** adaptively stratifies learned samples by confidence levels throughout the training process;
6) Forward and backpropagation in **MACS** are constrained by confidence stratification to stabilize the learning system amid unreliable annotations.

Our subject-independent cross-validation experiments, conducted on both neurocognitive and movement disorders using cross-center corpora, have demonstrated superior performance compared to existing related algorithms. This work not only improves EEG-based diagnostics for cross-center and small-setting brain diseases but also offers insights into extending **MACS** techniques to other data analyses, tackling data heterogeneity and annotation unreliability in multimedia and multimodal content understanding.


<p align="center">
  <img src="./MACS-Overview.png" alt="EEG-MACS overview" width="860">
</p>

### Requirements
* Python 3.8.16
* Pytorch 2.0.0
* Numpy 1.23.5
* scikit-learn 1.2.2

### Reproducing results on PD/MCI (and AD)
- **PD datasets** are publicly accessible (see the link below).  
- **MCI/AD datasets** are available **upon request** to the corresponding author after hospital IRB approval.  
- We released a **processed PD subset** at the Google Drive link below so users can run our scripts out-of-the-box.

>| Resource | Link |
>|---|---|
>| PD public data | http://predict.cs.unm.edu/ |
>| Processed PD subset | [Google Drive link](https://drive.google.com/drive/folders/1HD2lOxhpy3U9szeDI3hempHk9LTyG6AM?usp=sharing) |

