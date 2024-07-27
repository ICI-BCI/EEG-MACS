# EEG-MACS: Manifold Attention and Confidence Stratification for EEG-based Cross-Center Brain Disease Diagnosis under Unreliable Annotations

### Abstract:

Cross-center data heterogeneity and annotation unreliability significantly challenge the intelligent diagnosis of diseases using brain signals. A notable example is the EEG-based diagnosis of neurodegenerative diseases, which features subtler abnormal neural dynamics typically observed in small-group settings.To advance this area, in this work, we introduce a transferable framework employing Manifold Attention and Confidence Stratification (MACS) to diagnose neurodegenerative disorders based on EEG signals sourced from four centers with unreliable annotations.The MACS frame-work’s effectiveness stems from these features: 1) The Augmentor generates various EEG-represented brain variants to enrich the data space; 2) The Switcher enhances the feature space for trusted samples and reduces overfitting on incorrectly labeled samples; 3) The Encoder uses the Riemannian manifold and Euclidean metrics to capture spatiotemporal variations and dynamic synchronization in EEG; 4) The Projector, equipped with dual heads, monitors consistency across multiple brain variants and ensures diagnostic accuracy; 5) The Stratifier adaptively stratifies learned samples by confidence levels throughout the training process; 6) Forward and backpropagation in MACS are constrained by confidence stratification to stabilize the learning system amid unreliable annotations. Our subject-independent cross-validation experiments, conducted on both neurocognitive and movement disorders using cross-center corpora, have demonstrated superior performance compared to existing related algorithms. This work not only improves EEG-based diagnostics for cross-center and small-setting brain diseases but also offers insights into extending MACS techniques to other data analyses, tackling data heterogeneity and annotation unreliability in multimedia and multimodal content understanding.


### Requirements:
* Python 3.8.16
* Pytorch 2.0.0
* Numpy 1.23.5
* scikit-learn 1.2.2


### Running the code on MCI and PD
We have made our code available for execution on PD and MCI datasets. The PD datasets are publicly accessible and can be obtained from [http://predict.cs.unm.edu/]. Access to the MCI and AD datasets is available upon request to the corresponding author after the publication of this work, subject to approval from the collaborating hospitals. Additionally, we will uploaded the processed public PD dataset via [https://drive.google.com/drive/folders/1XIVBDAOPeI0keiH3Ap2nG-xIwPHdEXVg?usp=sharing], and users can execute our approach using the provided example scripts run.sh.

