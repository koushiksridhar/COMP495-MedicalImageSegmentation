# COMP495-MedicalImageSegmentation
An independent project for Medical Image Segmentation using UNET and Swin-UNETR. Built with MONAI and PyTorch libraries and trained on the BRaTS 2021 dataset. Advised by Dr. Jorge Silva.
Models were trained on the UNC Longleaf computing cluster on A100 GPUs. 

Abstract:
This paper explores UNET and Swin-UNET architectures for the task of multi-region segmentation of brain tumors using stacked modalities and combined MRI scans. UNET and Swin-UNET models were developed using MONAI and PyTorch, and transformer blocks were implemented for Swin-UNET architecture. The models were trained and evaluated on the BRaTS 2021 Task 1 Dataset. Models were evaluated based on standard metrics of Dice Score and Loss quantification. Predicted segmentation outputs were generated. The findings from this work lay a baseline for future iterations of models, and establish that further model finetuning is needed and possible with expanded training. 

### Data Citations (can be downloaded via Synapse):
[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[5] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF
