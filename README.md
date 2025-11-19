# Magic Photo: Image Restoration and Coloring

## Project Overview
Magic Photo is a project with user-friendly application designed to enhance and colorize images using machine learning and deep learning techniques. The application allows users to upload images, select a processing option (restoration or coloring), and receive the processed image as output. By leveraging pre-trained models and fine-tuning them, Magic Photo aims to restore and colorize not just images, but also the warm moments of life captured in photos.

## Features
- **Image Restoration**: Enhance images affected by various corruptions.
- **Image Coloring**: Add vibrant colors to grayscale images.
- **User Feedback Collection**: Gather user feedback for continuous improvement.
- **Focus Areas**: Faces, cities, pets, and nature for coloring; restoration focus is under discussion.


## Repository Structure
```text
Image-Restoration-and-Coloring/
├─ data_augmentation_methods/
│  ├─ data/
│  │  ├─ augmented_images/
│  │  ├─ selected_images/
│  │  ├─ train.csv
│  │  └─ test.csv
│  └─ testing.ipynb
├─ models_testing/
│  └─ restoration_superres/
│     ├─ data/
│     │  ├─ metrics_results.csv
│     │  ├─ metrics_summary.csv
│     │  ├─ restored_images/
│     │  │  ├─ city/
│     │  │  │  ├─ restored_restoration/
│     │  │  │  └─ restored_superres/
│     │  │  ├─ faces/
│     │  │  │  ├─ restored_restoration/
│     │  │  │  └─ restored_superres/
│     │  │  └─ nature/
│     │  │     ├─ restored_restoration/
│     │  │     └─ restored_superres/
│     │  └─ results_images/
│     └─ restorationtest.ipynb
├─ models_finetuning/
│  └─ real-esrgan/
│     ├─ real_esrgan_finetune.ipynb
│     └─ realesrgan_finetune.yml
├─ docs/
│  ├─ D1.1/
│  │  ├─ d11.tex
│  │  └─ d11.pdf
│  └─ D1.2/
│     ├─ d12.tex
│     └─ d12.pdf
└─ README.md
```

## Folder Descriptions
- **data_augmentation_methods/**: Implemented degradations and combined scenarios for training/evaluation datasets; includes sample CSV splits and notebook for testing methods.
- **models_testing/**: Baseline evaluation of restoration and super-resolution models (Real-ESRGAN, GFPGAN) with metrics and visual results.
- **models_finetuning/**: Notebooks/configs to fine-tune pretrained models (e.g., Real-ESRGAN) for our data and tasks.
- **docs/**: Course deliverables and progress reports.
  - D1.1: Initial project scope, existing solutions, dataset choices, and success criteria.
  - D1.2: Dataset EDA, augmentation techniques, and baseline testing results.







---
## Credentials
**Team Members**: \
Maksim Ilin (B23-DS-01),  
Rail Sabirov (B23-DS-01),  
Ivan Ilyichev (B23-DS-01)  
**Course**: Practical Machine Learning and Deep Learning, 3rd year Bachelor degree 
