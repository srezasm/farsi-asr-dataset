# Persian STT & TTS Data Collection

This repository is dedicated to collecting Persian speech datasets for automatic speech recognition (ASR) and text-to-speech (TTS) tasks.
The project aims to provide the codes that have been utilized to collect the **[Farsi Voice Dataset🤗](https://huggingface.co/datasets/srezas/fa_voice_dataset).**

## 🎯 Project Overview
Currently, the repository includes:
- **YouTube Data Collection Notebook**: A Colab notebook that crawls specific YouTube channels, downloads audio, extracts subtitles, and stores the data in a structured format on a HuggingFace🤗 repository.

## Current Resources

### 1. **YouTube Data Collection Notebook**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bmBiS1Xg4AvX4qyv0oewSccmdyvTTBR-?usp=sharing)

- Crawls specified YouTube channels.
- Downloads high-quality audio along with corresponding manually created subtitle.
- Batches up the downloaded files in tar.gz files and stores them in [HuggingFace🤗](https://huggingface.co/datasets/srezas/fa_voice_dataset) repository.

## 🚀 Contribution

Please feel free to contribute to improve and expand this dataset! Here’s how you can help:

- **Run the existing scripts**: Execute the Colab notebooks to expand the dataset and submit a pull request.
- **Suggest new data sources**: Open an issue and mention YouTube channels with high-quality subtitles or other open sources of Persian speech data.
- **Improve data processing**: Help refine cleaning, filtering, and segmentation methods.
