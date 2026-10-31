# ACL Region Localization in Knee MRI

## Project Overview
This project uses a fine-tuned SegFormer model to identify the approximate ACL region in knee MRI images.

## Dataset
Knee MRI ACL Tear Slices dataset:
https://zenodo.org/records/7977297

## Pretrained Model
nvidia/segformer-b0-finetuned-ade-512-512

## Fine-Tuned Model
https://huggingface.co/AjraBaluch488/knee-mri-acl-localization

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Streamlit

## Run Locally
Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Limitation
This project demonstrates approximate ACL region localization. It is not a medical diagnostic tool.
