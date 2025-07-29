# 🧠 Text-to-3D Model Generator with Shap-E

This project uses OpenAI’s [Shap-E](https://github.com/openai/shap-e) to generate 3D models from text prompts using a diffusion-based approach. You can enter any prompt and get a 3D asset that you can render, view, and download.

## 🚀 Features

- Generate 3D models from natural language prompts
- Use diffusion models with guidance scale and Karras sampling
- GPU acceleration (tested on Kaggle and Colab)
- Flask-based API coming soon for easy web interaction
- Download generated `.ply` model files

---

## 🛠️ Installation
- Clone the repository and run python setup.py install to install all the required packages. Also make sure that the packages in requirements.txt are already existing in your enviornment.
### Clone the repository

```bash
git clone https://github.com/KurooNxtsu/model_generator.git
cd model_generator
