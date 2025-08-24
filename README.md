# Military Aerial Vehicle Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://huggingface.co/spaces/MSarvesh2K6/Military-Aircrafts-Classifier)

### **[>> Try the Live Interactive Demo <<](https://huggingface.co/spaces/MSarvesh2K6/Military-Aircrafts-Classifier)**

---

## Project Overview

This project is a high-accuracy, deep learning model capable of classifying **40 different military aerial vehicles**, including fighter jets, bombers, helicopters, and drones. The model was built using TensorFlow and Keras, leveraging a fine-tuned `EfficientNetB3` architecture to achieve a final **test accuracy of 91.62%**.

The project showcases a complete, end-to-end machine learning workflow, from data curation and preprocessing to iterative model improvement and final deployment as an interactive web application.

## Visual Demo

A screenshot of the interactive Streamlit application in action.

![Streamlit App Demo](https://github.com/SARVESH2K6/Military-Aircraft-Classifier/blob/7fb1b7333025cd9abbd39375faf5a02850553332/Screenshot%20of%20App.png)

## Key Features

- **High Accuracy:** Achieved **91.62% accuracy** on the unseen test set.
- **State-of-the-Art Model:** Utilizes a fine-tuned `EfficientNetB3`, a powerful and efficient convolutional neural network.
- **Robust Training:** Employs data augmentation and standard callbacks like Early Stopping and Model Checkpointing to prevent overfitting and ensure generalization.
- **Interactive Application:** A user-friendly web app built with Streamlit that allows for real-time predictions, featuring a confidence threshold slider and a top-5 prediction analysis.
- **Curated Dataset:** The model is trained on a focused, high-quality dataset of 40 distinct aerial vehicle classes, refined from a larger original dataset.

## The Journey: From Baseline to High-Performance Model

This project followed a deliberate, iterative process to move beyond a simple baseline and build a truly robust model.

1.  **Baseline Model:** An initial model was trained, achieving a promising **82% validation accuracy**.
2.  **Real-World Testing & Diagnosis:** To test its true capabilities, the baseline model was evaluated on new, "in-the-wild" images. This test revealed a performance drop to approximately **47% accuracy**, diagnosing a classic overfitting problem.
3.  **Solution & Iteration:** To solve this, the model was retrained with **data augmentation** (`RandomFlip`, `RandomRotation`, `RandomZoom`) and robust callbacks like `EarlyStopping` and `ReduceLROnPlateau`.
4.  **Final Results:** This iterative process was highly successful, resulting in a final model with **91.62% accuracy** on the unseen test set, demonstrating a vast improvement in real-world performance.

## Dataset Curation

The original dataset from Kaggle contained 88 classes with varying image quality and sample sizes. To build a more focused and reliable classifier, the dataset was curated to **40 distinct and well-represented aircraft classes**. This crucial data preparation step ensured that the model was trained on a high-quality, balanced dataset, which was essential for achieving the final high accuracy. The full list of selected classes can be found in the training notebook.

## Technologies Used
- **Python**
- **TensorFlow & Keras** for model building and training
- **Streamlit** for the interactive web application
- **Hugging Face Spaces** for deployment
- **Pandas & NumPy** for data manipulation
- **Scikit-learn** for evaluation metrics
- **Matplotlib & Seaborn** for data visualization
- **Kaggle** for the training environment

## Running the Demo

### Live Demo (Recommended)
The easiest way to use the model is via the **live Streamlit demo** linked at the top of this README.

### Local Setup Instructions
To run the application on your local machine, follow these steps:
```bash
# 1. Clone the repository
git clone https://github.com/SARVESH2K6/Military-Aircraft-Classifier.git
cd Military-Aircraft-Classifier

# 2. Install the required dependencies
pip install -r requirements.txt

# 3. Download the model weights file (see section below) and place it in this folder.

# 4. Run the Streamlit app
streamlit run app.py
```

## Project Files

* **`app.py`:** The Python script for the Streamlit web application.
* **`military-aircraft-detection-model.ipynb`:** The complete Jupyter Notebook with the data processing, model training, and evaluation process.
* **`class_indices.json`:** The mapping from class labels to model output indices.
* **`requirements.txt`:** A list of necessary Python packages for reproducibility.
* **`model.weights.h5`:** The trained weights for the final high-performance model.

## Model & Dataset Source

### Model Weights
The trained model weights file (`model.weights.h5`) can be downloaded directly from the "Output" section of the Kaggle notebook.

* **Direct Download:** [Kaggle Notebook with Final Model](https://www.kaggle.com/code/msarvesh2k6/military-aircraft-detection-model-91-6-accuracy)

### Dataset
This project utilizes the Military Aircraft Dataset, sourced from Kaggle. Proper credit goes to the original creator of the dataset.

* **Source:** [Military Aircraft Detection Dataset on Kaggle](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)
* **License:** CC0: Public Domain


