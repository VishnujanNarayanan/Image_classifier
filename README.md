# Age & Gender Classifier

This project is a deep learning model that predicts **age** and **gender** from facial images using the [UTKFace dataset](https://susanqq.github.io/UTKFace/). The dataset contains over 20,000 labeled face images with annotations for age, gender, and ethnicity.

## Project Structure

- `Image_Classifier!.ipynb` – Main Jupyter Notebook with model training and evaluation.
- `UTKFace/` – Dataset containing face images.
- `README.md` – Project documentation.

## Features

- Predicts **age** (0–100 years).
- Classifies **gender** (male/female).
- Implements a **Convolutional Neural Network (CNN)** with multi-task learning.
- Uses preprocessing (face alignment, resizing, normalization).
- Supports Label Distribution Learning for more accurate age estimation.

## Requirements

- Python 3.8+
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Age_Gender_Classifier.git
   cd Age_Gender_Classifier
   ```

2. Place the `UTKFace` dataset in the project directory.

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "Image_Classifier!.ipynb"
   ```

4. Run all cells to train and evaluate the model.

## Results

- Achieves good accuracy on gender classification.
- Provides reasonable age predictions with improved performance using **Earth Mover’s Distance (EMD)** loss.

## Future Improvements

- Experiment with deeper CNN architectures.
- Add dropout and batch normalization.
- Implement model ensembling for better predictions.
- Deploy as a web app for real-time inference.

## License

This project is released under the MIT License.
