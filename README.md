
# Transfer-Learning with ResNet for Pathology Classification 

## Group Information
- **Team Name:** N


- **Team Members:**
  
 |   Name |   Student ID    |
 |---|---|
| Luiza Matan | 40212529 | 
| Nicolas Moscato-Barbeau | 40210244 |
| Valentyna Tsilinchuk| 40046092 |
| Julia Spinney| 40168091 |
| Gulnoor Kaur| 40114998 |

## Project Overview
Among deep learning methodologies, transfer learning serves as an efficient way of addressing data scarcity by reusing knowledge from one domain to another. Pre-trained CNN encoders can extract features from novel data, thus eliminating the need for full retraining. 

The key challenge posed by encoder transferability is ensuring that the feature extraction from the source to the target data is generalizable. The domain shift can cause performance losses if the transferred feature representations are irrelevant to the new input, or if the model is too complex to transfer knowledge.  In this project, we assess the effectiveness of a pre-trained encoder in generalising to new data, comparing its performance on data from similar and different domains as well as evaluating this approach against ImageNet fine-tuning. 

## Project Parts
The project is divided into two parts:
1. **Data Pre-Processing, Training Stage**
2. **Transfer Learning and Deep Evaluation**


---

## Task 1: Data Pre-Processing and ResNet Training

### Prerequisites

To successfully run the code, you will need the following libraries:

* pip
  ```sh
  pip install numpy
  pip install matplotlib
  pip install scikit-learn
  pip install scikit-image
  pip install opencv-python
  pip install torch, torchvision, torchsummary, skorch
  ```

To read the images in the correct format, use LFS extension:

* git
  ```sh
  git lfs install
  git lfs pull
  ```
### Data
 * [Dataset 1: Colorectal Cancer Classification](https://zenodo.org/records/1214456) : the original dataset containing 100,000 histological images of human colorectal cancer and healthy tissue.
 * [Dataset 2: Prostate Cancer Classification](https://zenodo.org/records/4789576) : collection of 6 datasets of digital pathology and artifacts.
 * [Dataset 3: Animal Faces Classification](https://www.kaggle.com/datasets/andrewmvd/animal-faces) : the original dataset of 3 classes of animals. 

### Data Cleaning Programs

> ⚠️ **Warning:** *Certain programs below permanenty alter directories by deleting files.
> Make sure folder paths are specified correctly before running.*

* [program_name.py](#)
  * **Description:** Desribe what the program does.

### Data Visualization Programs

* [program_name.py](#)
  * **Description:** Desribe what the program does.

### Model Training Programs 

* [program_name.py](#)
  * **Description:** Desribe what the program does.
 
### Execution Instructions

The code can be executed in the PyCharm IDE, or in Colab:

```sh
PS C:\PATH_TO_PROJECT> python 'program_name.py'
```

## Task 2: Transfer Learning and Deep Evaluation 

