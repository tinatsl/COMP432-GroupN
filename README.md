
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
Both computer vision and deep learning have become crucial tools in the healthcare field. The CNN-based algorithms show promising results in medical imaging, notably in pathology diagnosis.  CNN models were also robust enough to generalize among different medical concerns regarding image classification. For this project, we aim to apply the CNN-based algorithms to the various classifications of colorectal, prostate cancer and animal faces as computer vision tasks.

Cancer diagnosis, to this day, remains a milestone amongst pathology experts facing workloads that impact the efficiency and accuracy of the diagnosis. The CNN models aim to assist medical professionals with providing insights that increase accuracy and reduce human error. One of the main problems can be associated with transfer learning, where CNNs pre-trained on one dataset are applied towards feature extraction and classification on other datasets, and while this is helpful to generalize across multiple applications, it still poses the risk of overfitting and hence loss of accuracy.  


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
  pip install torch, torchvision
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

