
# Transfer-Learning with ResNet for Pathology Classification 

## Group Information
- **Team Name:** N


- **Team Members:**
  
 |   Name |   Student ID    |  User    |
 |---|---|
| Luiza Matan | 40212529 | LuizaMatan |
| Nicolas Moscato-Barbeau | 40210244 | moscaton |
| Valentyna Tsilinchuk| 40046092 | tinatsl |
| Julia Spinney| 40168091 | jspin2 |
| Gulnoor Kaur| 40114998 | gul2223 |

## Project Overview
Among deep learning methodologies, transfer learning serves as an efficient way of addressing data scarcity by reusing knowledge from one domain to another. Pre-trained CNN encoders can generalize to target datasets and extract meaningful features from novel inputs, thus eliminating the need for full retraining. 

The key challenge posed by encoder transferability is ensuring that the feature extraction from the source to the target data is generalizable and robust. The domain shift can cause performance losses if the transferred feature representations are irrelevant to the new input, or if the model is too complex to transfer knowledge.  In this project, we assess the effectiveness of a pre-trained encoder in generalising to new data, comparing its performance on data from similar and different domains as well as evaluating this approach against ImageNet fine-tuning. 

## Project Parts
The project is divided into two parts:
1. **Data Transformation and ResNet18 Training on Dataset 1**
2. **Encoder Transfer and Evaluation on Dataset 2 and 3**


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

### Obtaining the Datasets

Versions of the datasets used in the project as well as full datasets can be obtained at the following links.

 * [Dataset 1: Colorectal Cancer Classification](https://zenodo.org/records/1214456) : the original dataset containing 100,000 histological images of human colorectal cancer and healthy tissue. To get the project version, please visit the [Project Dataset link](https://onedrive.live.com/?authkey=%21ADmb8ZdEzwFMZoo&id=FB338EA7CF297329%21405133&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp) and click "Download". 
 * [Dataset 2: Prostate Cancer Classification](https://zenodo.org/records/4789576) : collection of 6 datasets of digital pathology and artifacts. To get the project version, please visit the [Project Dataset link](https://onedrive.live.com/?authkey=%21APy4wecXgMnQ7Kw&id=FB338EA7CF297329%21405132&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp) and click "Download". 
 * [Dataset 3: Animal Faces Classification](https://www.kaggle.com/datasets/andrewmvd/animal-faces) : the original dataset of 3 classes of animals. To get the project version, please visit the [Project Dataset link](https://onedrive.live.com/?authkey=%21AKqEWb1GDjWPbG0&id=FB338EA7CF297329%21405131&cid=FB338EA7CF297329&parId=root&parQt=sharedby&o=OneUp) and click "Download".

Alternatively, these datasets can be found in the "data" repository as 1, 2, and 3 respectively. 

### Model Training 

> ⚠️ **Warning:** *Certain programs below permanenty alter directories by deleting files.
> Make sure folder paths are specified correctly before running.*

* [program_name.py](#)
  * **Description:** Desribe what the program does.

### Model Validation

* [program_name.py](#)
  * **Description:** Desribe what the program does.

## Task 2: Encoder Transfer 

### Model Evaluation on Test Dataset 

* [program_name.py](#)
  * **Description:** Desribe what the program does.
 
### Execution Instructions

The code can be executed in the PyCharm IDE, or in Colab as follows:

```sh
PS C:\PATH_TO_PROJECT> python 'program_name.py'
```

