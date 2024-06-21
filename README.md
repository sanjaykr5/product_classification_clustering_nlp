# Price Runner Product Classification & Clustering
###
This repo contains the model development.

### Overview
Price Runner Dataset has Product Id, Product Title, Merchant Id, Cluster Id, Cluster Label, Category Id and Category Label.
Product Title is description of product name, Cluster Label is internal cluster name for that product and 
Category Label is the class of which the product belongs. There are 10 classes. 

### Steps to Run Start Docker Container:

1. **Clone the Repository**: `git clone <repository-url>`
2. **Navigate to Project Directory**: `cd project-directory`
3. `docker build -t <image_name> .`
4. **Run Docker Container**:
   ```bash
   docker run --name <container_name> --ipc=host -v /mountpath:/mounthpath -p 8888:8888 -p 6006:6006 --gpus all -it --rm <image_name>

### Project Structure
    ./
    ├── data
    │   ├── Directory to store data
    ├── model_files
    │   ├── Directory to store model weights
    ├── models
    │   ├── lstm.py
    └── utils
        ├── config.ini
        ├── helper_functions.py
    ├── data_exploration.ipynb
    ├── product_classification.ipynb
    ├── product_clustering.ipynb
    ├── README.md
    ├── requirements.txt
    ├── result.txt
    ├── sample_csv.csv
    ├── split_train_test.py
    ├── train_lstm.py
    ├── classification_report.ipynb
    └── create_vocab.py
  
### Data EDA
Exploratory Data Analysis of class distribution, token distribution etc  is done in data_exploration.ipynb. 

#### Run this commands before starting modelling 
1. **Split Train Test Val**: `python split_train_test.csv`

The dataset is split into 60% training, 20% validation & 20%test and saved to ensure validation dataset is consistent among different approces. 
finally Results are reported on Test Dataset. 

# Product Category Classification 
## Using Machine Learning Models
In the product_classification.ipynb following machine learning models are used to train product classifier.
The preprocessing is same for all models, which is lowercasing, remove punctuation & stopwords. Then Count Vector 
and TF-IDF is used to create word embeddings. 

1. Linear Regression
2. Support Vector Machine (SVM)
3. Naive Bayes
4. XGBoost
5. Random Forest

### Results on Test Dataset

|          Model          | Train Accuracy | Val Accuracy | Test Accuracy | 
|:-----------------------:|:--------------:|:------------:|:-------------:|
|   Logistic Regression   |     0.99      |    0.955     |     0.97        |
| Multinomial Naive Bayes |     0.96       |    0.94     |    0.96       |
|           SVM           |    0.99   |    0.957     |      0.98      |
|      Random Forest      |      0.99      |    0.955     |     0.98       |
|         Xgboost         |     0.95      |    0.945     |     0.95       |

## Using Glove + LSTM

#### Steps to train LSTM with Glove Embeddings 
1. **Script to train LSTM**: `python train_lstm.py`
2. **Script uses create_vocab.py to create vocab from train dataset, load embeddings found in glove otherwise initialize embeddings for words which are not found**
3. **model/lstm.py contains LSTM Model Definition**
4. **Other helper function are inside utils/helper_functions.**
5. **utils/config.ini contains lstm training hyperparameters.**

Note :- Glove 6B.100d was used. Only 20% vocab was found in glove embeddings because a lot of tokes are 
version names mix of alphanumeric characters such as ik-10600k, lu232df etc.


<figure><img src="model_files/lstm/tensorboard_images/Training%20Loss.svg" alt="Training Loss"> <figcaption>Traim Loss</figcaption></figure>
<figure><img src="model_files/lstm/tensorboard_images/Validation%20Loss.svg" alt="Validation Loss"><figcaption>Val Loss</figcaption></figure>
<figure><img src="model_files/lstm/tensorboard_images/Training%20Accuracy.svg" alt="Training Accuracy"><figcaption>Train Acc</figcaption></figure>
<figure><img src="model_files/lstm/tensorboard_images/Validation%20Accuracy.svg" alt="Validation Accuracy"><figcaption>Val Acc</figcaption></figure>

# Product Cluster Label Matching  

## Using String Similarity Methods
In the product_clustering.ipynb baseline string similarity metrics such as cosine similarity, dice, token matching are used to match
cluster label name. Around 50% accuracy is achieved without using modellings. 


## Classification Reports
In the classification_report.ipynb detailed performance is computed on product classification for test dataset. 
