# Automatic-Feature-Extraction-of-Person-from-Surveillance-Camera
Complexity in analysing the Large data collected from CCTV cameras.                            
Use of CCTV Cameras has raised drastically these days. For an Average Human It is not possible to maintain more than 3-4 frames at a time and the efficiency of work is less as the frames increasing. It is not economical to hire labour to the ratio of cctv cameras to human with 5:1. Hence, there is a need for an intelligent system to simplify the work of human.

## Introduction
   The availability of surveillance cameras placed in public locations has increased vastly in the last years, providing a safe environment to people at the cost of huge amount of visual data collected. Such data are mostly processed manually, a task which is labor intensive and prone to errors.Automatic approaches must be employed to enable the processing of the data, so that human operators only need to reason about selected portions. Aiming at solving problems in the domain of visual surveillance, computer vision techniques have been applied successfully for several years. They are rarely tackled in a scalable manner. With that in mind, in this project we try to tackle the feature extraction problem, one of the most expensive and necessary tasks in computer vision.

## Design Methodology
![image](https://user-images.githubusercontent.com/69564968/124391132-bb7fe800-dd0c-11eb-9a1e-86ce8fc212c7.png)

GoogleNet is a 22 layered (27 layers including polling layer)  convolutional neural network that uses the concept of inception. Input layer of googleNet takes in an image of dimension 224 *224. GoogleNet architecture is designed to have a great computation efficiency. The time required to train the model is less compared to other predecessor CNN network. The architecture is a winner of ILSVRC 2014 classification challenge where it came up with a top5 error of 6.67%.
## Details of Software / Hardware 
Computer/laptop,
CCTV Camera/Webcam,
Pycharm ,
Google Colab,
Python 3.7,
COM-SUR software

## Indivisual models
### Gender detection model
![image](https://user-images.githubusercontent.com/69564968/124391392-0e0dd400-dd0e-11eb-8ef4-b250ee233a17.png)
### Spectacle detection model
![image](https://user-images.githubusercontent.com/69564968/124391507-8a081c00-dd0e-11eb-8a52-0b140f7d6bf8.png)
### Beard detection model
![image](https://user-images.githubusercontent.com/69564968/124391557-cdfb2100-dd0e-11eb-8381-bbe6e316fde2.png)
### Age detection model
![image](https://user-images.githubusercontent.com/69564968/124391579-e53a0e80-dd0e-11eb-8cd0-e838c870fb98.png)

# Combining of All models:
![image](https://user-images.githubusercontent.com/69564968/124391622-1e727e80-dd0f-11eb-84d7-e4587615aab3.png)

# Model Output
![image](https://user-images.githubusercontent.com/69564968/124391640-3a762000-dd0f-11eb-960b-899b819e7f6a.png)

# Validating Model on CCTV: 
![image](https://user-images.githubusercontent.com/69564968/124391656-537ed100-dd0f-11eb-8e2b-5828fcbbcfe6.png)

# CONCLUSION:
The Project reviews an  intelligent surveillance video analysis techniques for feature extraction of person from surveillance camera. Reviewed project cover wide variety of applications.  The techniques, tools and dataset identified were listed in form of tables. The Trained models do work with sufficient accuracy under normal condition. Identifying each entity and their feature  is a difficult task. Methods analyzing crowd were discussed. The issues identified in existing methods were listed as future directions to provide efficient solution.
