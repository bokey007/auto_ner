# auto_ner
End to End application for Custom Named Entity Recognition. 
Highlights: 
1. Powerd by GenAi 
2. Few shot Learning 
3. Training and inference pipelines
4. `Auto_annotate` will take unlabeled text data and create labellied text data that can further be used for custom Named Entity Recognition (NER) Model training.

### Chechout the Demo hosted at [Link](https://huggingface.co/spaces/bokey/auto_ner)

## Installation

#### Pypi
run following command in terminal
```bash
pip install auto-ner
```

#### From source
Run following command in terminal
1. ```git clone https://github.com/bokey007/auto_ner.git```
2. ```cd auto_ner```
3. ```python setup.py sdist bdist_wheel```
4. ```pip install ./dist/auto_ner-0.1.2.tar.gz```

## Usage
```bash
auto_ner.run
```
- Above command will lauch the app on default port 8501. 
- Open the browser and go to http://localhost:8501
- Select the image and then select the appropriate set of operations you want to perform on that perticular image. 
- play with the parameters interatively untill you reach at optimal configuration.

```bash
auto_ner.run --port 8080
```
Above command can be used to specify the port on which you want to run the app.

## Application Workflow
![](https://github.com/bokey007/auto_ner/blob/main/doc_images/Application%20Workflow.png)

## System Architecture
![](https://github.com/bokey007/auto_ner/blob/main/doc_images/System%20Architecture.png)

## Demo
![](https://github.com/bokey007/auto_ner/blob/main/doc_images/auto_ner_corrected.gif)

## Solution is implemnted in following three steps 
1. Create the baseline
    Spacy Model ([Transformer implementation on Hold])
2. Meet the Expectations
    Training Bert ([ToDo])
3. Exeed the expectations
    - Few shot / Zero Shot NER
    - Beyond mere NER : entyity linking ([ToDo])
    
Development tools:

1. setuptools (https://pypi.org/project/setuptools/): Used to create a python package
2. pipreqs (https://pypi.org/project/pipreqs/): Used to create requirements.txt file
3. twine (https://pypi.org/project/twine/): Used to upload the package to pypi.org
4. Github Actions (): Used to automate the process of uploading the package to pypi.org
5. pytest (https://pypi.org/project/pytest/): Used to write unit tests
6. wheel (https://pypi.org/project/wheel/): Used to create a wheel file

