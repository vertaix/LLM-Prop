# LLM-Prop: Predicting Physical And Electronic Properties Of Crystalline Solids From Their Text Descriptions
This repository contains the implementation of the LLM-Prop model. LLM-Prop is an efficiently finetuned large language model (T5 encoder) on crystals text descriptions to predict their properties. Given a text sequence that describes the crystal structure, LLM-Prop encodes the underlying crystal representation from its text description and output its properties such as band gap and volume. 

For more details check our paper: 

## Installation
You can install LLM-Prop by following these steps:
```
git clone https://github.com/vertaix/LLM-Prop.git
cd LLM-Prop
conda create -n <environment_name> requirement.txt
conda activate <environment_name>
```
## Usage
### Training LLM-Prop from scratch
```python
python llmprop_train.py \
--train_data_path= \
--valid_data_path= \
--test_data_path= \
--epochs=200 \
--task_name=regression \
--property=band_gap

```
