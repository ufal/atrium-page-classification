# Image processing using ViT - for historical documents

### Goal: This project solves a task of page images classification

**Scope:** Processing of images, training and evaluation of ViT model,
input file/directory processing, class (category) results of top
N predictions output, predictions summarizing into a tabular format, 
HF 😊 hub support for the model

## Model description

Fine-tuned model files can be found here:  [huggingface.co/k4tel/vit-historical-page](https://huggingface.co/k4tel/vit-historical-page) 🔗

Base model repository: [huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) 🔗

### Data

Training set of the model: **8950** images 

#### Categories

| Label | Size (ratio)   | Description                                                                        |
| --- |----------------|------------------------------------------------------------------------------------|
| **DRAW 📈** | 	1182	(11.89%) | **drawings, maps, paintings with text**                                            |
|**DRAW_L 📈📏**| 	813	(8.17%)   | **drawings, maps, paintings with a table legend or inside tabular layout / forms** |
| **LINE_HW ✏️📏**| 	596	(5.99%)   | **handwritten text lines inside tabular layout / forms**                           |
| **LINE_P 📏**| 	603	(6.06%)   | **printed text lines inside tabular layout / forms**                               |
|**LINE_T 📏**| 	1332	(13.39%) | **machine typed text lines inside tabular layout / forms**                         |
| **PHOTO 🌄**| 	1015	(10.21%) | **photos with text**                                                               |
| **PHOTO_L 🌄📏**| 	782	(7.86%)   | **photos inside tabular layout / forms or with a tabular annotation**              |
| **TEXT 📰**| 	853	(8.58%)   | **mixed types of printed and handwritten texts**                                   |
| **TEXT_HW ✏️📄**| 	732	(7.36%)   | **only handwritten text**                                                          |
| **TEXT_P 📄**| 	691	(6.95%)   | **only printed text**                                                              |
| **TEXT_T 📄**| 	1346	(13.53%) | **only machine typed text**                                                        |

Evaluation set (10% of the above stats) [20250209-1534_model_1119_3_EVAL.csv](result/tables/20250209-1534_model_1119_3_EVAL.csv) 🔗:	**995** images 

## How to install 🔧 and run ▶️

Open [config.txt](config.txt) 🔗 and change folder path in the \[INPUT\] section, then optionally change **top_N** and **batch** in the \[SETUP\] section.

> [!NOTE]
>️ **Top-3** is enough to cover most of the images, setting **Top-5** will help with a small number of difficult to classify samples.
Finally, using **Top-11** option will give you a **raw version** of class scores returned by the model

> [!CAUTION]
> Do not try to change **base_model** and other section contents unless you know what you are doing

There are few option to obtain the trained model files:

- get a complete archive of the model folder from its developers ( Create a folder "**model**" next to this file, then place the model folder inside it)
- get a model and processor from the [HF 😊 repo](https://huggingface.co/k4tel/vit-historical-page) 🔗 using a specific flag described below

> [!IMPORTANT]
> Make sure you have **Python version 3.10+** installed on your machine 💻. Then create a separate virtual environment for this project 

Follow the **Linux** / **Windows**-specific instruction at the [venv docs](https://docs.python.org/3/library/venv.html) 👀🔗 if you don't know how to.
After creating the venv folder, activate the environment via:

    source <your_venv_dir>/bin/activate

and then inside your virtual environment to install python libraries (takes time ⌛) 

> [!NOTE]
> Up to **1 GB of space for model** files and checkpoints is needed, and up to **7 GB of space for the python libraries** (pytorch and its dependencies, etc)

Can be done via:

    pip install -r requirements.txt

To test that everything works okay and see the flag descriptions ❓ run:

    python3 run.py -h

There is an option to **load the model from the HF 😊 hub directly**, rather than use the local model folder.
To run any inference without locally saved model files, firstly load the model via:

    python3 run.py --hf

You should see a message about loading the model from hub and then saving it locally. 
Only after you obtain the trained model files, you can play with any commands provided below

### Common command examples 

Run the program from its starting point [run.py](run.py) 🔗 with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses 

> [!NOTE]
> Console output and all result tables contain **normalized** scores for the highest N class scores

**OR** if you are sure about default variables set in the [config.txt](config.txt) 🔗:

    python3 run.py -f '/full/path/to/file.png'

to run single PNG file classification - the output will be in the console. 

#### Directory processing 📁

    python3 run.py -tn 3 --inner -d '/full/path/to/directory' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses from all images found in the subdirectories of the given directory.

**OR** if you are really sure about default variables set in the [config.txt](config.txt) 🔗:

    python3 run.py --dir 

The classification results of PNG pages collected from the directory will be saved 💾 to related folders defined in [config.txt](config.txt)'s \[OUTPUT\] section.

> [!TIP]
> To parse all PNG files in the directory AND its subdirectories use the --inner flag

### Results 📊

Evaluation set's accuracy (**Top-3**):  **99.6%** 

![TOP-3 confusion matrix](result%2Fplots%2F20250209-1526_conf_mat.png)

Evaluation set's accuracy (**Top-1**):  **97.3%** 

![TOP-1 confusion matrix](result%2Fplots%2F20250218-1523_conf_mat.png)

#### Result tables

- Example of the manually ✍ **checked** results (small): [model_1119_3_TOP-5.csv](result%2Ftables%2Fmodel_1119_3_TOP-5.csv) 🔗

- Example of the manually ✍ **checked** evaluation dataset results (TOP-3): [20250209-1534_model_1119_3_TOP-3_EVAL.csv](result%2Ftables%2F20250209-1534_model_1119_3_TOP-3_EVAL.csv) 🔗

- Example of the manually ✍ **checked** evaluation dataset results (TOP-1): [20250218-1519_model_1119_3_TOP-1_EVAL.csv](result%2Ftables%2F20250218-1519_model_1119_3_TOP-1_EVAL.csv) 🔗

- Example of the **unchecked with TRUE** values results: [20250210-2034_model_1119_3_TOP-3.csv](result%2Ftables%2F20250210-2034_model_1119_3_TOP-3.csv) 🔗

#### Table columns

- **FILE** - name of the file
- **PAGE** - number of the page
- **CLASS-N** - label of the category, guess TOP-N 
- **SCORE-N** - score of the category, guess TOP-N

and optionally:
 
- **TRUE** - actual label of the category

## For devs

To train the model run: 

    python3 run.py --train  

To evaluate the model and create a confusion matrix plot 📊 run: 

    python3 run.py --eval  

Code of the algorithms can be found in the [classifier.py](classifier.py) 🔗 and [utils.py](utils.py) 🔗 files:

Code of the main function in the starting point [run.py](run.py) 🔗 file can be edited

#### Contacts

For support write to 📧 lutsai.k@gmail.com 📧
