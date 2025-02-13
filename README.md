# Image processing using ViT - for historical documents

**Goal:** This project solves a task of page images classification

**Scope:** Processing of images, training and evaluation of ViT model,
input file/directory processing, class (category) results of top
N predictions output, predictions summarizing into a tabular format, 
HF hub support for the model

## Model description:

Model files can be found here:  [huggingface.co/k4tel/vit-historical-page](https://huggingface.co/k4tel/vit-historical-page)

### Data:

Training set of the model: **8950** images 

#### Categories:

- **DRAW**:	1182	(11.89%)  - drawings, maps, paintings with text

- **DRAW_L**:	813	(8.17%)   - drawings, maps, paintings with a table legend or inside tabular layout / forms

- **LINE_HW**:	596	(5.99%)   - handwritten text lines inside tabular layout / forms

- **LINE_P**:	603	(6.06%)   - printed text lines inside tabular layout / forms

- **LINE_T**:	1332	(13.39%)  - machine typed text lines inside tabular layout / forms

- **PHOTO**:	1015	(10.21%)  - photos with text

- **PHOTO_L**:	782	(7.86%)   - photos inside tabular layout / forms

- **TEXT**:	853	(8.58%)   - mixed types, printed, and handwritten texts 

- **TEXT_HW**:	732	(7.36%)   - only handwritten text

- **TEXT_P**:	691	(6.95%)   - only printed text

- **TEXT_T**:	1346	(13.53%)  - only machine typed text

Evaluation set (10% of the above stats) [20250209-1534_model_1119_3_EVAL.csv](result/tables/20250209-1534_model_1119_3_EVAL.csv):	**995** images 

### Results:

Evaluation set's accuracy (Top-3):  **99.6%**

Regarding the model output, **Top-3** is enough to cover most of the images, 
setting **Top-5** will help with a small number of difficult to classify samples.
Finally, using **Top-11** option will give you a **raw version** of class scores returned by the model

#### Result tables:

- Example of the manually **checked** results (small): [model_1119_3.csv](result%2Ftables%2Fmodel_1119_3.csv)

- Example of the manually **checked** evaluation dataset results: [20250209-1534_model_1119_3_EVAL.csv](result/tables/20250209-1534_model_1119_3_EVAL.csv)

- Example of the **unchecked with TRUE** values results: [20250210-2034_model_1119_3_TOP-3.csv](result/tables/20250210-2034_model_1119_3_TOP-3.csv)

#### Table columns:

- **FILE** - name of the file
- **PAGE** - number of the page
- **CLASS-N** - label of the category, guess TOP-N 
- **SCORE-N** - score of the category, guess TOP-N

and optionally:
 
- **TRUE** - actual label of the category

## How to install and run:

Open [config.txt](config.txt) and change folder path in the \[INPUT\] section, then optionally change **top_N** and **batch** in the \[SETUP\] section.

**WARNING**: do not try to change **base_model** and other section contents unless you know what you are doing

There are few option to obtain the trained model files:

- get a complete archive of the model folder from its developers ( Create a folder "**model**" next to this file, then place the model folder inside it)
- get a model and processor from the [HF repo](https://huggingface.co/k4tel/vit-historical-page) using a specific flag described below

Make sure you have Python version 3.10+ installed on your machine.
Then create a virtual environment for this project following the Linux/Windows-specific instruction at the [venv docs](https://docs.python.org/3/library/venv.html)

Note that you will need up to **1 GB of space for model** files and checkpoints, 
and up to **7 GB of space for the python libraries** (pytorch and its dependencies, etc)

After creating the venv folder, activate the environment via:

    source <your_venv_dir>/bin/activate

and then inside your virtual environment to install the python libraries run:

    pip install -r requirements.txt

To test that everything works okay and see the flag descriptions run:

    python3 run.py -h

There is an option to **load the model from the HF hub directly**, rather than use the local model folder.
To run any inference without locally saved model files, firstly login to your **HF account** and find 
**Access tokens** section in the **account settings**, where you should **generate a new token**. 

After that change token variable in the \[HF\] section of the [config.txt](config.txt) to your generated token string.
Only now you can load the model via:

    python3 run.py --hf

You should see a message about loading the model from hub and then saving it locally. 
Only after you obtain the trained model files, you can play with any commands listed below

### Common command examples

Run the program from its starting point [run.py](run.py) with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses. 

**OR** if you are sure about default variables set in the [config.txt](config.txt):

    python3 run.py -f '/full/path/to/file.png'


to run single PNG file classification - the output will be in the console. 

#### Directory processing

    python3 run.py -tn 3 --inner -d '/full/path/to/directory' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses from all images found in the subdirectories of the given directory.

**OR** if you are really sure about default variables set in the [config.txt](config.txt):

    python3 run.py --dir 

to parse all PNG files in the directory (+ its subdirectories if use _--inner_) and classify all pages (**RECOMMENDED**)
The results of those PNG pages classification will be saved to related folders defined in [config.txt](config.txt)'s \[OUTPUT\] section.

## For devs

To train the model run: 

    python3 run.py --train  

To evaluate the model and create a confusion matrix plot run: 

    python3 run.py --eval  

Code of the algorithms can be found in the [classifier.py](classifier.py) and [utils.py](utils.py) files:

Code of the main function in the starting point [run.py](run.py) file can be edited

