**Goal:** This project solves a task of page images classification

**Scope:** Processing of images, training and evaluation of ViT model,
input file/directory processing, class (category) results of top N predictions output 
and summarizing into a tabular format 

## Model description:

### Categories:

- **DRAW**:	1182	(11.89%)  - drawings, maps, paintings 

- **DRAW_L**:	813	(8.17%)   - drawings, maps, paintings inside tabular layout

- **LINE_HW**:	596	(5.99%)   - handwritten text lines inside tabular layout

- **LINE_P**:	603	(6.06%)   - printed text lines inside tabular layout

- **LINE_T**:	1332	(13.39%)  - typed text lines inside tabular layout

- **PHOTO**:	1015	(10.21%)  - photos with text

- **PHOTO_L**:	782	(7.86%)   - photos inside tabular layout

- **TEXT**:	853	(8.58%)   - mixed types, printed, and handwritten texts

- **TEXT_HW**:	732	(7.36%)   - handwritten text

- **TEXT_P**:	691	(6.95%)   - printed text

- **TEXT_T**:	1346	(13.53%)  - typed text

### Data:

Training set of the model: **8950** images 

Evaluation set [20250209-1534_model_1119_3_EVAL.csv](result/tables/20250209-1534_model_1119_3_EVAL.csv):	**995** images - percentage correct (Top-3):  **99.6%**

### Result tables:

Example of the manually **checked** results: [model_1119_3.csv](result%2Ftables%2Fmodel_1119_3.csv)

Example of the manually **checked** evaluation dataset results: [20250209-1534_model_1119_3_EVAL.csv](result/tables/20250209-1534_model_1119_3_EVAL.csv)

Example of the **unchecked** with **TRUE** values results: [20250210-2034_model_1119_3_TOP-3.csv](result/tables/20250210-2034_model_1119_3_TOP-3.csv)

#### Table columns:

- FILE - name of the file
- PAGE - number of the page
- CLASS-N - label of the category, guess TOP-N 
- SCORE-N - score of the category, guess TOP-N

and optionally:
 
- TRUE - actual label of the category

## How to install and run:

Open [config.txt](config.txt) and change folder path in the \[INPUT\] section, then optionally change **top_N** and **batch** in the \[SETUP\] section.

**WARNING**: do not try to change **base_model** and other section contents unless you know what you are doing

Create a folder "**model**" next to this file, then place model folder inside it.  

Use pip to install dependencies into Python 3.10+ [venv](https://docs.python.org/3/library/venv.html):

    source <your_venv_dir>/bin/activate

and then inside your virtual environment:

    pip install -r requirements.txt

To test that everything works okay and see the flag descriptions run:

    python3 run.py -h

### Common command examples

Run the program from its starting point [run.py](run.py) with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses. 

**OR** if you are sure about default variables set in the [config.txt](config.txt):

    python3 run.py -f '/full/path/to/file.png'


to run single PNG file classification - the output will be in the console. 

#### Directory processing

    python3 run.py -tn 3 --dir -d '/full/path/to/directory' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses

**OR** if you are really sure about default variables set in the [config.txt](config.txt):

    python3 run.py --dir 

to parse all PNG files in the directory (+ its subdirectories) and classify all pages (**RECOMMENDED**)
The results of those PNG pages classification will be saved to related folders defined in [config.txt](config.txt)'s \[OUTPUT\] section.

## For devs

Code of the algorithms can be found in the [classifier.py](classifier.py) and [utils.py](utils.py) files:

Code of the main function in the starting point [run.py](run.py) - file can be edited 
If [config.txt](config.txt) variables are not loaded - change them in the main function of [run.py](run.py) manually.

