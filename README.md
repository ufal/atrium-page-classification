**Goal:** This project solves a task of page images classification

**Scope:** Processing of images, training and evaluation of ViT model,
input file/directory processing, class (category) results of top N predictions output 
and summarizing into a tabular format 

**Categories:**

**DRAW**:	1182	(11.89%)  - drawings, maps, paintings 

**DRAW_L**:	813	(8.17%)   - drawings, maps, paintings inside tabular layout

**LINE_HW**:	596	(5.99%)   - handwritten text lines inside tabular layout

**LINE_P**:	603	(6.06%)   - printed text lines inside tabular layout

**LINE_T**:	1332	(13.39%)  - typed text lines inside tabular layout

**PHOTO**:	1015	(10.21%)  - photos with text

**PHOTO_L**:	782	(7.86%)   - photos inside tabular layout

**TEXT**:	853	(8.58%)   - mixed types, printed, and handwritten texts

**TEXT_HW**:	732	(7.36%)   - handwritten text

**TEXT_P**:	691	(6.95%)   - printed text

**TEXT_T**:	1346	(13.53%)  - typed text

Dataloader of train dataset is ready:	**8950** images split into 746 batches of size 12

Dataloader of eval dataset is ready:	**995** images split into 83 batches of size 12

Percentage correct (Top-3):  99.6

Example of the checked results: [model_1119_3.csv](result%2Ftables%2Fmodel_1119_3.csv)

Example of eval dataset results: [20250209-1522_model_1119_3_EVAL.csv](result%2Ftables%2F20250209-1522_model_1119_3_EVAL.csv)

Example of the unchecked results: [20250209-1204_model_1119_3.csv](result%2Ftables%2F20250209-1204_model_1119_3.csv)

**How to run:**

Open [.env](.env) environment file where all output folder paths are defined - please change all of them

Change paths to folders by replacing the beginnings of directory paths with your own **FULL** directory paths (to 
existing or not directories)

Use pip to install dependencies into Python 3.10 venv:

    pip install -r requirements.txt

Run the program from its starting point [run.py](run.py) with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'

to run single PNG file classification with top 3 predictions

    python3 run.py -tn 3 --dir -d '/full/path/to/directory' -m '/full/path/to/model/folder'


    python3 run.py -tn 3 -d '/full/path/to/directory' -m '/full/path/to/model/folder'

to parse all PNG files in the directory (+ its subdirectories) and classify all pages (RECOMMENDED)

The results of PNG pages classification will be saved to related folders 

Code of the algorithms can be found in the [classifier.py](classifier.py) file:

Code of the main function in the starting point [run.py](run.py) file can be edited. 
If [.env](.env) variables are not loaded - change filenames in the main function of [run.py](run.py)


**TIP**     You can set up default values of _model_path_, _topn_, _file_ and _directory_ values in the main function of
[run.py](run.py) and then run the script via:

    python3 run.py --dir 

which is for directory (and subdirectories) level processing

    python3 run.py 

which is for PDF file level processing

