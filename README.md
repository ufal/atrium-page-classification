# Image classification using fine-tuned ViT or EffNetV2 - for historical document sorting

### Goal: solve a task of archive page images sorting (for their further content-based processing)

**Scope:** Processing of images, training / evaluation of ViT / EffNetV2 model,
input file/directory processing, class 🪧  (category) results of top
N predictions output, predictions summarizing into a tabular format, 
HF 😊 hub [^1] 🔗 support for the model, multiplatform (Win/Lin) data 
preparation scripts for PDF to PNG conversion

### Table of contents 📑

  * [Versions 🏁](#versions-)
  * [Model description 📇](#model-description-)
    + [Data 📜](#data-)
    + [Categories 🪧️](#categories-)
  * [How to install 🔧](#how-to-install-)
  * [How to run prediction 🪄 modes](#how-to-run-prediction--modes)
    + [Page processing 📄](#page-processing-)
    + [Directory processing 📁](#directory-processing-)
  * [Results 📊](#results-)
      - [Result tables and their columns 📏📋](#result-tables-and-their-columns-)
  * [Data preparation 📦](#data-preparation-)
    + [PDF to PNG 📚](#pdf-to-png-)
    + [PNG pages annotation 🔎](#png-pages-annotation-)
    + [PNG pages sorting for training 📬](#png-pages-sorting-for-training-)
  * [For developers 🪛](#for-developers-)
    * [Training 💪](#training-)
    * [Evaluation 🏆](#evaluation-)
  * [Contacts 📧](#contacts-)
  * [Acknowledgements 🙏](#acknowledgements-)
  * [Appendix 🤓](#appendix-)

----

## Versions 🏁

There are currently 2 version of the model available for download, both of them have the same set of categories, 
but different data annotations. The latest approved `v2.1` is considered to be default and can be found in the `main` branch
of HF 😊 hub [^1] 🔗 

| Version | Base                             | Pages |   PDFs   | Description                                                                        |
|--------:|----------------------------------|:-----:|:--------:|:-----------------------------------------------------------------------------------|
|  `v2.0` | `vit-base-patch16-224`           | 10073 | **3896** | annotations with mistakes, more heterogenous data                                  |
|  `v2.1` | `vit-base-patch16-224`           | 11940 | **5002** | `main`: more diverse pages in each category, less annotation mistakes              |
|  `v2.2` | `vit-base-patch16-224`           | 15855 | **5730** | same data as `v2.1` + some restored pages from `v2.0`                              |
|  `v3.2` | `vit-base-patch16-384`           | 15855 | **5730** | same data as `v2.2`, but a bit larger model base with higher resolution            |
|  `v5.2` | `vit-large-patch16-384`          | 15855 | **5730** | same data as `v2.2`, but the largest model base with higher resolution             |
|  `v1.2` | `efficientnetv2_s.in21k`         | 15855 | **5730** | same data as `v2.2`, but the smallest model base (CNN)                             |
|  `v4.2` | `efficientnetv2_l.in21k_ft_in1k` | 15855 | **5730** | same data as `v2.2`, CNN base model smaller than the largest, may be more accurate |
|  `v2.3` | `vit-base-patch16-224`           | 38625 |  **0**   | new data annotation phase data, more single-page documents used                    |
|  `v3.3` | `vit-base-patch16-384`           | 38625 |  **0**   | same data as `v2.3`, but a bit larger model base with higher resolution            |
|  `v5.3` | `vit-large-patch16-384`          | 38625 |  **0**   | same data as `v2.3`, but the largest model base with higher resolution             |
|  `v1.3` | `efficientnetv2_s.in21k`         | 38625 |  **0**   | same data as `v2.3`, but the smallest model base (CNN)                             |
|  `v4.3` | `efficientnetv2_l.in21k_ft_in1k` | 38625 |  **0**   | same data as `v2.3`, CNN base model smaller than the largest, may be more accurate |

<details>

<summary>Base model - size 👀</summary>

| **Version**                      | **Parameters (M)** | Resolution (px) |
|----------------------------------|--------------------|-----------------|
| `efficientnetv2_s.in21k`         | 48                 | 300             |
| `vit-base-patch16-224`           | 87                 | 224             |
| `vit-base-patch16-384`           | 87                 | 384             |
| `efficientnetv2_l.in21k_ft_in1k` | 119                | 384             |
| `vit-large-patch16-384`          | 305                | 384             |

</details>

## Model description 📇

![architecture.png](architecture.png)

🔲 **Fine-tuned** model repository: UFAL's **vit-historical-page** [^1] 🔗

🔳 **Base** model repository: 
- Google's **vit-base-patch16-224**,  **vit-base-patch16-384**, and  **vit-large-patch16-284** [^2] [^13] [^14] 🔗
- timm's **efficientnetv2_s.in21k** and **efficientnetv2_l.in21k_ft_in1k** [^15] [^16] 🔗

The model was trained on the manually ✍️ annotated dataset of historical documents, in particular, images of pages 
from the archival documents with paper sources that were scanned into digital form. 

The images contain various combinations of texts ️📄, tables 📏, drawings 📈, and photos 🌄 - 
categories 🪧 described [below](#categories-) were formed based on those archival documents. Page examples can be found in
the [category_samples](category_samples) 📁 directory.

The key **use case** of the provided model and data processing pipeline is to classify an input PNG image from PDF scanned 
paper source into one of the categories - each responsible for the following content-specific data processing pipeline.

> In other words, when several APIs for different OCR subtasks are at your disposal - run this classifier first to 
> mark the input data as machine-typed (old style fonts) / handwritten ✏️ / just printed plain ️📄 text 
> or structured in tabular 📏 format text, as well as to mark the presence of the printed 🌄 or drawn 📈 graphic 
> materials yet to be extracted from the page images.

### Data 📜

[//]: # (The dataset is provided under Public Domain license, and consists of **15855** PNG images of pages from the archival documents.)
[//]: # (The source image files and their annotation can be found in the LINDAT repository [^17] 🔗.)

**Training** 💪 set of the model: **8950** images for `v2.0`

**Training** 💪 set of the model: **10745** images for `v2.1`

**Training** 💪 set of the model: **14565** images for `vX.2` 

**Training** 💪 set of the model: **38625** images for `vX.3` 

> **90% of all** - proportion in categories 🪧 tabulated [below](#categories-)

**Evaluation** 🏆 set:  **1290** images (taken from `v2.2` annotations)

**Evaluation** 🏆 set:  **4823** images (for `vX.3` models)

> **10% of all** - same proportion in categories 🪧 as [below](#categories-) and demonstrated in [model_EVAL.csv](result%2Ftables%2F20250701-1057_model_v220105p_TOP-1_EVAL.csv) 📎

Manual ✍️ annotation was performed beforehand and took some time ⌛, the categories 🪧  were formed from
different sources of the archival documents originated in the 1920-2020 years span. 

> [!NOTE]
> Disproportion of the categories 🪧 in both training data and provided evaluation [category_samples](category_samples) 📁 is
> **NOT** intentional, but rather a result of the source data nature. 

In total, several thousands of separate PDF files were selected and split into PNG pages, ~4k of scanned documents 
were one-page long which covered around a third of all data, and ~2k of them were much longer (dozens and hundreds 
of pages) covering the rest (more than 60% of all annotated data). 

The specific content and language of the
source data is irrelevant considering the model's vision resolution, however, all of the data samples were from **archaeological 
reports** which may somehow affect the drawing detection preferences due to the common form of objects being ceramic pieces, 
arrowheads, and rocks formerly drawn by hand and later illustrated with digital tools (examples can be found in
[category_samples/DRAW](category_samples%2FDRAW) 📁)

![data_timeline.png](dataset_timeline.png)


### Categories 🪧

|    Label️ | Description                                                                                                      |
|----------:|:-----------------------------------------------------------------------------------------------------------------|
|    `DRAW` | **📈 - drawings, maps, paintings, schematics, or graphics, potentially containing some text labels or captions** |
|  `DRAW_L` | **📈📏 - drawings, etc but presented within a table-like layout or includes a legend formatted as a table**      |
| `LINE_HW` | **✏️📏 - handwritten text organized in a tabular or form-like structure**                                        |
|  `LINE_P` | **📏 - printed text organized in a tabular or form-like structure**                                              |
|  `LINE_T` | **📏 - machine-typed text organized in a tabular or form-like structure**                                        |
|   `PHOTO` | **🌄 - photographs or photographic cutouts, potentially with text captions**                                     |
| `PHOTO_L` | **🌄📏 - photos presented within a table-like layout or accompanied by tabular annotations**                     |
|    `TEXT` | **📰 - mixtures of printed, handwritten, and/or typed text, potentially with minor graphical elements**          |
| `TEXT_HW` | **✏️📄 - only handwritten text in paragraph or block form (non-tabular)**                                        |
|  `TEXT_P` | **📄 - only printed text in paragraph or block form (non-tabular)**                                              |
|  `TEXT_T` | **📄 - only machine-typed text in paragraph or block form (non-tabular)**                                        |

The categories were chosen to sort the pages by the following criteria: 

- **presence of graphical elements** (drawings 📈 OR photos 🌄)
- **type of text** 📄 (handwritten ✏️️ OR printed OR typed OR mixed 📰)
- **presence of tabular layout / forms** 📏

> The reasons for such distinction are different processing pipelines for different types of pages, which would be
> applied after the classification as mentioned [above](#model-description-).

Examples of pages sorted by category 🪧 can be found in the [category_samples](category_samples) 📁 directory
which is also available as a testing subset of the training data (can be used to run evaluation and prediction with a
necessary `--inner` flag).

----

## How to install 🔧

Step-by-step instructions on this program installation are provided here. The easiest way to obtain the model would 
be to use the HF 😊 hub repository [^1] 🔗 that can be easily accessed via this project. 

<details>

<summary>Hardware requirements 👀</summary>

**Minimal** machine 🖥️ requirements for slow prediction run (and very slow training / evaluation):
- **CPU** with a decent (above average) operational memory size

**Ideal** machine 🖥️ requirements for fast prediction (and relatively fast training / evaluation):
- **CPU** of some kind and memory size
- **GPU** (for real CUDA [^10] support - only one of Nvidia's cards)

</details>

> [!WARNING]
> Make sure you have **Python version 3.10+** installed on your machine 💻 and check its
> **hardware requirements** for correct program running provided above. 
> Then create a separate virtual environment for this project 

<details>

<summary>How to 👀</summary>

Clone this project to your local machine 🖥️️ via:

    cd /local/folder/for/this/project
    git init
    git clone https://github.com/ufal/atrium-page-classification.git

Then change to the Vit and EffNet models or CLIP models branch (`clip` or `vit`):

    cd atrium-page-classification
    git checkout vit

**OR** for updating the already cloned project with some changes, go to the folder containing (hidden) `.git` 
subdirectory and run pulling which will merge upcoming files with your local changes:

    cd /local/folder/for/this/project/atrium-page-classification
    git add <changed_file>
    git commit -m 'local changes'

And then for updating the project with the latest changes from the remote repository, run:

    git pull -X theirs

Alternatively, if you are interested in a specific branch (`clip` or `vit`), you can update  it via:

    git fetch origin
    git checkout vit        
    git pull --ff-only origin vit

Alternatively, if you do **NOT** care about local changes **OR** you want to get the latest project files, 
just remove those files (all `.py`, `.txt` and `README` files) and pull the latest version from the repository:

    cd /local/folder/for/this/project/atrium-page-classification

And then for a total clean up and update, run:

    rm *.py
    rm *.txt
    rm README*
    git pull

Alternatively, for a specific branch (`clip` or `vit`):

    git reset --hard HEAD
    git clean -fd
    git fetch origin
    git checkout vit
    git pull origin vit


Overall, a force update to the remote repository branch (`clip` or `vit`) looks like this:

    git fetch origin
    git checkout vit
    git reset --hard origin/vit

Next step would be a creation of the virtual environment. Follow the **Unix** / **Windows**-specific 
instruction at the venv docs [^3] 👀🔗 if you don't know how to.

After creating the venv folder, activate the environment via:

    source <your_venv_dir>/bin/activate

and then inside your virtual environment, you should install Python libraries (takes time ⌛) 

</details>

> [!CAUTION]
> Up to **1 GB of space for model** files and checkpoints is needed, and up to **7 GB 
> of space for the Python libraries** (Pytorch and its dependencies, etc)

Installation of Python dependencies can be done via:

    pip install -r requirements.txt

> [!NOTE]
> The so-called **CUDA [^10] support** for Python's PyTorch library is supposed to be automatically installed
> at this point - when the presence of the GPU on your machine 🖥️
> is checked for the first time, later it's also checked every time before the model initialization
> (for training, evaluation or prediction run).

After the dependencies installation is finished successfully, in the same virtual environment, you can
run the Python program.  

To test that everything works okay and see the flag 
descriptions call for `--help` ❓:

    python3 run.py -h

You should see a (hopefully) helpful message about all available command line flags. Your next step would be
to **pull the model from the HF 😊 hub repository [^1] 🔗** via:

    python3 run.py --hf

**OR** for specific model version (e.g. `main`, `v2.0`, `vX.2` or `vX.3`) use the `--revision` flag:
 
    python3 run.py --hf -rev v2.0

**OR** for specific base model version (e.g. `google/vit-large-patch16-384`) use the `--base` flag (only when the 
trained model version demands such base model as described [above](#versions-)):
 
    python3 run.py --hf -rev v5.2 -b google/vit-large-patch16-384

> [!IMPORTANT]
> If you already have the model files in the `model/movel_<revision>`
> directory next to this file, you do **NOT** have to use the `--hf` flag to download the
> model files from the HF 😊 repo [^1] 🔗 (only for the **model version update**).

You should see a message about loading the model from the hub and then saving it locally on
your machine 🖥️. 

Only after you have obtained the trained model files (takes less time ⌛ than installing dependencies), 
you can play with any commands provided [below](#how-to-run-prediction--modes).

After the model is downloaded, you should see a similar file structure: 

<details>

<summary>Initial project tree 🌳 files structure 👀</summary>
    
    /local/folder/for/this/project/atrium-page-classification
    ├── model
        └── movel_<revision> 
            ├── config.json
            ├── model.safetensors
            └── preprocessor_config.json
    ├── checkpoint
        ├── models--google--vit-base-patch16-224
            ├── blobs
            ├── snapshots
            └── refs
        └── .locs
            └── models--google--vit-base-patch16-224
    ├── data_scripts
        ├── windows
            ├── move_single.bat
            ├── pdf2png.bat
            └── sort.bat
        └── unix
            ├── move_single.sh
            ├── pdf2png.sh
            └── sort.sh
    ├── result
        ├── plots
            ├── date-time_conf_mat.png
            └── ...
        └── tables
            ├── date-time_TOP-N.csv
            ├── date-time_TOP-N_EVAL.csv
            ├── date-time_EVAL_RAW.csv
            └── ...
    ├── category_samples
        ├── DRAW
            ├── CTX193200994-24.png
            └── ...
        ├── DRAW_L
        └── ...
    ├── run.py
    ├── classifier.py
    ├── utils.py
    ├── requirements.txt
    ├── config.txt
    ├── README.md
    └── ...

</details>

Some of the folders may be missing, like mentioned [later](#for-developers-) `model_output` which is automatically created
only after launching the model.

----

## How to run prediction 🪄 modes

There are two main ways to run the program:

- **Single PNG file classification** 📄
- **Directory with PNG files classification** 📁

To begin with, open [config.txt](config.txt) ⚙ and change folder path in the `[INPUT]` section, then 
optionally change `top_N` and `batch` in the `[SETUP]` section.

> [!NOTE]
>️ **Top-3** is enough to cover most of the images, setting **Top-5** will help with a small number 
> of difficult to classify samples.

The `batch` variable value depends on your machine 🖥️ memory size

<details>

<summary>Rough estimations of memory usage per batch size 👀</summary>

| **Batch size** | **CPU / GPU memory usage** |
|----------------|----------------------------|
| 4              | 2 Gb                       |
| 8              | 3 Gb                       |
| 16             | 5 Gb                       |
| 32             | 9 Gb                       |
| 64             | 17 Gb                      |

</details>

It is safe to use batch size below **12** for a regular office desktop computer, and lower it to **4** if it's an old device.
For training on a High Performance Computing cluster, you may use values above **20** for
the `batch` variable in the `[SETUP]` section.

> [!CAUTION]
> Do **NOT** try to change **base_model** and other section contents unless you know what you are doing

<details>

<summary>Rough estimations of disk space needed for trained model in relation to the base model 👀</summary>

| **Version**             | **Disk space** |
|-------------------------|----------------|
| `efficientnetv2_s`      | 82 Mb          |
| `vit-base-patch16-224`  | 344 Mb         |
| `vit-base-patch16-384`  | 345 Mb         |
| `efficientnetv2_l`      | 471 Mb         |
| `vit-large-patch16-384` | 1.2 Gb         |

</details>

Make sure the virtual environment with all the installed libraries is activated, you are in the project 
directory with Python files and only then proceed. 

<details>

<summary>How to 👀</summary>

    cd /local/folder/for/this/project/
    source <your_venv_dir>/bin/activate
    cd atrium-page-classification

</details>

> [!IMPORTANT]
> All the listed below commands for Python scripts running are adapted for **Unix** consoles, while
> **Windows** users must use `python` instead of `python3` syntax

### Page processing 📄

The following prediction should be run using the `-f` or `--file` flag with the path argument. Optionally, 
you can use the `-tn` or `--topn` flag with the number of guesses you want to get, and also the `-m` or 
`--model` flag with the path to the model folder argument. For the specific image file format collection from
the input fictionary use `-ff` or `--file_format` flag with the format argument (default is `jpeg`).

<details>

<summary>How to 👀</summary>

Run the program from its starting point [run.py](run.py) 📎 with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses with a console output.

**OR** if you are sure about default variables set in the [config.txt](config.txt) ⚙:

    python3 run.py -f '/full/path/to/file.png'

to run a single PNG file classification - the output will be in the console. 

</details>

> [!NOTE]
> Console output and all result tables contain **normalized** scores for the highest N class 🪧  scores

### Directory processing 📁

The following prediction type does **NOT** require explicit directory path setting with the `-d` or `--directory`, 
since its default value is set in the [config.txt](config.txt) ⚙ file and awakens when the `--dir` flag 
is used. The same flags for the number of guesses and the model folder path as for the single-page 
processing can be used. In addition, 2 directory-specific flags  `--inner` and `--raw` are available. 

> [!CAUTION]
> You must either explicitly set the `-d` flag's argument or use the `--dir` flag (calling for the preset in 
> `[INPUT]` section default value of the input directory) to process PNG files on the directory
> level, otherwise, nothing will happen

Worth mentioning that the **directory 📁 level processing is performed in batches**, therefore you should refer to
the hardware's memory capacity requirements for different batch sizes tabulated [above](#how-to-run-prediction--modes).

Moreover, in case you have a large amount of files (more than 500,000) that you attempt to process in one run,
you should keep in mind that even listing all of the files from all of the subdirectories may take a while ⌛,
not to mention the actual processing time.

<details>

<summary>How to 👀</summary>

    python3 run.py -tn 3 -d '/full/path/to/directory' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses in tabular format from all images found in the given directory.

**OR** if you are really sure about default variables set in the [config.txt](config.txt) ⚙:

    python3 run.py --dir 
    
    python3 run.py -rev v3.2 -b google/vit-base-patch16-384 --inner --dir

    python3 run.py -m "./models/model_v43" --raw --dir -tn 3

</details>

The classification results of PNG pages collected from the directory will be saved 💾 to related [results](result) 📁
folders defined in `[OUTPUT]` section of [config.txt](config.txt) ⚙ file.

> [!TIP]
> To additionally get raw class 🪧 probabilities from the model along with the TOP-N results, use
> `--raw` flag when processing the directory (**NOT** available for single file processing)
 
> [!TIP]
> To process all PNG files in the directory **AND its subdirectories** use the `--inner` flag
> when processing the directory, or switch its default value to `True` in the `[SETUP]` section 
 
Naturally, processing of the large amount of PNG pages takes time ⌛ and progress of this process
is recorded in the console via messages like `Processed <B×N> images` where `B`
is batch size set in the `[SETUP]` section of the [config.txt](config.txt) ⚙ file, 
and `N` is an iteration of the current dataloader processing loop. 

Only after all images from the input directory are processed, the output table is
saved 💾 in the `result/tables` folder. 

----

## Results 📊

There are accuracy performance measurements and plots of confusion matrices for the evaluation 
dataset (10% of the provided in `[TRAIN]`'s folder data). Both graphic plots and tables with 
results can be found in the [result](result) 📁 folder.

| **Revision** | **Top-1** | **Top-3** |
|--------------|-----------|-----------|
| `v1.2`       | 97.73     | 99.87     |
| `v2.2`       | 97.54     | 99.94     |
| `v3.2`       | 96.49     | 99.94     |
| `v4.2`       | 97.73     | 99.87     |
| `v5.2`       | 97.86     | 99.87     |
| `v1.3`       | -         | -         |
| `v2.3`       | -         | -         |
| `v3.3`       | -         | -         |
| `v4.3`       | -         | -         |
| `v5.3`       | -         | -         |

`v2.2` Evaluation set's accuracy (**Top-1**):  **97.54%** 🏆

<details>

<summary>Confusion matrix 📊 TOP-1 👀</summary>

![TOP-1 confusion matrix](result%2Fplots%2F20250701-1136_model_v220105p_conf_mat_TOP-1.png)

</details>

`v3.2` Evaluation set's accuracy (**Top-1**):  **96.49%** 🏆

<details>

<summary>Confusion matrix 📊 TOP-1 👀</summary>

![TOP-1 confusion matrix](result%2Fplots%2F20250701-1142_model_v320105p_conf_mat_TOP-1.png)

</details>

`v5.2` Evaluation set's accuracy (**Top-1**):  **97.73%** 🏆

<details>

<summary>Confusion matrix 📊 TOP-1 👀</summary>

![TOP-1 confusion matrix](result%2Fplots%2F20250701-1203_model_v520105p_conf_mat_TOP-1.png)

</details>

`v1.2` Evaluation set's accuracy (**Top-1**):  **97.73%** 🏆

<details>

<summary>Confusion matrix 📊 TOP-1 👀</summary>

![TOP-1 confusion matrix](result%2Fplots%2F20250709-1831_model_v120106s_conf_mat_TOP-1.png)

</details>

`v4.2` Evaluation set's accuracy (**Top-1**):  **97.86%** 🏆

<details>

<summary>Confusion matrix 📊 TOP-1 👀</summary>

![TOP-1 confusion matrix](result%2Fplots%2F20250709-1829_model_v120106l_conf_mat_TOP-1.png)

</details>

> **Confusion matrices** provided above show the diagonal of matching gold and predicted categories 🪧
> while their off-diagonal elements show inter-class errors. By those graphs you can judge 
> **what type of mistakes to expect** from your model. 

By running tests on the evaluation dataset after training you can generate the following output files:

- **date-time_model_TOP-N_EVAL.csv** - (by default) results of the evaluation dataset with TOP-N guesses
- **date-time_model_conf_mat_TOP-N.png** - (by default) confusion matrix plot for the evaluation dataset also with TOP-N guesses
- **date-time_model_EVAL_RAW.csv** - (by flag `--raw`) raw probabilities for all classes of the evaluation dataset 

> [!NOTE]
> Generated tables will be sorted by **FILE** and **PAGE** number columns in ascending order. 

Additionally, results of prediction inference run on the directory level without checked results are included.

### Result tables and their columns 📏📋

<details>

<summary>General result tables 👀</summary>

Demo files  `v2.2`:

- Manually ✍️ **checked** evaluation dataset (TOP-1): [model_TOP-1_EVAL.csv](result%2Ftables%2F20250701-1057_model_v220105p_TOP-1_EVAL.csv) 📎

- Manually ✍️ **checked** evaluation dataset (TOP-3): [model_TOP-3_EVAL.csv](result%2Ftables%2F20250710-1925_model_v220105p_TOP-3_EVAL.csv) 📎

- **Unchecked with TRUE** values (small): [model_TOP-1.csv](result%2Ftables%2F20250710-1939_model_v220105p_TOP-1.csv)📎

Demo files  `v3.2`:

- Manually ✍️ **checked** evaluation dataset (TOP-1): [model_TOP-1_EVAL.csv](result%2Ftables%2F20250701-1057_model_v320105p_TOP-1_EVAL.csv) 📎

- Manually ✍️ **checked** evaluation dataset (TOP-3): [model_TOP-3_EVAL.csv](result%2Ftables%2F20250710-1927_model_v320105p_TOP-3_EVAL.csv) 📎

- **Unchecked with TRUE** values (small): [model_TOP-1.csv](result%2Ftables%2F20250710-1936_model_v320105p_TOP-1.csv)📎

Demo files  `v5.2`:

- Manually ✍️ **checked** evaluation dataset (TOP-1): [model_TOP-1_EVAL.csv](result%2Ftables%2F20250701-1057_model_v520105p_TOP-1_EVAL.csv) 📎

- Manually ✍️ **checked** evaluation dataset (TOP-3): [model_TOP-3_EVAL.csv](result%2Ftables%2F20250710-1928_model_v520105p_TOP-3_EVAL.csv) 📎

- **Unchecked with TRUE** values (small): [model_TOP-1.csv](result%2Ftables%2F20250710-1938_model_v520105p_TOP-1.csv)📎

Demo files  `v1.2`:

- Manually ✍️ **checked** evaluation dataset (TOP-1): [model_TOP-1_EVAL.csv](result%2Ftables%2F20250709-1825_model_v120106s_TOP-1_EVAL.csv) 📎

- Manually ✍️ **checked** evaluation dataset (TOP-3): [model_TOP-3_EVAL.csv](result%2Ftables%2F20250710-1924_model_v120106s_TOP-3_EVAL.csv) 📎

- **Unchecked with TRUE** values (small): [model_TOP-1.csv](result%2Ftables%2F20250710-1941_model_v120106s_TOP-1.csv)📎

Demo files  `v4.2`:

- Manually ✍️ **checked** evaluation dataset (TOP-1): [model_TOP-1_EVAL.csv](result%2Ftables%2F20250709-1823_model_v120106l_TOP-1_EVAL.csv) 📎

- Manually ✍️ **checked** evaluation dataset (TOP-3): [model_TOP-3_EVAL.csv](result%2Ftables%2F20250710-1921_model_v120106l_TOP-3_EVAL.csv) 📎

- **Unchecked with TRUE** values (small): [model_TOP-1.csv](result%2Ftables%2F20250710-1942_model_v120106l_TOP-1.csv)📎


With the following **columns** 📋:

- **FILE** - name of the file
- **PAGE** - number of the page
- **CLASS-N** - label of the category 🪧, guess TOP-N 
- **SCORE-N** - score of the category 🪧, guess TOP-N

and optionally
 
- **TRUE** - actual label of the category 🪧 

</details>

<details>

<summary>Raw result tables 👀</summary>

Demo files `v2.2`:

- Manually ✍️ **checked** evaluation dataset **RAW**: [model_RAW_EVAL.csv](result%2Ftables%2F20250710-1925_model_v220105p_EVAL_RAW.csv) 📎

- **Unchecked with TRUE** values (small) **RAW**: [model_RAW.csv](result%2Ftables%2F20250710-1939_model_v220105p_RAW.csv) 📎

Demo files `v3.2`:

- Manually ✍️ **checked** evaluation dataset **RAW**: [model_RAW_EVAL.csv](result%2Ftables%2F20250710-1925_model_v220105p_EVAL_RAW.csv) 📎

- **Unchecked with TRUE** values (small) **RAW**: [model_RAW.csv](result%2Ftables%2F20250710-1936_model_v320105p_RAW.csv) 📎

Demo files `v5.2`:

- Manually ✍️ **checked** evaluation dataset **RAW**: [model_RAW_EVAL.csv](result%2Ftables%2F20250710-1925_model_v220105p_EVAL_RAW.csv) 📎

- **Unchecked with TRUE** values (small) **RAW**: [model_RAW.csv](result%2Ftables%2F20250710-1938_model_v520105p_RAW.csv) 📎

Demo files `v1.2`:

- Manually ✍️ **checked** evaluation dataset **RAW**: [model_RAW_EVAL.csv](result%2Ftables%2F20250710-1925_model_v220105p_EVAL_RAW.csv) 📎

- **Unchecked with TRUE** values (small) **RAW**: [model_RAW.csv](result%2Ftables%2F20250710-1941_model_v120106s_RAW.csv) 📎

Demo files `v4.2`:

- Manually ✍️ **checked** evaluation dataset **RAW**: [model_RAW_EVAL.csv](result%2Ftables%2F20250710-1925_model_v220105p_EVAL_RAW.csv) 📎

- **Unchecked with TRUE** values (small) **RAW**: [model_RAW.csv](result%2Ftables%2F20250710-1942_model_v120106l_RAW.csv) 📎

With the following **columns** 📋:

- **FILE** - name of the file
- **PAGE** - number of the page
- **<CATEGORY_LABEL>** - separate columns for each of the defined classes 🪧 
- **TRUE** - actual label of the category 🪧 

</details>

The reason to use the `--raw` flag is the possible convenience of results review, 
since the rows will be basically sorted by categories, and most ambiguous ones will
have more small probabilities instead of zeros than the most obvious (for the model) 
categories 🪧.

----

## Data preparation 📦

You can use this section as a guide for creating your own dataset of pages, which will be suitable for
further model processing.

There are useful multiplatform scripts in the [data_scripts](data_scripts) 📁 folder for the whole process of data preparation. 

> [!NOTE]
> The `.sh` scripts are adapted for **Unix** OS and `.bat` scripts are adapted for **Windows** OS, yet 
> their functionality remains the same

On **Windows** you must also install the following software before converting PDF documents to PNG images:
- ImageMagick [^5] 🔗 - download and install the latest version
- Ghostscript [^6] 🔗 - download and install the latest version (32 or 64-bit) by AGPL

### PDF to PNG 📚

The source set of PDF documents must be converted to page-specific PNG images before processing. The following steps
describe the procedure of converting PDF documents to PNG images suitable for training, evaluation, or prediction inference.

Firstly, copy the PDF-to-PNG converter script to the directory with PDF documents.

<details>

<summary>How to 👀</summary>

 **Windows**:

    move \local\folder\for\this\project\data_scripts\pdf2png.bat \full\path\to\your\folder\with\pdf\files

**Unix**:

    cp /local/folder/for/this/project/data_scripts/pdf2png.sh /full/path/to/your/folder/with/pdf/files

</details>

Now check the content and comments in [pdf2png.sh](data_scripts%2Funix%2Fpdf2png.sh) 📎 or [pdf2png.bat](data_scripts%2Fwindows%2Fpdf2png.bat) 📎 
script, and run it. 

> [!IMPORTANT]
> You can optionally comment out the **removal of processed PDF files** from the script, yet it's **NOT** 
> recommended in case you are going to launch the program several times from the same location. 

<details>

<summary>How to 👀</summary>

**Windows**:

    cd \full\path\to\your\folder\with\pdf\files
    pdf2png.bat

**Unix**:

    cd /full/path/to/your/folder/with/pdf/files
    pdf2png.sh

</details>

After the program is done, you will have a directory full of document-specific subdirectories
containing page-specific images with a similar structure:

<details>

<summary>Unix folder tree 🌳 structure 👀</summary>

    /full/path/to/your/folder/with/pdf/files
    ├── PdfFile1Name
        ├── PdfFile1Name-001.png
        ├── PdfFile1Name-002.png
        └── ...
    ├── PdfFile2Name
        ├── PdfFile2Name-01.png
        ├── PDFFile2Name-02.png
        └── ...
    ├── PdfFile3Name
        └── PdfFile3Name-1.png 
    ├── PdfFile4Name
    └── ...

</details>

> [!NOTE]
> The page numbers are padded with zeros (on the left) to match the length of the last page number in each PDF file,
> this is done automatically by the pdftoppm command used on **Unix**. While ImageMagick's [^5] 🔗 convert command used 
> on **Windows** does **NOT** pad the page numbers.

<details>

<summary>Windows folder tree 🌳 structure 👀</summary>

    \full\path\to\your\folder\with\pdf\files
    ├── PdfFile1Name
        ├── PdfFile1Name-1.png
        ├── PdfFile1Name-2.png
        └── ...
    ├── PdfFile2Name
        ├── PdfFile2Name-1.png
        ├── PDFFile2Name-2.png
        └── ...
    ├── PdfFile3Name
        └── PdfFile3Name-1.png 
    ├── PdfFile4Name
    └── ...

</details>

Optionally you can use the [move_single.sh](data_scripts%2Funix%2Fmove_single.sh) 📎 or [move_single.bat](data_scripts%2Fwindows%2Fmove_single.bat) 📎 script to move 
all PNG files from directories with a single PNG file inside to the common directory of one-pagers. 

By default, the scripts assume that the `onepagers` is the back-off directory for PDF document names without a 
corresponding separate directory of PNG pages found in the PDF files directory (already converted to 
subdirectories of pages).

<details>

<summary>How to 👀</summary>

**Windows**:

    move \local\folder\for\this\project\atrium-page-classification\data_scripts\move_single.bat \full\path\to\your\folder\with\pdf\files
    cd \full\path\to\your\folder\with\pdf\files
    move_single.bat

**Unix**:
    
    cp /local/folder/for/this//project/atrium-page-classification/data_scripts/move_single.sh /full/path/to/your/folder/with/pdf/files
    cd /full/path/to/your/folder/with/pdf/files 
    move_single.sh 

</details>

The reason for such movement is simply convenience in the following annotation process [below](#png-pages-annotation-). 
These changes are cared for in the next [sort.sh](data_scripts%2Funix%2Fsort.sh) 📎 and [sort.bat](data_scripts%2Fwindows%2Fsort.bat) 📎 scripts as well.

### PNG pages annotation 🔎

The generated PNG images of document pages are used to form the annotated gold data. 

> [!NOTE]
> It takes a lot of time ⌛ to collect at least several hundred examples per category.

Prepare a CSV table with exactly 3 columns:

- **FILE** - name of the PDF document which was the source of this page
- **PAGE** - number of the page (**NOT** padded with 0s)
- **CLASS** - label of the category 🪧 

> [!TIP]
> Prepare equal-in-size categories 🪧 if possible, so that the model will not be biased towards the over-represented labels 🪧 

For **Windows** users, it's **NOT** recommended to use MS Excel for writing CSV tables, the free 
alternative may be Apache's OpenOffice [^9] 🔗. As for **Unix** users, the default LibreCalc should be enough to 
correctly write a comma-separated CSV table.

<details>

<summary>Table in .csv format example 👀</summary>

    FILE,PAGE,CLASS
    PdfFile1Name,1,Label1
    PdfFile2Name,9,Label1
    PdfFile1Name,11,Label3
    ...

</details>

### PNG pages sorting for training 📬

Cluster the annotated data into separate folders using the [sort.sh](data_scripts%2Funix%2Fsort.sh) 📎 or [sort.bat](data_scripts%2Fwindows%2Fsort.bat) 📎 
script to copy data from the source folder to the training folder where each category 🪧 has its own subdirectory.
This division of PNG images will be used as gold data in training and evaluation.

> [!WARNING]
> It does **NOT** matter from which directory you launch the sorting script, but you must check the top of the script for 
> (**1**) the path to the previously described **CSV table with annotations**, (**2**) the path to the previously described 
> directory containing **document-specific subdirectories of page-specific PNG pages**, and (**3**) the path to the directory
> where you want to store the **training data of label-specific directories with annotated page images**.

<details>

<summary>How to 👀</summary>

**Windows**:

    sort.bat

**Unix**:
    
    sort.sh

</details>

After the program is done, you will have a directory full of label-specific subdirectories 
containing document-specific pages with a similar structure:

<details>

<summary>Unix folder tree 🌳 structure 👀</summary>

    /full/path/to/your/folder/with/train/pages
    ├── Label1
        ├── PdfFileAName-00N.png
        ├── PdfFileBName-0M.png
        └── ...
    ├── Label2
    ├── Label3
    ├── Label4
    └── ...

</details>

<details>

<summary>Windows folder tree 🌳 structure 👀</summary>
    
    \full\path\to\your\folder\with\train\pages
    ├── Label1
        ├── PdfFileAName-N.png
        ├── PdfFileBName-M.png
        └── ...
    ├── Label2
    ├── Label3
    ├── Label4
    └── ...

</details>

The sorting script can help you in moderating mislabeled samples before the training. Accurate data annotation
directly affects the model performance. 

Before running the training, make sure to check the [config.txt](config.txt) ⚙️ file for the `[TRAIN]` section variables, where you should
set a path to the data folder. Make sure label directory names do **NOT** contain special characters like spaces, tabs or paragraph splits.

> [!TIP]
> In the [config.txt](config.txt) ⚙️ file tweak the parameter of `max_categ`
> for a maximum number of samples per category 🪧, in case you have **over-represented labels** significantly dominating in size.
> Set `max_categ` higher than the number of samples in the largest category 🪧 to use **all** data samples.

From this point, you can start model training or evaluation process.

----

## For developers 🪛

You can use this project code as a base for your own image classification tasks. The detailed guide on 
the key phases of the whole process (settings, training, evaluation) is provided here.

<details>

<summary>Project files description 📋👀</summary>

| File Name        | Description                                                                                                     |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `classifier.py`  | Model-specific classes and related functions including predefined values for training arguments                 |
| `utils.py`       | Task-related algorithms                                                                                         |
| `run.py`         | Starting point of the program with its main function - can be edited for flags and function argument extensions |
| `config.txt`     | Changeable variables for the program - should be edited                                                         |

</details>

Most of the changeable variables are in the [config.txt](config.txt) ⚙ file, specifically,
in the `[TRAIN]`, `[HF]`, and `[SETUP]` sections. 

In the dev sections of the configuration ⚙ file, you will find many boolean variables that can be changed from the default `False` 
state to `True`, yet it's recommended to awaken those variables solely through the specific 
**command line flags implemented for each of these boolean variables**.

For more detailed training process adjustments refer to the related functions in [classifier.py](classifier.py) 📎 
file, where you will find some predefined values not used in the [run.py](run.py) 📎 file.

> [!IMPORTANT]
> For both training and evaluation, you must make sure that the training pages directory is set right in the 
> [config.txt](config.txt) ⚙ and it contains category 🪧 subdirectories with images inside. 
> Names of the category 🪧 subdirectories are sorted in the alphabetic order and become actual
> label names and replace the default categories 🪧 list

Device 🖥️ requirements for training / evaluation:
- **CPU** of some kind and memory size
- **GPU** (for real CUDA [^10] support - better one of Nvidia's cards)

Worth mentioning that the efficient training is possible only with a CUDA-compatible GPU card.

<details>

<summary>Rough estimations of memory usage 👀</summary>

| **Batch size** | **CPU / GPU memory usage** |
|----------------|----------------------------|
| 4              | 2 Gb                       |
| 8              | 3 Gb                       |
| 16             | 5 Gb                       |
| 32             | 9 Gb                       |
| 64             | 17 Gb                      |

</details>

For test launches on the **CPU-only device 🖥️** you should set **batch size to lower than 4**, and even in this
case, **above-average CPU memory capacity** is a must-have to avoid a total system crush.

### Training 💪

To train the model run: 

    python3 run.py --train

The training process has an automatic progress logging into console, and should take approximately 5-12h 
depending on your machine's 🖥️ CPU / GPU memory size and prepared dataset size. 

> [!TIP]
> Run the training with **default hyperparameters** if you have at least ~10,000 and **less than 50,000 page samples** 
> of the very similar to the initial source data - meaning, no further changes are required for fine-tuning model 
> for the same task on an expanded (or new) dataset of document pages, even number of categories 🪧 does 
> **NOT** matter while it stays under **20**

<details>

<summary>Training hyperparameters 👀</summary>
 
* eval_strategy "epoch"
* save_strategy "epoch"
* learning_rate **5e-5**
* per_device_train_batch_size 8
* per_device_eval_batch_size 8
* num_train_epochs **3**
* warmup_ratio **0.1**
* logging_steps **10**
* load_best_model_at_end True
* metric_for_best_model "accuracy" 

</details>

Above are the default hyperparameters or TrainingArguments [^11] used in the training process that can be partially
(only `epoch` and `log_step`) changed in the `[TRAIN]` section, plus `batch` in the `[SETUP]`section, 
of the [config.txt](config.txt) ⚙ file.

> You are free to play with the **learning rate** right in the training function arguments called in the [run.py](run.py) 📎 file, 
> yet **warmup ratio and other hyperparameters** are accessible only through the [classifier.py](classifier.py) 📎 file.

Playing with training hyperparameters is
recommended only if **training 💪 loss** (error rate) descends too slow to reach 0.001-0.001
values by the end of the 3rd (last by default) epoch.

In the case **evaluation 🏆 loss** starts to steadily going up after the previous descend, this means
you have reached the limit of worthy epochs, and next time you should set `epochs` to the
number of epoch that has successfully ended before you noticed the evaluation loss growth.

During training image transformations [^12] are applied sequentially with a 50% chance.

> [!NOTE]
> No rotation, reshaping, or flipping was applied to the images, mainly color manipulations were used. The 
> reason behind this are pages containing specific form types, general text orientation on the pages, and the default
> reshape of the model input to the square 224x224 resolution images. 

<details>

<summary>Image preprocessing steps 👀</summary>

* transforms.ColorJitter(**brightness** 0.5)
* transforms.ColorJitter(**contrast** 0.5)
* transforms.ColorJitter(**saturation** 0.5)
* transforms.ColorJitter(**hue** 0.5)
* transforms.Lambda(lambda img: ImageEnhance.**Sharpness**(img).enhance(random.uniform(0.5, 1.5)))
* transforms.Lambda(lambda img: img.filter(ImageFilter.**GaussianBlur**(radius=random.uniform(0, 2))))

</details>

More about selecting the image transformation and the available ones you can read in the PyTorch torchvision docs [^12].

After training is complete the model will be saved 💾 to its separate subdirectory in the `model` directory, by default, 
the **naming of the model folder** corresponds to the length of its training batch dataloader and the number of epochs - 
for example `model_<S/B>_E` where `E` is the number of epochs, `B` is the batch size, and `S` is the size of your 
**training** dataset (by defaults, 90% of the provided in `[TRAIN]`'s folder data).

<details>

<summary>Full project tree 🌳 files structure 👀</summary>
    
    /local/folder/for/this/project/atrium-page-classification
    ├── model
        ├── movel_v<HFrevision1> 
            ├── config.json
            ├── model.safetensors
            └── preprocessor_config.json
        ├── movel_v<HFrevision2>
        └── ...
    ├── checkpoint
        ├── models--google--vit-base-patch16-224
            ├── blobs
            ├── snapshots
            └── refs
        └── .locs
            └── models--google--vit-base-patch16-224
    ├── model_output
        ├── checkpoint-version1
            ├── config.json
            ├── model.safetensors
            ├── trainer_state.json
            ├── optimizer.pt
            ├── scheduler.pt
            ├── rng_state.pth
            └── training_args.bin
        ├── checkpoint-version2
        └── ...
    ├── data_scripts
        ├── windows
        └── unix
    ├── result
        ├── plots
        └── tables
    ├── category_samples
        ├── DRAW
        ├── DRAW_L
        └── ...
    ├── run.py
    ├── classifier.py
    ├── utils.py
    └── ...

</details>

> [!IMPORTANT] 
> The `movel_<revision>` folder naming is generated from the HF 😊 repo [^1] 🔗 `revision` value and does **NOT** 
> affect the trained model naming, other training parameters do. 
> Since the length of the dataloader depends not only on the size of the dataset but also on the preset batch size, 
> and test subset ratio. 

You can slightly change the `test_size` and / or
the `batch` variable value in the [config.txt](config.txt) ⚙ file to train a differently named model on the same dataset.
Alternatively, adjust the **model naming generation** in the [classifier.py](classifier.py)'s 📎 training function.

In terms of the input data splitting, **this code is adapted to the filenames containing date stamps** which are leveraged
in the filenames sorting, and then randomized step selection, of separate categories 🪧 for the final evaluation and the 
training-time-evaluation (so-called, dev) subsets - both of the same `test_ratio` size. This behaviour is specifically
triggered when the `--cross` argument or `cross_runs` variable in the `[TRAIN]` section of the [config.txt](config.txt) ⚙ file 
is set above 1, as well as when the `--train` flag is used for a single run. 

> [!TIP]
> The cross-validation takes more time and reselects the data subsets for each run based on a `seed` variable of the
> `[SETUP]` section in the [config.txt](config.txt) ⚙ file which gets simply incremented by one for each fold (run) of the
> cross-validation process. The listed data splits are recorded as `.txt` files in the `result/stats` directory 📁 for 
> each fold of the overall model training run, as well as the fold's final test set predictions are saved in 
> `result/tables` directory 📁. The trained models are saved as model_<revision><fold>.

### Evaluation 🏆

After the fine-tuned model is saved 💾, you can explicitly call for evaluation of the model to get a table of TOP-N classes for
the semi-randomly composed subset (10% in size by default) of the training page folder. The class proportions are preserved, 
and the data is uniformly spread across the time span of the provided dataset.

There is an option of setting `test_size` to 0.4 and use all the sorted by category pages provided 
in `[TRAIN]`'s folder for evaluation, but do **NOT** launch it on the whole training data you have actually used up
for the evaluated model training.

To do this in the unchanged configuration ⚙, automatically create a 
confusion matrix plot 📊 and additionally get raw class probabilities table run: 

    python3 run.py --eval --raw

**OR** when you don't remember the specific `[SETUP]` and `[TRAIN]` variables' values for the trained model, you can use:

    python3 run.py --eval -m './model/model_<your_model_number_code>'

Finally, when your model is trained and you are happy with its performance tests, you can uncomment a code line
in the [run.py](run.py) 📎 file for **HF 😊 hub model push**. This functionality has already been implemented and can be
accessed through the `--hf` flag using the values set in the `[HF]` section for the `token` and `repo_name` variables.

In this case, you must **rename the trained model folder** in respect to the `revision` value (dots in the naming are skipped, e.g. 
revision `v1.9.22` turns to `model_v1922` model folder), and only then run repo push.

> [!CAUTION]
> Set your own `repo_name` to the empty one of yours on HF 😊 hub, then in the **Settings** of your HF 😊 account
> find the **Access Tokens** section and generate a new token - copy and paste its value to the `token` variable. Before committing 
> those [config.txt](config.txt) ⚙ file changes via git replace the full `token` value with its shortened version for security reasons.

----

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

> Information about the authors of this project, including their names and ORCIDs, can 
> be found in the [CITATION.cff](CITATION.cff) 📎 file.

## Acknowledgements 🙏

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4]  💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗
- **Model type:** 
  - fine-tuned ViT with a 224x224 [^2] 🔗 or 384x384 [^13] [^14] 🔗 resolution size 
  - fine-tuned EffNetV2 with a 300x300 [^15] 🔗 or 384x384 [^16] 🔗 resolution size 

**©️ 2022 UFAL & ATRIUM**

----

## Appendix 🤓

<details>

<summary>README emoji codes 👀</summary>

- 🖥 - your computer
- 🪧 - label/category/class
- 📄 - page/file
- 📁 - folder/directory
- 📊 - generated diagrams or plots
- 🌳 - tree of file structure
- ⌛ - time-consuming process
- ✍️ - manual action
- 🏆 - performance measurement
- 😊 - Hugging Face (HF)
- 📧 - contacts 
- 👀 - click to see
- ⚙️ - configuration/settings
- 📎 - link to the internal file
- 🔗 - link to the external website

</details>

<details>

<summary>Content specific emoji codes 👀</summary>

- 📏 - table content
- 📈 - drawings/paintings/diagrams
- 🌄 - photos
- ✏️ - handwritten content
- 📄 - text content
- 📰 - mixed types of text content, maybe with graphics

</details>

<details>

<summary>Decorative emojis 👀</summary>

- 📇📜🔧▶🪄🪛️📦🔎📚🙏👥📬🤓 - decorative purpose only

</details>

> [!TIP]
> Alternative version of this README file is available in [README.html](README.html) 📎 webpage

[^1]: https://huggingface.co/ufal/vit-historical-page
[^2]: https://huggingface.co/google/vit-base-patch16-224
[^3]: https://docs.python.org/3/library/venv.html
[^4]: https://atrium-research.eu/
[^5]: https://imagemagick.org/script/download.php#windows
[^6]: https://www.ghostscript.com/releases/gsdnld.html
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-page-classification
[^9]: https://www.openoffice.org/download/
[^10]: https://developer.nvidia.com/cuda-python
[^11]: https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
[^12]: https://pytorch.org/vision/0.20/transforms.html
[^13]: https://huggingface.co/google/vit-base-patch16-384
[^14]: https://huggingface.co/google/vit-large-patch16-384
[^15]: https://huggingface.co/timm/tf_efficientnetv2_s.in21k
[^16]: https://huggingface.co/timm/tf_efficientnetv2_l.in21k_ft_in1k
[^17]: http://hdl.handle.net/20.500.12800/1-5959
