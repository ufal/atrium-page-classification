# Image classification using fine-tuned ViT - for historical :bowtie: documents sorting

### Goal: solve a task of archive page images sorting (for their further content-based processing)

**Scope:** Processing of images, training and evaluation of ViT model,
input file/directory processing, class 🏷️ (category) results of top
N predictions output, predictions summarizing into a tabular format, 
HF 😊 hub support for the model, data preparation scripts for PDF to PNG conversion

### Table of contents 📑

  * [Model description 📇](#model-description-)
    + [Data 📜](#data-)
    + [Categories 🏷️](#categories-)
  * [How to install 🔧](#how-to-install-)
  * [How to run ▶️](#how-to-run-)
    + [Page processing 📄](#page-processing-)
    + [Directory processing 📁](#directory-processing-)
  * [Results 📊](#results-)
      - [Result tables 📏](#result-tables-)
      - [Table columns 📋](#table-columns-)
  * [For developers 🛠️](#for-developers-)
  * [Data preparation 📦](#data-preparation-)
    + [PDF to PNG 📚](#pdf-to-png-)
    + [PNG pages annotation 🔎](#png-pages-annotation-)
    + [PNG pages sorting for training 📬](#png-pages-sorting-for-training-)
  * [Contacts 📧](#contacts-)
  * [Acknowledgements 🙏](#acknowledgements-)

----

## Model description 📇

🔲 Fine-tuned model repository: **ufal's vit-historical-page** [^1] 🔗

🔳 Base model repository: **google's vit-base-patch16-224** [^2] 🔗

### Data 📜

Training set of the model: **8950** images 

### Categories 🏷️

|      Label️ |  Ratio  | Description                                                                  |
|------------:|:-------:|:-----------------------------------------------------------------------------|
|    **DRAW** | 	11.89% | **📈 - drawings, maps, paintings with text**                                 |
|  **DRAW_L** | 	8.17%  | **📈📏 - drawings ... with a table legend or inside tabular layout / forms** |
| **LINE_HW** |  5.99%  | **✏️📏 - handwritten text lines inside tabular layout / forms**              |
|  **LINE_P** | 	6.06%  | **📏 - printed text lines inside tabular layout / forms**                    |
|  **LINE_T** | 	13.39% | **📏 - machine typed text lines inside tabular layout / forms**              |
|   **PHOTO** | 	10.21% | **🌄 - photos with text**                                                    |
| **PHOTO_L** |  7.86%  | **🌄📏 - photos inside tabular layout / forms or with a tabular annotation** |
|    **TEXT** | 	8.58%  | **📰 - mixed types of printed and handwritten texts**                        |
| **TEXT_HW** |  7.36%  | **✏️📄 - only handwritten text**                                             |
|  **TEXT_P** | 	6.95%  | **📄 - only printed text**                                                   |
|  **TEXT_T** | 	13.53% | **📄 - only machine typed text**                                             |

Evaluation set (10% of the all, with the same proportions as above) [model_EVAL.csv](result%2Ftables%2F20250209-1534_model_1119_3_EVAL.csv) 📎:	**995** images 

----

## How to install 🔧

> [!WARNING]
> Make sure you have **Python version 3.10+** installed on your machine 💻. 
> Then create a separate virtual environment for this project 

Clone this project to your local machine 🖥️ via:

    cd /local/folder/for/this/project
    git init
    git clone https://github.com/ufal/atrium-page-classification.git

Follow the **Linux** / **Windows**-specific instruction at the venv docs [^3] 👀🔗 if you don't know how to.
After creating the venv folder, activate the environment via:

    source <your_venv_dir>/bin/activate

and then inside your virtual environment, you should install python libraries (takes time ⌛) 

> [!NOTE]
> Up to **1 GB of space for model** files and checkpoints is needed, and up to **7 GB 
> of space for the python libraries** (pytorch and its dependencies, etc)

Can be done via:

    pip install -r requirements.txt

To test that everything works okay and see the flag descriptions call for **--help** ❓:

    python3 run.py -h

To **pull the model from the HF 😊 hub repository directly**, load the model via:

    python3 run.py --hf

You should see a message about loading the model from hub and then saving it locally. 
Only after you have obtained the trained model files (takes less time ⌛ than installing dependencies), 
you can play with any commands provided below.

> [!IMPORTANT]
> Unless you already have the model files in the **'model/model_version'**
directory next to this file, you must use the **--hf** flag to download the
> model files from the HF 😊 repo [^1] 🔗

----

## How to run ▶️

Open [config.txt](config.txt) ⚙ and change folder path in the **\[INPUT\]** section, then 
optionally change **top_N** and **batch** in the **\[SETUP\]** section.

> [!NOTE]
>️ **Top-3** is enough to cover most of the images, setting **Top-5** will help with a small number 
> of difficult to classify samples.

> [!CAUTION]
> Do not try to change **base_model** and other section contents unless you know what you are doing

### Page processing 📄

Run the program from its starting point [run.py](run.py) 📎 with optional flags:

    python3 run.py -tn 3 -f '/full/path/to/file.png' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses 

> [!NOTE]
> Console output and all result tables contain **normalized** scores for the highest N class 🏷️ scores

**OR** if you are sure about default variables set in the [config.txt](config.txt) ⚙:

    python3 run.py -f '/full/path/to/file.png'

to run single PNG file classification - the output will be in the console. 

### Directory processing 📁

    python3 run.py -tn 3 -d '/full/path/to/directory' -m '/full/path/to/model/folder'

for exactly TOP-3 guesses from all images found in the given directory.

**OR** if you are really sure about default variables set in the [config.txt](config.txt) ⚙:

    python3 run.py --dir 

The classification results of PNG pages collected from the directory will be saved 💾 to related [results](result) 📁
folders defined in **\[OUTPUT\]** section of [config.txt](config.txt) ⚙ file.

> [!TIP]
> To additionally get raw class 🏷️ probabilities from the model along with the TOP-N results, use
> **--raw** flag when processing the directory
 
> [!TIP]
> To process all PNG files in the directory **AND its subdirectories** use the **--inner** flag
> when processing the directory

----

## Results 📊

<details>

<summary>Confusion matrix plots 📊</summary>

Evaluation set's accuracy (**Top-3**):  **99.6%** 🏆

![TOP-3 confusion matrix](result%2Fplots%2F20250209-1526_conf_mat.png)

Evaluation set's accuracy (**Top-1**):  **97.3%** 🏆

![TOP-1 confusion matrix](result%2Fplots%2F20250218-1523_conf_mat.png)

</details>

#### Result tables 📏

<details>

<summary>Examples of the result tables 👀</summary>

- Example of the manually ✍ **checked** results (small): [model_TOP-5.csv](result%2Ftables%2Fmodel_1119_3_TOP-5.csv) 📎

- Example of the manually ✍ **checked** evaluation dataset results (TOP-3): [model_TOP-3_EVAL.csv](result%2Ftables%2F20250209-1534_model_1119_3_TOP-3_EVAL.csv) 📎

- Example of the manually ✍ **checked** evaluation dataset **RAW** results [model_RAW_EVAL.csv](result%2Ftables%2F20250220-1342_model_1119_3_EVAL_RAW.csv) 📎

- Example of the manually ✍ **checked** evaluation dataset results (TOP-1): [model_TOP-1_EVAL.csv](result%2Ftables%2F20250218-1519_model_1119_3_TOP-1_EVAL.csv) 📎

- Example of the **unchecked with TRUE** values results: [model_TOP-3.csv](result%2Ftables%2F20250210-2034_model_1119_3_TOP-3.csv) 📎

- Example of the **unchecked with TRUE** values **RAW** results: [model_RAW.csv](result%2Ftables%2F20250220-1331_model_1119_3_RAW.csv) 📎

</details>

#### Table columns 📋

<details>

<summary>General result columns 👀</summary>

- **FILE** - name of the file
- **PAGE** - number of the page
- **CLASS-N** - label of the category 🏷️, guess TOP-N 
- **SCORE-N** - score of the category 🏷️, guess TOP-N

and optionally
 
- **TRUE** - actual label of the category 🏷️

</details>

<details>

<summary>Raw result columns 👀</summary>

- **FILE** - name of the file
- **PAGE** - number of the page
- **<CATEGORY_LABEL>** - separate columns for each of the defined classes 🏷️
- **TRUE** - actual label of the category 🏷️

</details>

> The reason to use **--raw** flag is possible convenience of results review, 
> since the most ambiguous cases are expected to be at the bottom of the table sorted in
> descending order by all **<CATEGORY_LABEL>** columns, while the most obvious (for the model)
> cases are expected to be at the top.

----

## For developers 🛠️

<details>

<summary>File details 👀</summary>

Most of the changeable variables are in the [config.txt](config.txt) ⚙ file, specifically,
in the **\[TRAIN\]**, **\[HF\]**, and **\[SETUP\]** sections.

For more detailed training process adjustments refer to the related functions in [classifier.py](classifier.py) 📎 
file, where you will find some predefined values not used in the [run.py](run.py) 📎 file.

| File Name        | Description                                                                                                     |
|------------------|-----------------------------------------------------------------------------------------------------------------|
| `classifier.py`  | Contains model-specific classes and related functions. Also includes predefined values for training adjustments |
| `utils.py`       | Contains task-related algorithms                                                                                |
| `run.py`         | Starting point of the program with its main function. Can be edited for flags and function argument extensions  |
| `config.txt`     | Contains all the changeable variables for the program. Should be edited                                         |

</details>

To train the model run: 

    python3 run.py --train  

To evaluate the model and create a confusion matrix plot 📊 run: 

    python3 run.py --eval  

> [!IMPORTANT]
> In both cases, you must make sure that training data directory is set right in the 
> [config.txt](config.txt) ⚙ and it contains category 🏷️ subdirectories with images inside. 
> Names of the category 🏷️ subdirectories become actual label names, and replaces the default categories 🏷️ list.

----

## Data preparation 📦

There are useful multiplatform :accessibility: scripts in the [data_scripts](data_scripts) 📁 folder for the whole process of data preparation. 

> [!NOTE]
> The .sh scripts are adapted for **Unix** OS and .bat scripts are adapted for **Windows** OS

On **Windows** you must also install the following software before converting PDF documents to PNG images:
- ImageMagick [^5] 🔗 - download and install latest version
- Ghostscript [^6] 🔗 - download and install latest version (32 or 64 bit) by AGPL

### PDF to PNG 📚

The source set of PDF documents must be converted to page-specific PNG images.

Firstly, copy the PDF-to-PNG converter script to the directory with PDF documents.

For **Windows**:

    move \local\folder\for\this\project\data_scripts\pdf2png.bat \full\path\to\your\folder\with\pdf\files

For **Unix**:

    cp /local/folder/for/this/project/data_scripts/pdf2png.sh /full/path/to/your/folder/with/pdf/files


Now check the content and comments in [pdf2png.sh](data_scripts%2Funix%2Fpdf2png.sh) 📎 or [pdf2png.bat](data_scripts%2Fwindows%2Fpdf2png.bat) 📎 
script, and run it.

For **Windows**:

    cd \full\path\to\your\folder\with\pdf\files
    pdf2png.bat

For **Unix**:

    cd /full/path/to/your/folder/with/pdf/files
    pdf2png.sh

After the program is done, you will have a directory full of document-specific subdirectories
containing page-specific images with a similar structure:

<details>

<summary>Unix folder structure 👀</summary>

    /full/path/to/your/folder/with/pdf/files
    ├── PdfFile1Name
        ├── PdfFile1Name-001.png
        ├── PdfFile1Name-002.png
        ...
    ├── PdfFile2Name
        ├── PdfFile2Name-01.png
        ├── PDFFile2Name-02.png
        ...
    ├── PdfFile3Name
        ├── PdfFile3Name-1.png 
    ├── PdfFile4Name
    ...

</details>

> [!NOTE]
> The page numbers are padded with zeros (on the left) to match the length of the last page number in each PDF file,
> this is done automatically by the pdftoppm command used on **Unix**. While ImageMagick's convert command used 
> on **Windows** does not pad the page numbers.

<details>

<summary>Windows folder structure 👀</summary>

    \full\path\to\your\folder\with\pdf\files
    ├── PdfFile1Name
        ├── PdfFile1Name-1.png
        ├── PdfFile1Name-2.png
        ...
    ├── PdfFile2Name
        ├── PdfFile2Name-1.png
        ├── PDFFile2Name-2.png
        ...
    ├── PdfFile3Name
        ├── PdfFile3Name-1.png 
    ├── PdfFile4Name
    ...

</details>

Optionally you can use the [move_single.sh](data_scripts%2Funix%2Fmove_single.sh) 📎 or [move_single.bat](data_scripts%2Fwindows%2Fmove_single.bat) 📎 script to move 
all PNG files from directories with a single PNG file inside to the common directory of one-pagers.

For **Windows**:

    move \local\folder\for\this\project\data_scripts\move_single.bat \full\path\to\your\folder\with\pdf\files
    cd \full\path\to\your\folder\with\pdf\files
    move_single.bat

For **Unix**:
    
    cp /local/folder/for/this//project/data_scripts/move_single.sh /full/path/to/your/folder/with/pdf/files
    cd /full/path/to/your/folder/with/pdf/files 
    move_single.sh 

The reason for such movement is simply convenience in the following annotation process. 
These changes are cared for in the next [sort.sh](data_scripts%2Funix%2Fsort.sh) 📎 and [sort.bat](data_scripts%2Fwindows%2Fsort.bat) 📎 scripts as well.

### PNG pages annotation 🔎

Prepare a CSV table with such columns:

- **FILE** - name of the PDF document which was the source of this page
- **PAGE** - number of the page (**NOT** padded with 0s)
- **CLASS** - label of the category 🏷️

> [!TIP]
> Prepare equal in size categories 🏷️ if possible, so that the model will not be biased towards the over-represented labels 🏷️

### PNG pages sorting for training 📬

Cluster the annotated data into separate folders using the [sort.sh](data_scripts%2Funix%2Fsort.sh) 📎 or [sort.bat](data_scripts%2Fwindows%2Fsort.bat) 📎 
script to copy data from the source folder to the training folder where each category 🏷️ has its own subdirectory:

For **Windows**:

    sort.bat

For **Unix**:
    
    sort.sh

> [!WARNING]
> It does not matter from which directory you launch the sorting script, but you must check the top of the script for 
> the path to the CSV table with annotations, path to the directory containing document-specific
> subdirectories of page-specific PNG pages, and path to the directory where you want to store the training data of
> label-specific directories with annotated page images.

After the program is done, you will have a directory full of label-specific subdirectories 
containing document-specific pages with a similar structure:

<details>

<summary>Unix folder structure 👀</summary>

    /full/path/to/your/folder/with/train/pages
    ├── Label1
        ├── PdfFileAName-00N.png
        ├── PdfFileBName-0M.png
        ...
    ├── Label2
    ├── Label3
    ├── Label4
    ...

</details>

<details>

<summary>Windows folder structure 👀</summary>
    
    \full\path\to\your\folder\with\train\pages
    ├── Label1
        ├── PdfFileAName-N.png
        ├── PdfFileBName-M.png
        ...
    ├── Label2
    ├── Label3
    ├── Label4
    ...

</details>

Before running the training, make sure to check the [config.txt](config.txt) ⚙️ file for the **\[TRAIN\]** section variables, where you should
set a path to the data folder. 

> Optionally, in the [config.txt](config.txt) ⚙️ file tweak the parameter of **max_categ**
> for maximum number of samples per category 🏷️, in case you have over-represented labels️ significantly dominating in size.
> Set **max_categ** higher than the number of samples in the largest category 🏷️ to use **all** data samples.

----

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for this repository [^8]

## Acknowledgements 🙏

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4]  💰
- **Shared by** ATRIUM [^4] & UFAL [^7]
- **Model type:** fine-tuned ViT with a 224x224 resolution size [^2]

**©️ 2022 UFAL & ATRIUM**

[^1]: https://huggingface.co/ufal/vit-historical-page
[^2]: https://huggingface.co/google/vit-base-patch16-224
[^3]: https://docs.python.org/3/library/venv.html
[^4]: https://atrium-research.eu/
[^5]: https://imagemagick.org/script/download.php#windows
[^6]: https://www.ghostscript.com/releases/gsdnld.html
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-page-classification
