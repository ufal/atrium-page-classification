# Image classification using finetuned models - for historical document sorting

### Goal: solve a task of archival page images classification based on their content (for further processing with relevant tools)

**Scope:** Processing of images, training / evaluation of models,
input file / directory processing, class (category) results of top-N 
predictions output, predictions summarizing into a tabular format, 
HF 😊 hub support for the best models, multiplatform (Windows / Unix) data 
preparation scripts for PDF conversion to PNGs / JPEGs of its pages.

#### Branches of the GitHub repository (and directories in the current branch):

- **CLIP** - 8 base models (5 are ViT-B/16 variants): [local README.md](clip/README.md) and [branch](https://github.com/ufal/atrium-page-classification/tree/clip/README.md)
  - HF 😊 hub: [https://huggingface.co/ufal/clip-historical-page](https://huggingface.co/ufal/clip-historical-page)

- **ViT** - 3 base models (plus EffNetV2-M & RegNetY-16GF): [local README.md](vit/README.md) and [branch](https://github.com/ufal/atrium-page-classification/tree/vit/README.md)
  - HF 😊 hub: [https://huggingface.co/ufal/vit-historical-page](https://huggingface.co/ufal/vit-historical-page)

## Contacts 📧

**For support write to:** lutsai.k@gmail.com responsible for this GitHub repository [^8] 🔗

> Information about the authors of this project, including their names and ORCIDs, can 
> be found in the [CITATION.cff](CITATION.cff) 📎 file.

## Acknowledgements 🙏

- **Developed by** UFAL [^7] 👥
- **Funded by** ATRIUM [^4]  💰
- **Shared by** ATRIUM [^4] & UFAL [^7] 🔗

**©️ 2022 UFAL & ATRIUM**

[^4]: https://atrium-research.eu/
[^7]: https://ufal.mff.cuni.cz/home-page
[^8]: https://github.com/ufal/atrium-page-classification
