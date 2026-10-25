# The eleven page categories

For anyone who reads a result, annotates pages or trains a model: what each of the eleven labels means, how to read
a label, and why the order of the labels matters.

This page is the one place the categories are defined. The shared vocabulary module
[atrium_vocab.py](../../atrium_vocab.py) copies the definitions below word for word into its `page-category` concept
scheme, and [service/README.md](../../service/README.md) and the hub link here.

## Definitions

| Label     | Definition                                                                                              | In short                                             |
|-----------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| `DRAW`    | drawings, maps, paintings, schematics, or graphics, potentially containing some text labels or captions | Drawing (maps, paintings, schematics)                |
| `DRAW_L`  | drawings, etc but presented within a table-like layout or includes a legend formatted as a table        | Structured drawing (drawings within a layout/legend) |
| `LINE_HW` | handwritten text organized in a tabular or form-like structure                                          | Handwritten table                                    |
| `LINE_P`  | printed text organized in a tabular or form-like structure                                              | Printed table                                        |
| `LINE_T`  | machine-typed text organized in a tabular or form-like structure                                        | Typed table                                          |
| `PHOTO`   | photographs or photographic cutouts, potentially with text captions                                     | Photo (photographs/cutouts)                          |
| `PHOTO_L` | photos presented within a table-like layout or accompanied by tabular annotations                       | Structured photo (photos in a table layout)          |
| `TEXT`    | mixtures of printed, handwritten, and/or typed text, potentially with minor graphical elements          | Mixed text (printed, handwritten, typed)             |
| `TEXT_HW` | only handwritten text in paragraph or block form (non-tabular)                                          | Handwritten text (paragraphs)                        |
| `TEXT_P`  | only printed text in paragraph or block form (non-tabular)                                              | Printed text (published paragraphs)                  |
| `TEXT_T`  | only machine-typed text in paragraph or block form (non-tabular)                                        | Typed text (machine-typed paragraphs)                |

The short forms were the service README's category table; the service's web interface
([service/frontend/script.js](../../service/frontend/script.js)) shows similar ones.

## Why these categories

The categories were chosen to sort the pages by three criteria:

- **presence of graphical elements** (drawings or photos);
- **type of text** (handwritten, printed, typed or mixed);
- **presence of a tabular layout or a form.**

The reason for the distinction is that different kinds of page go to different processing pipelines after the
classification: OCR for printed or typed text, handwritten text recognition, table extraction, image extraction.
Which category suggests which next step is on the hub's
[workflow page](https://ufal.github.io/atrium-project/workflows/page-classification/); the classifier does not route
pages itself.

## Reading a label

The stem names the dominant content, and the suffix qualifies it:

| Stem    | Meaning                 | Suffixes                                                                          |
|---------|-------------------------|-----------------------------------------------------------------------------------|
| `TEXT`  | running text            | `_HW` handwritten · `_P` printed · `_T` typewritten · none: a mixture             |
| `LINE`  | text in a table or form | `_HW` handwritten · `_P` printed · `_T` typewritten                               |
| `DRAW`  | drawings, maps, plans   | `_L` inside a table-like layout or with a tabular legend · none: free-standing    |
| `PHOTO` | photographs             | `_L` inside a table-like layout or with tabular annotations · none: free-standing |

## The order of the labels

`model_registry.CATEGORIES` declares the labels in the order of the table above, which is alphabetical. That order
is the label-to-index binding every checkpoint was trained against: reordering the list would silently relabel every
prediction. Training and evaluation read the category list from the folder names of the training tree, sorted
alphabetically, which gives the same order; inference always uses the fixed list. The hub declares the same labels,
in the same order, as `PAGE_CATEGORIES` in `atrium_vocab.py`.

Each label has a stable SKOS identifier, such as `https://w3id.org/atrium/page-category/TEXT_HW`. Records carry the
bare label; `model_registry.category_uri()` turns it into the URI
([SKOS and the ATRIUM vocabulary](https://ufal.github.io/atrium-project/contracts/skos/)).

The document schema deliberately has no `enum` for the category field. `validate_document()` is a live output gate
that raises, so an enum would turn a single labelling slip into a stalled pipeline; the vocabulary registry reports
an unknown label instead of refusing it.

## Examples

Example pages of each category are in [small_data_samples](../../small_data_samples), one folder per category
(CC BY-NC 4.0, its [LICENSE](../../small_data_samples/LICENSE)). The folder can be used to try prediction and
evaluation; process it with `--inner`. Its pages come from theses, slides, papers and web pages, outside the training
data, and their folders were never checked as evaluation labels ([result/README.md](../../result/README.md)).
