import pandas as pd
import layoutparser as lp
import cv2
from common_utils import *

# PNG files layout extraction
class PNG_Layout:
    def __init__(self, output_folder=os.environ['FOLDER_LAYOUTS']):
        self.layout_output_folder = output_folder

        self.png_file_list = []

        pl_model_path = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config"
        pl_labelmap = {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion",
                       6: "OtherRegion"}

        self.model = lp.Detectron2LayoutModel(config_path=pl_model_path, label_map=pl_labelmap,
                                            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])

        if not os.path.exists(self.layout_output_folder):
            os.makedirs(self.layout_output_folder)

    # PNG layout detection
    def _png_detect(self, png_file: str):
        image = cv2.imread(png_file)
        # print(image)
        # cv2.imshow("img", image)
        image = image[..., ::-1]

        layout_pl = self.model.detect(image)
        # img = lp.draw_box(img, layout_pl, box_width=3,
        #                   show_element_type=True, color_map=pl_palette)
        # img.show()

        df = layout_pl.to_dataframe()
        # print(df)
        # layout = lp.io.load_dataframe(df)
        # print(layout)

        layout_file_name = f"{self.layout_output_folder}/{png_file.split('/')[-1].split('.')[0]}.txt"
        df.to_csv(layout_file_name, sep="\t", index=False)
        print(f"Layout of {png_file.split('/')[-1]} containing {len(df.index)} blocks saved to {layout_file_name}")
        return df

    # called to process directory path
    def folder_to_layouts(self, path: str) -> None:
        pdf_list = directory_scraper(path, self.png_file_list)

        for file in pdf_list:
            self._png_detect(file)

# possible configs for found layout parser (worse than chosen according to manual tests)

# hjd_path = "lp://HJDataset/mask_rcnn_R_50_FPN_3x/config"
# pln_path = "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"
# pl_path = "lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config"

# hjd_labelmap = {1: "Page Frame", 2: "Row", 3: "Title Region", 4: "Text Region", 5: "Title", 6: "Subtitle",
#                 7: "Other"}
# pln_labelmap = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
# pl_labelmap = {1: "TextRegion", 2: "ImageRegion", 3: "TableRegion", 4: "MathsRegion", 5: "SeparatorRegion",
#                6: "OtherRegion"}

# hjd_model = lp.Detectron2LayoutModel(config_path=hjd_path, label_map=hjd_labelmap, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
# pln_model = lp.Detectron2LayoutModel(config_path=pln_path, label_map=pln_labelmap, extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])
# pl_model = lp.Detectron2LayoutModel(config_path=pl_path, label_map=pl_labelmap,
#                                     extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8])

# hjd_palette = {"Page Frame": "red", "Row": "red", "Title Region": "red", "Text Region": "red", "Title": "red",
#                "Subtitle": "red", "Other": "red"}
# pln_palette = {"Text": "green", "Title": "green", "List": "green", "Table": "green", "Figure": "green"}
# pl_palette = {"TextRegion": "blue", "ImageRegion": "blue", "TableRegion": "blue", "MathsRegion": "blue",
#               "SeparatorRegion": "blue", "OtherRegion": "blue"}
