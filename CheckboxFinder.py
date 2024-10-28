from PIL import Image, ImageOps
import numpy as np
import scipy as sp
from scipy import optimize
import os
import math
import itertools
import pandas as pd

class CheckboxFinder():

    """ Reference values for top-left position for a checbox (dimension 65x65) """
    _reference_sheet_checkbox_positions = [
    [[1243+ x*233, 796] for x in range(0, 5)],
    [[1243+ x*233, 884] for x in range(0, 5)],
    [[1243+x*233, 989] for x in range(0, 5)],
    [[1243+x*233, 1143] for x in range(0, 5)],
    [[1243+x*233, 1283] for x in range(0, 5)],
    [[1243+x*233, 1418] for x in range(0, 5)],
    [[1243+x*233, 1608] for x in range(0, 5)],
    [[1243+x*233, 1728] for x in range(0, 5)],
    [[1243+x*233, 1848] for x in range(0, 5)],
    [[1243+x*233, 2043] for x in range(0, 5)],
    [[1243+x*233, 2131] for x in range(0, 5)],
    [[1243+x*233, 2316] for x in range(0, 5)],
    [[1213, 2524], [1383, 2522], [1593, 2522], [1853, 2523], [2183, 2521]],
    [[1243+x*233, 2706] for x in range(0, 5)]]

    _reference_corner_positions = [
    [1127, 381], # upper left
    [2313, 372], # upper right
    [2311, 2741], # lower right
    [1125, 2746] # lower left
    ]

    _corner_name_map = {
        0: "ul.png",
        1: "ur.png",
        2: "lr.png",
        3: "ll.png"
    }

    def __init__(self, path_to_sheets, path_to_reference_sheet="./Evaluationshelfer_Daten/boegen/Bogen1.jpg", path_to_reference_corners="./Evaluationshelfer_Daten/masks/"):

        self._path_to_sheets = path_to_sheets

        self._reference_sheet = Image.open(path_to_reference_sheet).convert("RGBA")

        self._path_to_reference_corners = path_to_reference_corners

        self._checkbox_positions = {}


    def calculate_checkbox_positions(self, sheets):
        """ Calculates checkbox positions sheets.

            Arguments:
            sheets -- List of unique filenames of sheets
        """
        for sheet in sheets:
            if sheet not in self._checkbox_positions:
                corner_positions = self._find_corner_positions(sheet)
                C_matrix = self._build_C_matrix()
                d_vector = self._build_d_vector(corner_positions)

                result_x_Ab = optimize.minimize(self._epsilon, [1, 0, 0, 1, 0, 0], args=(C_matrix, d_vector), method="Powell", tol=0.001).x

                checkbox_positions = []

                for question in range(14):
                    checkbox_positions.append([])
                    for rating in range(5):
                        trans_x, trans_y = \
                            self._phi(self._reference_sheet_checkbox_positions[question][rating],\
                                      result_x_Ab)
                        checkbox_positions[question].append([round(trans_x), round(trans_y)])

                self._checkbox_positions[sheet] = checkbox_positions



    def get_checkbox_image(self, sheet, question, rating):
        """ Returns PIL-Image of cut out checkbox

            Arguments:
            sheet -- Filename of sheet
            question -- question number of checkbox. Ranges from 1 to 14
            rating -- rating number of checkbox. Ranges from 0 (++) to 4 (--)
        """
        assert 1 <= question <= 14, "Question number out of range"
        assert 0 <= rating <= 4, "Rating number out of range"

        bogen_image = Image.open(self._path_to_sheets + sheet).convert("RGBA")

        if sheet not in self._checkbox_positions:
            self.calculate_checkbox_positions([sheet])

        checkbox_position = self._checkbox_positions[sheet][question-1][rating]

        return bogen_image.crop((checkbox_position[0], checkbox_position[1],\
                                 checkbox_position[0]+65, checkbox_position[1]+65)).resize((40, 40))


    def _image_difference(self, image_a, image_b):
        width, height = min(image_a.size[0], image_b.size[0]), min(image_a.size[1], image_b.size[1])

        # Crop both images to same size and convert to grayscale
        image_a = ImageOps.grayscale(image_a.crop((0, 0, width, height)))
        image_b = ImageOps.grayscale(image_b.crop((0, 0, width, height)))

        # Convert images to numpy matricies
        image_a_matrix = np.asarray(image_a, dtype="double")
        image_b_matrix = np.asarray(image_b, dtype="double")

        assert image_a_matrix.shape == image_b_matrix.shape, "Image-Matricies don't have same shape"

        # Return squared pixelwise difference
        return np.sum((image_a_matrix - image_b_matrix)**2)

    def _find_corner_position(self, sheet_image, corner_no, radius):
        search_center = self._reference_corner_positions[corner_no]
        search_positions = itertools.product(*[range(p-radius, p+radius) for p in search_center])

        best_match_value = math.inf # Minimize this
        best_match_pos = [0, 0]

        corner_img = Image.open(self._path_to_reference_corners + self._corner_name_map[corner_no]).convert("RGBA")


        for pos in search_positions:
            cropped_bogen = sheet_image.crop((pos[0], pos[1], pos[0]+100, pos[1]+100)) # gets cropped again anyways
            match_val = self._image_difference(cropped_bogen, corner_img)
            if match_val < best_match_value:
                best_match_pos = pos
                best_match_value = match_val
        return best_match_pos

    def _find_corner_positions(self, sheet, radius=30):
        corner_positions = []

        for corner_no in range(4):
            sheet_image = Image.open(self._path_to_sheets + sheet).convert("RGBA")
            corner_positions.append(self._find_corner_position(sheet_image, corner_no, radius))

        return corner_positions

    def _build_C_matrix(self):
        p_d = self._reference_corner_positions
        return np.matrix([
        [p_d[0][0], p_d[0][1], 0, 0, 1, 0],
        [p_d[1][0], p_d[1][1], 0, 0, 1, 0],
        [p_d[2][0], p_d[2][1], 0, 0, 1, 0],
        [p_d[3][0], p_d[3][1], 0, 0, 1, 0],
        [0, 0, p_d[0][0], p_d[0][1], 0, 1],
        [0, 0, p_d[1][0], p_d[1][1], 0, 1],
        [0, 0, p_d[2][0], p_d[2][1], 0, 1],
        [0, 0, p_d[3][0], p_d[3][1], 0, 1],
    ], dtype="double")

    def _build_d_vector(self, p):
        return np.asarray([
        [p[0][0]],
        [p[1][0]],
        [p[2][0]],
        [p[3][0]],
        [p[0][1]],
        [p[1][1]],
        [p[2][1]],
        [p[3][1]],
    ], dtype="double")

    def _epsilon(self, x, C_matrix, D_vector):
        """ Calculates Epsilon as specified """
        return (((C_matrix*np.asmatrix(x).transpose()- D_vector).A1)**2).sum()

    def _phi(self, pos, x_Ab):
        """ Transforms position pos according to values in x_Ab """
        return [x_Ab[0]*pos[0]+x_Ab[1]*pos[1]+x_Ab[4], x_Ab[2]*pos[0]+x_Ab[3]*pos[1]+x_Ab[5]]


class TrainTestData:
    """
    this class is used to obtain, save and load test and training data
    """
    def __init__(self, path = "./Evaluationshelfer_Daten/crosses/"):
        self._path = path
        self.checked = ["work_type_empty/","work_type_crossed/"]
        self._saving_loc = "./Evaluationshelfer_Daten"
        self._training_data = []
        self._train_df = None

    def obtain_training_data (self):
        print("loading the training data... ")
        columns = [f"pixel {i+1}" for i in range(1600)]
        columns.extend(["filled", "not filled"])
        dict_list = []
        for filled, check in enumerate(self.checked):
            file_names = os.listdir(self._path+check)
            for file_name in file_names:
                box = list(Image.open(self._path + check + file_name).convert("L").getdata())
                box = [1-i/255 for i in box]
                self._training_data.append((np.array(box).reshape(1600,1), np.array([[filled],[1-filled]])))
                box.append(filled)
                tmp = 1-filled
                box.append(tmp)
                dict_list.append(dict(zip(columns, box)))
            self._train_df = pd.DataFrame.from_dict(dict_list)
        print("data imported")
        return self._training_data

    def save_train_data(self, path = None):
        if path != None:
            self._saving_loc = path
        # if os.path.exists(path) == False:
        #     os.mkdir(path)
        self._train_df.to_parquet(self._saving_loc+"/train_data.parquet")
        print("training data file successfully saved")

    def load_train_data(self, path):
        self._train_df = pd.read_parquet(path)
        for row in self._train_df.iterrows():
            self._training_data.append((np.array(row[-1][:-2]).reshape(1600,1), np.array(row[-1][-2:]).reshape(2,1)))
        print("training data file opened successfully")
        return self._training_data

if __name__ == "__main__":
    checkboxFinder = CheckboxFinder("./Evaluationshelfer_Daten/boegen/")
    checkboxFinder.calculate_checkbox_positions(["Bogen"+str(i)+".jpg" for i in range(1, 29)])
    checkboxFinder.get_checkbox_image("Bogen16.jpg", 4, 0).show()
