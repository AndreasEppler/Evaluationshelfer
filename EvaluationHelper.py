import Neural_Network as nn
import CheckboxFinder as cf
import numpy as np
import os

obtain_data = cf.TrainTestData()
# training_data = obtain_data.obtain_training_data()
# obtain_data.save_train_data()
training_data = obtain_data.load_train_data("./Evaluationshelfer_Daten/train_data.parquet")

neural_net = nn.NeuralNetwork()

# neural_net.add(nn.Layer(1600, "sigmoid"))
# neural_net.add(nn.Layer(400, "sigmoid", init="sn"))
# neural_net.add(nn.Layer(50, "sigmoid", init="sn"))
# neural_net.add(nn.Layer(2, "sigmoid", init="sn"))

# neural_net.train(training_data, epochs = 5000, batch_size = 50, eta = 0.1)
# neural_net.save_model()

# neural_net = nn.NeuralNetwork()
neural_net.load_model("./Saved Models/2021_09_29 02h 18, 1600-2, 1000 Epochen.ann", verbose = True)
# neural_net.train(training_data, epochs = 1000, batch_size = 50, eta = 0.1)
# neural_net.save_model()

source_input = input("please enter the location, the survey sheets you want to evaluate can be found at: ")
assert type(source_input) == str, "enter valid location"
source_input = os.path.abspath(source_input)+"\\"
# source_input = (r"D:\Master_CIW\SS21\Einführung in Python\Abschlussprojekt\Code\Evaluationshelfer\Evaluationshelfer_Daten\boegen\\")

file_names = os.listdir(source_input)
print(f"{len(file_names)} survey sheets found, starting evaluation...")

checkboxFinder = cf.CheckboxFinder(source_input)
checkboxFinder.calculate_checkbox_positions(file_names)


# Auswertung
test_data = []      # list of lists, first index = qustion, second index = sheet, value = good...bad
for question in range(1,15):
    test_data.append([])
    for file in file_names:
        tmp_rating = []
        for rating in range(0,5):  
            box = checkboxFinder.get_checkbox_image(file, question, rating).convert("L").getdata()
            tmp = np.array(box).reshape(1600,1)/255
            tmp = neural_net.predict(np.ones_like(tmp) - tmp)
            tmp_rating.append(tmp[0][0])  #neu
        crossed_box = tmp_rating.index(max(tmp_rating))     # find box with the highest activation value of the "crossed" Neuron
        test_data[question-1].append(crossed_box)

print(f"\nStatistik zur Auswertung der {len(file_names)} Bögen:")
print(f"0 = trifft voll zu; 4 = trifft überhaupt nicht zu")
for question in range(1,15):
    print(f"Question number: {question}")
    # print(f"{len(test_data[question-1])} checked boxes identified")
    print(f"mean: {np.mean(test_data[question-1]):.5} (std. deviation: +-{np.std(test_data[question-1]):.5})")
    print(f"min: {min(test_data[question-1])}    max: {max(test_data[question-1])}")
    print("------------------")
