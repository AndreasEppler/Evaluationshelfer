# Survey Evaluation Project
This project serves as a survey sheet evaluation helper. It reads and evaluates scanned survey sheets in three steps. First, it detects relevant edges in order to calculate a transformation matrix relative to a reference survey sheet. It then extracts the ticking boxes of each question for each of the survey sheets. In a third step, a neural network that was coded from scratch is used in order to evaluate whether the boxes are ticked or not.

# Files
Neural_Network.py: Implements a simple neural network from scratch.
Checkbox_Finder.py: Detects and extracts checkboxes from a survey sheet.
EvaluationHelper.py: Uses the other Neural_Network and Checkbox_Finder modules to evaluate multiple survey forms. It returns the mean, standard deviation, minimum, and maximum of the answers of each question.

# Contributing
This project was created in the course of a python class project and was coded in collaboration with two other course attendees. 