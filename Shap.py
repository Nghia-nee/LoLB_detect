import pandas as pd
import numpy as np
import fasttext
import re
import pickle
import shap
import matplotlib.pyplot as plt
from unidecode import unidecode
from shap import waterfall_plot
from shap import Explainer, Explanation
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


# Paths to files
fasttext_Model = 'model_fasttext.bin'
label_Data_File = 'labeled_data.xlsx'
class_onehot_File = 'class_onehot.xlsx'
ensemble_model_File = 'ensemble_model.bin'

# Load fasttext model
model_wordVector = fasttext.load_model(fasttext_Model)
with open('ensemble_model.bin', 'rb') as f:
    ensemble_model = pickle.load(f)

# Load data from Excel file
label_df = pd.read_excel(label_Data_File)
class_onehot_df = pd.read_excel(class_onehot_File)
class_names = class_onehot_df['Class'].tolist()

# Divide the command for class
classified_data = {}
unclassified_data = []
for index, row in label_df.iterrows():
    command = row['Command']
    label = row['Label']
    found_class = False
    for class_name in class_names:
        pattern = r"(?<!\w)" + re.escape(class_name) + r"(?!\w)"
        if re.search(pattern, command, flags=re.IGNORECASE):
            if class_name not in classified_data:
                classified_data[class_name] = []
            classified_data[class_name].append((command, label))  # Lưu cả command và label
            found_class = True
            break  
    if not found_class:
        unclassified_data.append((command, label))  # Lưu cả command và label
  
        
#print("Classified Data:")        
#for class_name, data in classified_data.items():
#    print("Class:", class_name)
#    for command, label in data:
#        print("Command:", command)
#        print("Label:", label)
#        print("---")
#print("Unclassified Data:")
#for command in unclassified_data:
#    print("	 Command:", command)


# Tokenize the command lines and remove number
def tokenize_lines(commands):
    tokens = []
    for token in re.findall(r'\w+|[^\w\s]', unidecode(command)):
        if token.isdigit():
            tokens.append("number")
        elif re.match(r'^\w+$', token):
            tokens.append(token)
    return tokens

      
# Biểu diễn one-hot cho từng tên class
# Convert one-hot encodings to a dictionary
class_onehot_encodings = {}
for index, row in class_onehot_df.iterrows():
    class_name = row['Class']
    class_vector = np.array(eval(row['Vector']))
    class_onehot_encodings[class_name] = class_vector

#for class_name, encoding in class_onehot_encodings.items():
#    print("Class:", class_name)
#    print("Encoding Length:", len(encoding))
#    print("---")


def calculate_token_scores(token_vectors, model):
    scores = model.predict_proba(token_vectors.reshape(1, -1))[:, 1] 
    return scores


avfvector_list = []
label_list = []

for class_name, commands in classified_data.items():
    for command, label in commands:
        # Tokenize command
        tokenized_command = tokenize_lines(command)
        class_vector = class_onehot_encodings[class_name]
        count = len(tokenized_command)
        rare_token = 0
        command_vector = []
        token_score_list_per_command = []
        for token in tokenized_command:
            if token not in model_wordVector:
                rare_token += 1
                
            token_embedding = model_wordVector[token]  # Get word vector from model
            token_vector = np.concatenate((class_vector, token_embedding), axis=0)

            token_score = calculate_token_scores(token_vector, ensemble_model)

            command_vector.append(token_embedding)
            token_score_list_per_command.append(token_score)

        command_vector = np.array(command_vector)
        min_pooled_vector = np.min(command_vector, axis=0)
        max_pooled_vector = np.max(command_vector, axis=0)
        avg_pooled_vector = np.average(command_vector, axis=0, weights=np.array(token_score_list_per_command).flatten())
        token_score_list_per_command = sorted(token_score_list_per_command, reverse=True)[:2]
        avfvector = np.concatenate((min_pooled_vector, max_pooled_vector, avg_pooled_vector, np.array(token_score_list_per_command).flatten(), np.array([count]), np.array([rare_token]), class_vector))
        avfvector_list.append(avfvector)
        label_list.append(label)


#for avfvector, label in zip(avfvector_list, label_list):
#    print("AVF Vector:", avfvector)
#    print("Label:", label)
#    print("---")

##################################################################################################################

# Tạo dữ liệu và chia thành tập train và test
X = np.array(avfvector_list)
y = np.array(label_list)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Wrapper để biến đổi mô hình thành hàm có thể gọi được
def model_wrapper(X):
    return nb_model.predict_proba(X)

explainer = shap.Explainer(model_wrapper, X_train, algorithm="permutation", max_evals=1000)
shap_values = explainer(X_train)

# Hiển thị feature importance plot
shap.plots.bar(shap.Explanation(shap_values[0].data))
shap.plots.bar(shap.Explanation(shap_values[1].data))
shap.plots.bar(shap.Explanation(shap_values[2].data))
shap.plots.bar(shap.Explanation(shap_values[10].data))

plt.show()
