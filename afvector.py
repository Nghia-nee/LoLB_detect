import pickle
import pandas as pd
import numpy as np
import fasttext
import re
from unidecode import unidecode
from sklearn.ensemble import VotingClassifier


# Paths to files
fasttext_Model = 'model_fasttext.bin'
data_need_check_File = 'data_need_check.xlsx'
ensemble_model_File = 'ensemble_model.bin'
class_onehot_File = 'class_onehot.xlsx'

# Load model
model_wordVector = fasttext.load_model(fasttext_Model)
with open('ensemble_model.bin', 'rb') as f:
    ensemble_model = pickle.load(f)
with open('naive_bayes_model.pkl', 'rb') as f:
    predict_model = pickle.load(f)

# Load data from Excel file
command_df = pd.read_excel(data_need_check_File)
class_onehot_df = pd.read_excel(class_onehot_File)

# Get class names from class_onehot_df
class_names = class_onehot_df['Class'].tolist()

# Divide the command for class
classified_data = {}
unclassified_data = []
for index, row in command_df.iterrows():
    command = row['Command']
    found_class = False
    for class_name in class_names:
        pattern = r"(?<!\w)" + re.escape(class_name) + r"(?!\w)"
        if re.search(pattern, command, flags=re.IGNORECASE):
            if class_name not in classified_data:
                classified_data[class_name] = []
            classified_data[class_name].append(command)
            found_class = True
            break  
    if not found_class:
        unclassified_data.append(command)
        
#print("Classified Data:")
#for class_name, commands in classified_data.items():
#    print("Class:", class_name)
#    for command in commands:
#        print("	    Command:", command)
#print("Unclassified Data:")
#for command in unclassified_data:
#    print("	 Command:", command)
        

# Tokenize the command lines and remove numbers
def tokenize_lines(commands):
    tokenized_lines = []
    for command in commands:
        tokens = []
        for token in re.findall(r'\w+|[^\w\s]', unidecode(command)):
            if token.isdigit():
                tokens.append("number")
            elif re.match(r'^\w+$', token):
                tokens.append(token)
        tokenized_lines.append(tokens)
    return tokenized_lines

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

for class_name, commands in classified_data.items():
    # Tokenize commands
    tokenized_commands = tokenize_lines(commands)

    for command_tokens in tokenized_commands:
        class_vector = class_onehot_encodings[class_name]   
        count = len(command_tokens)
        rare_token = 0
        command_vector = []
        token_score_list_per_command = []

        for token in command_tokens:
            if token not in model_wordVector:
                rare_token += 1
            token_embedding = model_wordVector[token]  # Get word vector from model
            token_vector = np.concatenate((class_vector, token_embedding), axis=0)

            token_score = calculate_token_scores(token_vector, ensemble_model)

            command_vector.append(token_embedding)
            token_score_list_per_command.append(token_score)

        if command_vector:
            command_vector = np.array(command_vector)
            min_pooled_vector = np.min(command_vector, axis=0)
            max_pooled_vector = np.max(command_vector, axis=0)
            avg_pooled_vector = np.average(command_vector, axis=0, weights=np.array(token_score_list_per_command).flatten())
            token_score_list_per_command = sorted(token_score_list_per_command, reverse=True)[:2]

            avfvector = np.concatenate((min_pooled_vector, max_pooled_vector, avg_pooled_vector, np.array(token_score_list_per_command).flatten(), np.array([count]), np.array([rare_token]), class_vector))
            avfvector_list.append(avfvector)



for class_name, commands in classified_data.items():
    avf_index = 0
    print("Class:", class_name)
    print("Number of commands:", len(commands))
    for command in commands:
        avf = avfvector_list[avf_index]
        prediction = predict_model.predict([avf])
        print("Command: ", command)
        print("Dự đoán: ", prediction[0])
        avf_index += 1

#for class_name, commands in classified_data.items():
#    avf_index = 0
#    for command in commands:
#        avf = avfvector_list[avf_index]
#        prediction = predict_model.predict([avf])
#        print("Command: ", command)
#        print("Dự đoán: ", prediction[0])
#        print("---")
#        avf_index += 1

for avf in avfvector_list:
    prediction = predict_model.predict([avf])
    print("Dự đoán:", prediction[0])

print("các command sau không thể đánh giá vì không xếp loại vào class được: ")
print(unclassified_data)
    

#unclassified_df = pd.DataFrame({"Command": unclassified_data})
#unclassified_df.to_excel("unclassified.xlsx", index=False)
