import pandas as pd
import numpy as np
import fasttext
import re
from unidecode import unidecode
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import pickle


# Paths to files
fasttext_Model = 'model_fasttext.bin'
label_Data_File = 'labeled_data.xlsx'
class_File = 'class.xlsx'

# Load fasttext model
model_wordVector = fasttext.load_model(fasttext_Model)

# Load data from Excel file
label_df = pd.read_excel(label_Data_File)
class_df = pd.read_excel(class_File)
class_names = class_df['Class'].tolist()

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
#for class_name, commands in classified_data.items():
#    print("Class:", class_name)
#    for command in commands:
#        print("	    Command:", command)
#print("Unclassified Data:")
#for command in unclassified_data:
#    print("	 Command:", command)

# Tokenize the command lines and remove number
def tokenize_lines(commands_with_labels):
    tokenized_lines = []
    for command_with_label in commands_with_labels:
        command = command_with_label[0]  # Extract the command from the tuple
        tokens = []
        for token in re.findall(r'\w+|[^\w\s]', unidecode(command)):
            if token.isdigit():
                tokens.append("number")
            elif re.match(r'^\w+$', token):
                tokens.append(token)
        tokenized_lines.append(tokens)
    return tokenized_lines

      
# Biểu diễn one-hot cho từng tên class
class_names = list(set(class_names))  # Lấy danh sách tên các LOLBIN (class) duy nhất
num_classes = len(class_names)  # Số lượng class
class_onehot_encodings = {}

for i, class_name in enumerate(class_names):
    class_onehot_encoding = [0] * num_classes  # Khởi tạo vector one-hot với tất cả giá trị là 0
    class_onehot_encoding[i] = 1  # Đặt giá trị 1 tại vị trí tương ứng với class
    class_onehot_encodings[class_name] = class_onehot_encoding
    
#for class_name, encoding in class_onehot_encodings.items():
#    print("Class:", class_name)
#    print("Encoding Length:", len(encoding))
#    print("---")

# Get the word vector for each token
tokenized_lines = tokenize_lines(label_df[['Command', 'Label']].values)
word_vecs = []
for line in tokenized_lines:
    vecs = []
    for token in line:
        vecs.append(model_wordVector[token])
    word_vecs.append(np.array(vecs))    
    
data = []
for class_name, commands_with_labels in classified_data.items():
    # Get one-hot encoding for class
    class_vector = class_onehot_encodings[class_name]

    for command_with_label in commands_with_labels:
        command = command_with_label[0]  # Extract the command from the tuple
        command_tokens = tokenize_lines([command_with_label])[0]
        label = command_with_label[1]  # Corresponding label for the command

        for token in command_tokens:
            if token in model_wordVector:
                # Create token vector
                token_vector = class_vector.copy()  # Start with the class one-hot encoding

                token_embedding = model_wordVector[token]  # Get word vector from model
                token_vector = np.concatenate((token_vector, token_embedding), axis=0)

                data.append((class_name, command_tokens, token, token_vector, int(label == 'malicious')))

                
#for class_name, command_tokens, token, token_vector, label in data:
#    print("Class:", class_name)
#    print("Command", command_tokens)
#    print("token", token)
#    print("Token Vector:", token_vector)
#    print("Label:", label)
#    print("---")

# Convert data to numpy arrays
X = np.array([token_vector for _, _, _, token_vector, _ in data])
y = np.array([label for _, _, _, _, label in data])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create ensemble models
rf_model = RandomForestClassifier()
lgb_model = lgb.LGBMClassifier()
svc_model = SVC(probability=True)

# Train the ensemble models
rf_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
svc_model.fit(X_train, y_train)

# Create a voting classifier for ensemble learning
models = [('Random Forest', rf_model), ('LightGBM', lgb_model), ('SVC', svc_model)]
ensemble_model = VotingClassifier(models, voting='soft')

# Fit the ensemble model
ensemble_model.fit(X_train, y_train)

# Perform ensemble predictions on training set
ensemble_predictions_train = ensemble_model.predict(X_train)

# Perform ensemble predictions on testing set
ensemble_predictions_test = ensemble_model.predict(X_test)

# Function to calculate scores for each token
def calculate_token_scores(token_vectors, models):
    scores = np.zeros(len(token_vectors))
    for model in models:
        predictions = model.predict(token_vectors)
        scores += predictions
    scores /= len(models)  # Calculate average score
    return scores


#onehot_df = pd.DataFrame(list(class_onehot_encodings.items()), columns=['Class', 'Vector'])
#onehot_df.to_excel('class_onehot.xlsx', index=False)


#with open('ensemble_model.bin', 'wb') as f:
#    pickle.dump(ensemble_model, f)

# Calculate scores for each token in training set
token_scores_train = calculate_token_scores(X_train, [rf_model, lgb_model, svc_model])

# Calculate scores for each token in testing set
token_scores_test = calculate_token_scores(X_test, [rf_model, lgb_model, svc_model])

# Calculate accuracy on training set
accuracy_train = np.mean(ensemble_predictions_train == y_train)
print("Training ensemble Accuracy:", accuracy_train)

# Calculate accuracy on testing set
accuracy_test = np.mean(ensemble_predictions_test == y_test)
print("Testing Accuracy:", accuracy_test)

##################################################################################################################
