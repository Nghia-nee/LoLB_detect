import pandas as pd
import numpy as np
import fasttext
import re
from unidecode import unidecode
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, r2_score, f1_score, precision_score
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt

# Paths to files
fasttext_Model = 'model_fasttext.bin'
label_Data_File = 'labeled_data.xlsx'
class_onehot_File = 'class_onehot.xlsx'
ensemble_model_File = 'xgb_model.bin'

# Load fasttext model
model_wordVector = fasttext.load_model(fasttext_Model)
with open('xgb_model.bin', 'rb') as f:
    xgb_model = pickle.load(f)

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

            token_score = calculate_token_scores(token_vector, xgb_model)

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

X = np.array(avfvector_list)
y = np.array(label_list)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Tạo mô hình Naive Bayes
#nb_model = GaussianNB()

# Huấn luyện mô hình trên tập huấn luyện
#nb_model.fit(X_train, y_train)

#with open('naive_bayes_model.pkl', 'wb') as f:
#    pickle.dump(nb_model, f)
#################################################################################################
### Active learning
precision_list = []
fraction_TP_found_list = []
for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    # Tạo mô hình Naive Bayes
    nb_model = GaussianNB()
    
    # Huấn luyện mô hình
    nb_model.fit(X_train, y_train)
    
    # Dự đoán trên tập test
    y_pred = nb_model.predict(X_test)
    
    # Tính precision
    precision = precision_score(y_test, y_pred, pos_label='malicious')
    
    # Dự đoán xác suất dương
    y_pred_proba = nb_model.predict_proba(X_test)[:, 1]
    
    # Thay đổi cách tính fraction of TP found
    fraction_TP_found = np.mean((y_pred_proba>0.5).astype(int))
    
    # Lưu giá trị vào danh sách
    precision_list.append(precision)
    fraction_TP_found_list.append(fraction_TP_found)

# Tạo trục x từ 1 đến 10
x = range(1, 51)

# Vẽ biểu đồ precision
plt.plot(x, precision_list, label='Precision')

# Vẽ biểu đồ fraction of TP found
plt.plot(x, fraction_TP_found_list, label='Fraction of TP Found')

# Đặt tiêu đề và nhãn trục
plt.title('Precision and Fraction of TP Found')
plt.xlabel('Iteration')
plt.ylabel('Value')

# Hiển thị chú thích
plt.legend()

# Hiển thị biểu đồ
plt.show()
#################################################################################################
### Evaluate model
# Dự đoán nhãn trên tập kiểm tra
#y_pred = nb_model.predict(X_test)
# Tính độ chính xác
#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy of naive bayes :", accuracy)
# Calculate accuracy on training set
#accuracy_train = np.mean(nb_model.predict(X_train) == y_train)
#print("Training Accuracy:", accuracy_train)
# Calculate accuracy on testing set
#accuracy_test = np.mean(nb_model.predict(X_test) == y_test)
#print("Testing Accuracy:", accuracy_test)
# Calculate F1 score
#f1 = f1_score(y_test, y_pred, pos_label='malicious')
#print("F1 score:", f1)
# Calculate confusion matrix
#conf_matrix = confusion_matrix(y_test, y_pred, labels=['benign', 'malicious'])
#false_positives = conf_matrix[0, 1]
#true_negatives = conf_matrix[0, 0]
#fp_rate = false_positives / (false_positives + true_negatives)
#print("False Positives rate:", fp_rate)
# Create figures
#fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
# Plot training accuracy
#axes[0, 0].bar(['Training Accuracy'], [accuracy_train])
#axes[0, 0].set_xlabel('Accuracy')
#axes[0, 0].set_ylabel('Value')
#axes[0, 0].set_title('Training Accuracy')
#axes[0, 0].set_ylim([0, 1])
# Plot testing accuracy
#axes[0, 1].bar(['Testing Accuracy'], [accuracy_test])
#axes[0, 1].set_xlabel('Accuracy')
#axes[0, 1].set_ylabel('Value')
#axes[0, 1].set_title('Testing Accuracy')
#axes[0, 1].set_ylim([0, 1])
# Plot F1 score
#axes[1, 0].bar(['F1 Score'], [f1])
#axes[1, 0].set_xlabel('F1 Score')
#axes[1, 0].set_ylabel('Value')
#axes[1, 0].set_title('F1 Score')
#axes[1, 0].set_ylim([0, 1])
# Plot false positive rate
#axes[1, 1].bar(['False Positive Rate'], [fp_rate])
#axes[1, 1].set_xlabel('False Positive Rate')
#axes[1, 1].set_ylabel('Value')
#axes[1, 1].set_title('False Positive Rate')
#axes[1, 1].set_ylim([0, 1])
#plt.tight_layout()  # Adjust spacing between subplots
# Display the figures
#plt.show()
#############################################################################

