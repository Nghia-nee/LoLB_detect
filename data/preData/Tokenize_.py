import re
from unidecode import unidecode

with open('filtered.txt', 'r') as f:
    lines = f.readlines()

tokenized_lines = []
for line in lines:
    line = unidecode(line)  # xóa hết dấu
    tokens = [word for word in re.findall(r'\w+|[^\w\s]', line) if re.match(r'^\w+$', word)]  # chứa cả chữ cái và chữ số
    tokenized_lines.append(tokens)

with open('tokenized.txt', 'w') as f:
    for tokens in tokenized_lines:
        f.write(' '.join(tokens) + '\n')
