input_file = 'rawCommand.txt'
output_file = 'rawCommand_Tokenized.txt'

with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        tokens = line.split()
        processed_tokens = []
        for token in tokens:
            if token.isdigit():
                processed_tokens.append("number")
            else:
                processed_tokens.append(token)
        processed_line = " ".join(processed_tokens)
        f_out.write(processed_line + '\n')
