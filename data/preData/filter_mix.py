import os
import pandas as pd
import yaml

# Đường dẫn tới thư mục chứa các file .md và .yml
dir_path = r'D:\Desktop\LOLBAS'

# Set để theo dõi các dòng đã lưu trước đó
saved_lines = set()

# Duyệt qua các tệp tin .md và .yml trong thư mục và các thư mục con
with open('D:\\Desktop\\filtered.txt', 'w', encoding='utf-8') as f:
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            # Nếu là file .md
            if file_name.endswith('.md'):
                with open(file_path, 'r', encoding='utf-8') as f1:
                    lines = f1.readlines()

                    # Tìm chỉ số của dòng chứa "Full_Path:"
                    full_path_index = None
                    for i, line in enumerate(lines):
                        if line.strip() == "Full_Path:":
                            full_path_index = i
                            break

                    # Tìm chỉ số của dòng chứa "Detection:"
                    detection_index = None
                    for i, line in enumerate(lines):
                        if line.strip() == "Detection:":
                            detection_index = i
                            break

                    # Lấy nội dung từ "Full_Path:" tới "Detection:"
                    if full_path_index is not None and detection_index is not None:
                        full_path_content = file_name
                        detection_content = ''
                        for line in lines[full_path_index + 1:]:
                            line = line.strip()
                            if line == '':
                                continue
                            if line.startswith('Full_Path:'):
                                break
                            if line.startswith('Detection:'):
                                break
                            detection_content += line + ' '

                        # Kiểm tra xem dòng hiện tại có bị trùng với các dòng đã lưu hay không
                        if detection_content not in saved_lines:
                            saved_lines.add(detection_content)
                            f.write(f"{full_path_content}: {detection_content}\n")

            # Nếu là file .yml
            elif file_name.endswith('.yml'):
                with open(file_path, 'r') as file:
                    data = yaml.safe_load(file)

                # Trích xuất các nội dung theo yêu cầu và in ra
                if 'Commands' in data:
                    for command in data['Commands']:
                        if 'Command' in command and 'Description' in command:
                            command_content = command['Command']
                            if command_content not in saved_lines:
                                saved_lines.add(command_content)
                                f.write(command_content + '\n')
                else:
                    f.write(f"")
