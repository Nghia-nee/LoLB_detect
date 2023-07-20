import os
import pandas as pd

folder_path = "LOLBAS"  # Đường dẫn tương đối đến thư mục cần lọc
extensions = (".md", ".xml")  # Các phần mở rộng cần lọc
output_file = "class.xlsx"  # Đường dẫn tương đối đến file Excel đầu ra


# Hàm đệ quy để lọc các tên file trong thư mục con và cháu
def filter_files_recursively(folder):
    filtered_files = []
    for root, dirs, files in os.walk(folder):
        # Kiểm tra quyền truy cập vào các thư mục con
        try:
            os.listdir(root)
        except PermissionError:
            continue

        for file in files:
            filename, ext = os.path.splitext(file)
            if '.' in filename:
                filename = filename.split('.', 1)[0] # chỉ lấy tên file đến dấu chấm đầu tiên
            if ext in extensions:
                filtered_files.append(filename)  # Chỉ lấy tên file, không bao gồm phần mở rộng
    return filtered_files



# Gọi hàm đệ quy để lấy danh sách các tên file đã lọc
filtered_files = filter_files_recursively(folder_path)

# Tạo DataFrame từ danh sách các tên file đã lọc
df = pd.DataFrame({'File Path': filtered_files})

# Ghi DataFrame vào file Excel
df.to_excel(output_file, index=False)

print("Danh sách các tên file đã lọc đã được ghi vào file Excel thành công.")


