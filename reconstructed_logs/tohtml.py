import os
import glob
import subprocess

def convert_notebooks_to_html():
    notebook_files = glob.glob("*.ipynb")
    
    if not notebook_files:
        print("Không tìm thấy file .ipynb nào trong thư mục này.")
        return

    for file_name in notebook_files:
        try:
            cmd = f'jupyter nbconvert --to html "{file_name}"'
            subprocess.run(cmd, shell=True, check=True)
        
        except subprocess.CalledProcessError as e:
            print(f"Lỗi khi chuyển đổi file {file_name}: {e}")

if __name__ == "__main__":
    convert_notebooks_to_html()
