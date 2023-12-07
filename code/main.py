import os
import subprocess
def main():
   try:
      os.system(f'python {file_path}')
   except FileNotFoundError:
      print(f"Error: The file '{file_path}' does not exist.")

if __name__ == "__main__":
    main()