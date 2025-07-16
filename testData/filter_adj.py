import os
import shutil

INPUT_DIR = 'gazebo_summer' # file name in eth wood_autmn or gazebo_summer
OUTPUT_DIR = 'filtered_matrices'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for folder in os.listdir(INPUT_DIR):
    if folder.startswith('scans_') and os.path.isdir(os.path.join(INPUT_DIR, folder)):
        adj_path = os.path.join(INPUT_DIR, folder, 'adj.txt')
        
        if not os.path.exists(adj_path):
            print(f"[!] {folder}: adj.txt not found.")
            continue
        
        with open(adj_path, 'r') as f:
            lines = f.readlines()
            row_count = len(lines)
            if row_count == 0:
                print(f"[x] {folder}: empty file.")
                continue
            col_count = len(lines[0].strip().split())

        print(f"{folder}: {row_count}x{col_count}", end=' ')
        if row_count > 1000 and col_count > 1000:
            shutil.copyfile(adj_path, os.path.join(OUTPUT_DIR, folder + '.txt'))
            print("=> copied.")
        else:
            print("=> skipped.")

print("\n==== DONE ====")