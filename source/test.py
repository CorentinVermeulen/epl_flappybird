
import os

root = '../../exps/exp_1_f/'

# I want to add in all param_log.txt file some lines

added = "PLAYER_FLAP_ACC_VARIANCE: 0\nGRAVITY_VARIANCE: 0"

for dir in os.listdir(root):
    dir_path = os.path.join(root, dir)
    if os.path.isdir(dir_path):
        para_path = os.path.join(dir_path, 'param_log.txt')
        if os.path.exists(para_path):
            with open(para_path, 'r') as f:
                s = f.read()
                if added not in s:
                    with open(para_path, 'a') as f:
                        f.write(added)
                        print(f"Added to {para_path}")
