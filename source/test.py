import os


replacements = [('pipes_are_random', "Random_pipes"),
                ('PLAYER_FLAP_ACC_VARIANCE', "Jump_Force_k"),
                ('GRAVITY_VARIANCE', "Gravity_k")
                ]

def replace_in_file(root, replacements):
    for dir in os.listdir(root):
        dir_path = os.path.join(root, dir)
        if os.path.isdir(dir_path):
            para_path = os.path.join(dir_path, 'param_log.txt')
            if os.path.exists(para_path):
                with open(para_path, 'r') as f:
                    content = f.read()
                for old, new in replacements:
                    content = content.replace(old, new)
                with open(para_path, 'w') as f:
                    f.write(content)


for i in range(4):
    root = f'../../exps/exp_{i}_f'
    replace_in_file(root, replacements)
    print(root, "done")