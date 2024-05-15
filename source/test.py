import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 12,
    "font.size": 12,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10
}

plt.rcParams.update(tex_fonts)
sns.set_context("paper")
#sns.set_style("whitegrid")
large_size = (11, 5.5)
small_size = (7, 3.5)
square_size = (5.5, 5.5)

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


# for i in range(4):
#     root = f'../../exps/exp_{i}_f'
#     replace_in_file(root, replacements)
#     print(root, "done")



import pandas as pd



def decay(estart, eend, edecay, x):
    return eend + (estart - eend) * np.exp(-1. * x / edecay)


x = np.linspace(0, 100000, 1000)
df = pd.DataFrame({'x': x,
                    '1000': decay(0.9, 0.001, 1000, x),
                   '2500': decay(0.9, 0.001, 2500, x),
                   '5000': decay(0.9, 0.001, 5000, x),
                   '10000': decay(0.9, 0.001, 10000, x),
                   })

stacked_df = df.melt(id_vars='x', var_name='decay')

plt.figure(figsize=small_size)
sns.lineplot(data=stacked_df, x='x', y='value', hue='decay')
plt.xlabel('Steps')
plt.ylabel('Epsilon')
#plt.show()
plt.savefig('epsilon_decay.pdf',format="pdf" ,dpi=300, bbox_inches='tight')
