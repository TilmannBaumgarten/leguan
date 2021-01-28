import tables
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

ID = 0

eval_file = tables.open_file(f"D:\\leguan_data\\float_training_data\\normalized_train_evals_{ID}.hdf5", mode='r')

array = eval_file.root.data



df = pd.DataFrame()
df['evaluation']= pd.Series(array)

plot = sns.displot(df, x="evaluation", binwidth=0.01)

plt.savefig('plot.png')