import pandas as pd 
import matplotlib.pyplot as plt 

df = pd.read_csv('./metrics.csv')

print(df.head())

df["dice_loss: _epoch"].dropna().plot()
plt.show()