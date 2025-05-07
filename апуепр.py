1.import pandas as pd
df_excel = pd.read_excel('Книга1.xlsx')
print(df_excel.head())
print(df_excel.sample(5))
print(df_excel.tail(5))
a=int(input("напиши ка мне номер строки"))
df_excel = pd.read_excel('Книга (1).xlsx',index_col=a)
print(df_excel)

2.import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data.xlsx'
df = pd.read_excel(file_path)

print(df.head())
plt.figure(figsize=(5, 7))
plt.plot(df['x'], df['y'], marker='o', label='Зависимость y от x')
plt.title('График из Excel')
plt.xlabel('Ось X')
plt.ylabel('Ось Y')
plt.legend()
plt.grid(True)
plt.show()

3.import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource

X = np.linspace(-100, 52, 1488)
Y = np.linspace(-3, 3, 33)
X_grid , Y_grid = np.meshgrid(X, Y)

Z_grid = np.sin(np.sqrt(X_grid + Y_grid)) * 2 + np.cos(Y_grid) * 1.5

ls = LightSource(azdeg=270, altdeg=95)

shaded_Z = ls.shade(Z_grid, cmap=plt.cm.terrain, vert_exag=0.2, blend_mode='soft')

fig = plt.figure(figsize=(4, 9))
ax = fig.add_subplot(235, projection='3d')

ax.plot_surface(X_grid, Y_grid, Z_grid, facecolors=shaded_Z, rstride=1, cstride=1)

ax.set_xlabel("X координата")
ax.set_ylabel("Y координата")
ax.set_zlabel("Высота (Z)")

plt.show()
