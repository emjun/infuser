import pandas as pd
from scipy import stats	

"WRANGLING"

df = pd.read_excel("../data/iris_data.xlsx")
print(df['species'])

df.groupby("species")['petal_length'].describe()

setosa = df[(df['species'] == 'Iris-setosa')]
virginica = df[(df['species'] == 'Iris-virginica')]



'''
# just a test if apply works
def mult(x, k):
    return k*x


ndf = virginica['sepal_length'].apply(lambda x: mult(x,2))
print(ndf)
'''

"ANALYSIS"

sh1 = stats.shapiro(setosa['petal_length'])
print(sh1)

sh2 = stats.shapiro(virginica['petal_length'])
print(sh2)

t1, p1 = stats.ttest_ind(setosa['petal_length'], virginica['petal_length'], equal_var = False)


def welch_dof(x,y):
        dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
        print(f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")
        
welch_dof(setosa['petal_length'], virginica['petal_length'])

def welch_ttest(x, y): 
    ## Welch-Satterthwaite Degrees of Freedom ##
    dof = (x.var()/x.size + y.var()/y.size)**2 / ((x.var()/x.size)**2 / (x.size-1) + (y.var()/y.size)**2 / (y.size-1))
   
    t, p = stats.ttest_ind(x, y, equal_var = False)
    
    print("\n",
          f"Welch's t-test= {t:.4f}", "\n",
          f"p-value = {p:.4f}", "\n",
          f"Welch-Satterthwaite Degrees of Freedom= {dof:.4f}")

welch_ttest(setosa['petal_length'], virginica['petal_length'])