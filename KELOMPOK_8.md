---
jupyter:
  colab:
    toc_visible: true
  kernelspec:
    display_name: Python 3
    name: python3
  language_info:
    name: python
  nbformat: 4
  nbformat_minor: 0
---

<div class="cell markdown" id="DFzWDd1ap-AN">

KELOMPOK 8

1.  Saktia Wardana (5311421004)
2.  Muhammad Abdul Kharim(5311421018)
3.  Riean Noer Hakikie (5311421033)

</div>

<div class="cell code" execution_count="26" id="4rDU6-IApwrB">

``` python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from random import randint
%matplotlib inline
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
def split(df,label):
    X_tr, X_te, Y_tr, Y_te = train_test_split(df, label, test_size=0.25, random_state=42)
    return X_tr, X_te, Y_tr, Y_te

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score

classifiers = ['LinearSVM', 'RadialSVM',
               'Logistic',  'RandomForest',
               'AdaBoost',  'DecisionTree',
               'KNeighbors','GradientBoosting']

models = [svm.SVC(kernel='linear'),
          svm.SVC(kernel='rbf'),
          LogisticRegression(max_iter = 1000),
          RandomForestClassifier(n_estimators=200, random_state=0),
          AdaBoostClassifier(random_state = 0),
          DecisionTreeClassifier(random_state=0),
          KNeighborsClassifier(),
          GradientBoostingClassifier(random_state=0)]


def acc_score(df,label):
    Score = pd.DataFrame({"Classifier":classifiers})
    j = 0
    acc = []
    X_train,X_test,Y_train,Y_test = split(df,label)
    for i in models:
        model = i
        model.fit(X_train,Y_train)
        predictions = model.predict(X_test)
        acc.append(accuracy_score(Y_test,predictions))
        j = j+1
    Score["Accuracy"] = acc
    Score.sort_values(by="Accuracy", ascending=False,inplace = True)
    Score.reset_index(drop=True, inplace=True)
    return Score

def plot(score,x,y,c = "b"):
    gen = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    plt.figure(figsize=(6,4))
    ax = sns.pointplot(x=gen, y=score,color = c )
    ax.set(xlabel="Generation", ylabel="Accuracy")
    ax.set(ylim=(x,y))
```

</div>

<div class="cell code" execution_count="27" id="WykCouhgrNRe">

``` python
def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population):
    scores = []
    for chromosome in population:
        logmodel.fit(X_train.iloc[:,chromosome],Y_train)
        predictions = logmodel.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(Y_test,predictions))
    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])


def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel):
    pop_nextgen = pop_after_sel
    for i in range(0,len(pop_after_sel),2):
        new_par = []
        child_1 , child_2 = pop_nextgen[i] , pop_nextgen[i+1]
        new_par = np.concatenate((child_1[:len(child_1)//2],child_2[len(child_1)//2:]))
        pop_nextgen.append(new_par)
    return pop_nextgen

def mutation(pop_after_cross,mutation_rate,n_feat):
    mutation_range = int(mutation_rate*n_feat)
    pop_next_gen = []
    for n in range(0,len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0,mutation_range):
            pos = randint(0,n_feat-1)
            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop_next_gen.append(chromo)
    return pop_next_gen
def generations(df,label,size,n_feat,n_parents,mutation_rate,n_gen,X_train,
                                   X_test, Y_train, Y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen)
        print('Best score in generation',i+1,':',scores[:1])  #2
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate,n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score
```

</div>

<div class="cell code" execution_count="28"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="QZ1Nrg7VrofV" outputId="09b2e374-8535-4f81-ce78-89cbb9c2dec2">

``` python
data_pd = pd.read_csv("/content/Parkinsson disease.csv")
label_pd = data_pd["status"]
data_pd.drop(["status","name"],axis = 1,inplace = True)

print("Parkinson's disease dataset:\n",data_pd.shape[0],"Records\n",data_pd.shape[1],"Features")
```

<div class="output stream stdout">

    Parkinson's disease dataset:
     195 Records
     22 Features

</div>

</div>

<div class="cell code" execution_count="29"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:274}"
id="4CjiNf-pwI-v" outputId="724dd0e5-33d3-4353-a1ef-aad205db72ca">

``` python
display(data_pd.head())
print("All the features in this dataset have continuous values")
```

<div class="output display_data">

       MDVP:Fo(Hz)  MDVP:Fhi(Hz)  MDVP:Flo(Hz)  MDVP:Jitter(%)  MDVP:Jitter(Abs)  \
    0      119.992       157.302        74.997         0.00784           0.00007   
    1      122.400       148.650       113.819         0.00968           0.00008   
    2      116.682       131.111       111.555         0.01050           0.00009   
    3      116.676       137.871       111.366         0.00997           0.00009   
    4      116.014       141.781       110.655         0.01284           0.00011   

       MDVP:RAP  MDVP:PPQ  Jitter:DDP  MDVP:Shimmer  MDVP:Shimmer(dB)  ...  \
    0   0.00370   0.00554     0.01109       0.04374             0.426  ...   
    1   0.00465   0.00696     0.01394       0.06134             0.626  ...   
    2   0.00544   0.00781     0.01633       0.05233             0.482  ...   
    3   0.00502   0.00698     0.01505       0.05492             0.517  ...   
    4   0.00655   0.00908     0.01966       0.06425             0.584  ...   

       MDVP:APQ  Shimmer:DDA      NHR     HNR      RPDE       DFA   spread1  \
    0   0.02971      0.06545  0.02211  21.033  0.414783  0.815285 -4.813031   
    1   0.04368      0.09403  0.01929  19.085  0.458359  0.819521 -4.075192   
    2   0.03590      0.08270  0.01309  20.651  0.429895  0.825288 -4.443179   
    3   0.03772      0.08771  0.01353  20.644  0.434969  0.819235 -4.117501   
    4   0.04465      0.10470  0.01767  19.649  0.417356  0.823484 -3.747787   

        spread2        D2       PPE  
    0  0.266482  2.301442  0.284654  
    1  0.335590  2.486855  0.368674  
    2  0.311173  2.342259  0.332634  
    3  0.334147  2.405554  0.368975  
    4  0.234513  2.332180  0.410335  

    [5 rows x 22 columns]

</div>

<div class="output stream stdout">

    All the features in this dataset have continuous values

</div>

</div>

<div class="cell code" execution_count="30"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:300}"
id="ccmkjBsZwMb1" outputId="3a3a3528-e744-4f13-8208-49faf56f15b8">

``` python
score3 = acc_score(data_pd,label_pd)
score3
```

<div class="output execute_result" execution_count="30">

             Classifier  Accuracy
    0      RandomForest  0.918367
    1          Logistic  0.897959
    2  GradientBoosting  0.897959
    3         LinearSVM  0.877551
    4          AdaBoost  0.877551
    5      DecisionTree  0.877551
    6         RadialSVM  0.836735
    7        KNeighbors  0.836735

</div>

</div>

<div class="cell code" execution_count="33"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="wAVgKxeBwP0w" outputId="a7be4cbc-04c4-429f-d389-3a95422f1288">

``` python
logmodel = DecisionTreeClassifier(random_state=0)
X_train,X_test, Y_train, Y_test = split(data_pd,label_pd)
chromo_df_pd,score_pd=generations(data_pd,label_pd,size=80,n_feat=data_pd.shape[1],n_parents=64,mutation_rate=0.20,n_gen=15,
                         X_train = X_train,X_test = X_test,Y_train = Y_train,Y_test = Y_test)
```

<div class="output stream stdout">

    Best score in generation 1 : [0.9795918367346939]
    Best score in generation 2 : [0.9795918367346939]
    Best score in generation 3 : [0.9591836734693877]
    Best score in generation 4 : [0.9591836734693877]
    Best score in generation 5 : [0.9591836734693877]
    Best score in generation 6 : [0.9795918367346939]
    Best score in generation 7 : [0.9591836734693877]
    Best score in generation 8 : [0.9795918367346939]
    Best score in generation 9 : [0.9795918367346939]
    Best score in generation 10 : [0.9795918367346939]
    Best score in generation 11 : [0.9795918367346939]
    Best score in generation 12 : [0.9387755102040817]
    Best score in generation 13 : [0.9795918367346939]
    Best score in generation 14 : [0.9795918367346939]
    Best score in generation 15 : [0.9795918367346939]

</div>

</div>

<div class="cell code" execution_count="34"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:393}"
id="-y17mmMBwUVq" outputId="e5776d16-5aa4-48bc-fb29-66fb41b17938">

``` python
plot(score_pd,0.9,1.0,c = "orange")
```

<div class="output display_data">

![](d1d65344063ecac3871fb10605f0ffdf488722ff.png)

</div>

</div>
