import os
import sys
import io

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#Deep Learning - 
coursera_dl_course_2_path = os.path.abspath('D:/Git Projects/bot-winter-project/study material/cousera deeplearning/Course 2 - Improving NN/scripts')
if coursera_dl_course_2_path not in sys.path:
    sys.path.append(coursera_dl_course_2_path)
import coursera_dp_master

#### Data Science Utils ###

#Describe data Utils
def print_new_section(title, body):
    o_separator = '#'*80
    i_separator = '_'*80
    print(o_separator)
    print(title)
    print(i_separator)
    print(body)
    print(i_separator)

def describe_columns( df ):
    var = [] ; l = [] ; t = [] ; c= []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
        c.append(df[x].count())

    levels = pd.DataFrame({'Variable' : var , 'Levels' : l , 'Count' : c , 'Datatype' : t})
    levels.sort_values(by = 'Levels' , inplace = True)
    return levels

def get_correlation(df, feature_name, output_name, sort_by_feature=False, sort_ascending=False):
    sorting_feature = feature_name if sort_by_feature else output_name
    return df[[feature_name, output_name]].groupby([feature_name], as_index=False).mean().sort_values(by=sorting_feature, ascending=sort_ascending)

def describe_dataset(df, output_name, show_all_correlations=False):
    #columns
    print_new_section("Columns", df.columns.values)
    #columns datatypes and where the data is missing
    print_new_section("DType Info", describe_columns(df))
    #statistics about the numerical features
    print_new_section("Numerical features Stats", df.describe())
    #statistics about the categorical features
    print_new_section("Categorical features Stats", df.describe(include=['O']))
    #check correlation for all non-unique features
    if (show_all_correlations):
        for x in df:
            if (x == output_name):
                continue

            count = df[x].count()
            unique_count = len(pd.value_counts(df[x]))
            if (count == unique_count):
                continue

            print_new_section("Correlation: " + x, get_correlation(df, x, output_name))
    #

    


#Plot Utils
def initialize_plot_config():
    mpl.style.use( 'ggplot' )
    sns.set_style( 'white' )
    pylab.rcParams[ 'figure.figsize' ] = 8 , 6

def plot_distribution(df, var, target, row=None, col=None):
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row=row , col=col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()
    return facet

def plot_categories( df , cat , target, row=None, col=None):
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()
    return facet

def plot_histograms( df , variables , n_rows , n_cols ):
    fig = plt.figure( figsize = ( 16 , 12 ) )
    for i, var_name in enumerate( variables ):
        ax=fig.add_subplot( n_rows , n_cols , i+1 )
        df[ var_name ].hist( bins=10 , ax=ax )
        ax.set_title( 'Skew: ' + str( round( float( df[ var_name ].skew() ) , ) ) ) # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels( [] , visible=False )
        ax.set_yticklabels( [] , visible=False )
    fig.tight_layout()  # Improves appearance a bit.
    return fig

def plot_correlation_map(df):
    corr = df.corr()
    _ , ax = plt.subplots(figsize=( 12 , 10 ))
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    hm = sns.heatmap(
        corr, 
        cmap=cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot=True, 
        annot_kws={ 'fontsize' : 12 }
    )
    return hm

def plot_variable_importance( X , y ):
    tree = DecisionTreeClassifier( random_state = 99 )
    tree.fit( X , y )
    plot_model_var_imp( tree , X , y )

def plot_model_var_imp( model , X , y ):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print (model.score( X , y ))

#Feature Engineering
    # for tensor flow, see this: https://www.tensorflow.org/tutorials/linear#feature_columns_and_transformations
def fill_na(df_o, serie_name, type="mean", custom_val=None):
    df = df_o.copy()
    dropna_serie = df[serie_name].dropna()
    
    if (type == "mean"):
        df[serie_name] = df[serie_name].fillna(dropna_serie.mean())
    elif (type == "median"):
        df[serie_name] = df[serie_name].fillna(dropna_serie.median())
    elif (type == "mode"):
        df[serie_name] = df[serie_name].fillna(dropna_serie.mode()[0])
    elif (type == "zero"):
        df[serie_name] = df[serie_name].fillna(0)
    elif (type == "custom"):
        df[serie_name] = df[serie_name].fillna(custom_val)

    """
    #guess
    #iterate over sex(0 or 1) and Pclass(1,2,3) to get the avg of the age of those 6 combinations
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)
    """

    return df
    
def convert_to_numeric(df, serie_name, one_hot=True):
    if (one_hot):
        numeric_serie = pd.get_dummies(df[serie_name], prefix=serie_name)
        df.drop([serie_name], axis=1, inplace=True)
        df = df.join(numeric_serie, how='left') #pd.concat([df, numeric_serie], axis=1)
    else:
        unique_values = pd.value_counts(df[serie_name])
        val = 0
        for x in unique_values.index:
            unique_values[x] = val
            val += 1

        numeric_serie = df[serie_name].map(unique_values).astype(int)
        df[serie_name] = numeric_serie

    return df

def convert_to_range_band(df_o, serie_name, bands=None, num_bands=None):
    df = df_o.copy()
    if bands is None:
        if num_bands is None:
            num_bands = 10 #default to 10 bands

        bands = []
        min_range = df[serie_name].min()
        max_range = df[serie_name].max()
        range_val = (max_range - min_range) / num_bands

        for i in range(1, num_bands):
            bands.append(min_range + (range_val * i))
    
    num_bands = len(bands)
    for i in range(num_bands + 1):
        if (i == 0):
            df.loc[df[serie_name] <= bands[i], serie_name] = i
        elif (i == (num_bands)):
            df.loc[df[serie_name] > bands[i-1], serie_name] = i
        else:
            df.loc[(df[serie_name] > bands[i-1]) & (df[serie_name] <= bands[i]), serie_name] = i

    df[serie_name] = df[serie_name].astype(int)
    return df

def test_data_science_utils():
    ## Aquire Data ##
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')

    initialize_plot_config()

    describe_dataset(train_df, "Survived", show_all_correlations=False)
    #plot_correlation_map(train_df) ; plt.show()
    #plot_distribution(train_df, var="Age", target="Survived", row="Sex") ; plt.show()
    #plot_distribution(train_df, var="Fare", target="Survived") ; plt.show()
    #plot_categories(train_df, cat ="Embarked", target="Survived") ; plt.show()
    #plot_categories(train_df, cat ="Sex", target="Survived") ; plt.show()
    #plot_categories(train_df, cat ="Pclass", target="Survived") ; plt.show()
    #plot_categories(train_df, cat ="SibSp", target="Survived") ; plt.show()
    #plot_categories(train_df, cat ="Parch", target="Survived") ; plt.show()

    df = train_df.append(test_df, ignore_index=True) #crate one single df so we can process the data all together

    df = convert_to_numeric(df, serie_name="Sex", one_hot=False)
    df = convert_to_numeric(df, serie_name="Pclass", one_hot=False) #not as a one hot because class 1 is better than 2 (ordinal!)

    #df = fill_na(df, serie_name="Embarked", type="mode")
    #df = convert_to_numeric(df, serie_name="Embarked", one_hot=True)
    df.drop(["Embarked"], axis=1, inplace=True) #pointless

    df = fill_na(df, serie_name="Age", type="mean")
    df = convert_to_range_band(df, serie_name="Age", num_bands=7)

    df = fill_na(df, serie_name="Fare", type="mean")
    df = convert_to_range_band(df, serie_name="Fare", num_bands=5)

    df["Title"] = df["Name"].str.extract(' ([A-Za-z]+)\.', expand=False) #regex matches first word which ends with a dot
    df.drop(["Name"], axis=1, inplace=True)
    df["Title"] = df["Title"].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df["Title"] = df["Title"].replace('Mlle', 'Miss')
    df["Title"] = df["Title"].replace('Ms', 'Miss')
    df["Title"] = df["Title"].replace('Mme', 'Mrs')
    df = convert_to_numeric(df, serie_name="Title", one_hot=True)

    df = fill_na(df, serie_name="Cabin", type="custom", custom_val="Unkown")
    df["Cabin"] = df["Cabin"].map(lambda c : c[0])   #mapping each Cabin value with the cabin letter
    df = convert_to_numeric(df, serie_name="Cabin", one_hot=True)

    df["FamilySize"] = df['SibSp'] + df['Parch'] + 1
    df = convert_to_range_band(df, serie_name="FamilySize", bands=[1, 2, 4])
    df = convert_to_numeric(df, serie_name="FamilySize", one_hot=True)
    df.drop(["SibSp", "Parch"], axis=1, inplace=True)

    df = df.drop(['Ticket', "PassengerId"], axis=1)

    train_df_mod = df[:train_df.shape[0]]
    test_df_mod = df[-test_df.shape[0]:]
    
    #plot_correlation_map(train_df) ; plt.show()
    
    #divide X and Y from data set
    X_train = train_df_mod.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df_mod.drop("Survived", axis=1)
    print(X_train.shape, Y_train.shape, X_test.shape)
    plot_variable_importance(X_train, Y_train) ; plt.show()

    #MODEL - coursera 2 course
        #DeepNN
    layers = [{"n":50},   #   "activation":"relu",        "dropout_keep_prob":1.0},
              {"n":25},   #   "activation":"relu",        "dropout_keep_prob":1.0},
              {"n":10},   #   "activation":"relu",        "dropout_keep_prob":1.0},    
              {"n":1}]    #  "activation":"sigmoid",     "dropout_keep_prob":1.0}]

    #deep_nn_adam = coursera_dp_master.DeepNN(layers = layers)
    deep_nn_tf = coursera_dp_master.TFDeepNN(layers = layers)
    #train
    train_X, train_Y, test_X = X_train.values.T, Y_train.values.reshape(1, Y_train.shape[0]), X_test.values.T
    
    #train_X = coursera_dp_master.feature_standardization(train_X)
    #test_X = coursera_dp_master.feature_standardization(test_X)

    #costs = deep_nn_adam.fit(train_X, train_Y, num_epocs=2000, mini_batch_size=64, learning_rate=0.00001, adam_hyperparams=coursera_dp_master.AdamHyperParams())
    costs = deep_nn_tf.fit(train_X, train_Y, num_epochs=2000, mini_batch_size=64, learning_rate=0.001)
    plt.clf()
    plt.plot(costs)
    plt.show()

    #predict
    #train_Y_pred = deep_nn_adam.predict(train_X)
    #test_Y = deep_nn_adam.predict(test_X)
    train_Y_pred = deep_nn_tf.predict(train_X)
    test_Y = deep_nn_tf.predict(test_X)

     # Print train Errors= 
    train_accuracy = coursera_dp_master.calc_accuracy(train_Y_pred, train_Y)
    print("DNN train accuracy: {} %".format(train_accuracy))
    
    # RandomTreeClassifier
    #random_forest = RandomForestClassifier(n_estimators=100)
    #random_forest.fit(X_train, Y_train)
    #test_Y = random_forest.predict(X_test)
    #random_forest.score(X_train, Y_train)
    #acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
    #print("RandomForestClassifoer train accuracy: {} %".format(acc_random_forest))
    
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_Y.flatten()
    })

    submission.to_csv('output/submission4.csv', index=False)


if __name__ == "__main__":
    #change dir to this script location
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    test_data_science_utils()

    """
    ## Aquire Data ##
    train_df = pd.read_csv('input/train.csv')
    test_df = pd.read_csv('input/test.csv')
    combine = [train_df, test_df]


        # print data to see how it looks
    print(train_df.columns.values)
    print('_'*40)

    print(train_df.head())
    print('_'*40)

    train_df.info() #get the datatypes and where the data is missing
    print('_'*40)
    test_df.info()
    print('_'*40)

    print(train_df.describe()) #get some statistics about the numerical featres
    print('_'*40)

    print(train_df.describe(include=['O'])) #get some statistics about the categorical featres
    print('_'*40)

        #check correlation of indivual features with output
    print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print('_'*40)
    print(train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
    print('_'*40)

        #use histogram to analyze continuos variables (like age)
    grid = sns.FacetGrid(train_df, col='Survived')
    grid.map(plt.hist, 'Age', bins=20)
    #plt.show()

        #combine multiple features for identifying correlations using a single plot
    grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend();
    #plt.show()

    grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
    grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
    grid.add_legend()
    #plt.show()

    grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()
    #plt.show()

    #drop bad features (ticket number and cabin number)
    train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
    test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
    combine = [train_df, test_df]

       #create new feature based on existent (feature engineering)
        #extrace title from name before dropping the name
    for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #regex matches first word which ends with a dot
    
    print(pd.crosstab(train_df['Title'], train_df['Sex']))

        #simply titles (most commun types and put all the rares one together)
    for dataset in combine:
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

        #convert from categories to ordinal
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
    for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

    print(train_df.head())

        #drop name and passanger id (title is better)
    train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
    test_df = test_df.drop(['Name'], axis=1)
    combine = [train_df, test_df]

    #convert categorical features -Sex
    for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    
    #completing numerical continues feature - age
        # use random guesses or base the guessing on correlation of other features (Pclass and gender combinations)
        #visualze the correlation
    grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.add_legend()

    #prepare array of guesses
    guess_ages = np.zeros((2,3))
    guess_ages

    #iterate over sex(0 or 1) and Pclass(1,2,3) to get the avg of the age of those 6 combinations
    for dataset in combine:
        for i in range(0, 2):
            for j in range(0, 3):
                guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
                age_guess = guess_df.median()

                # Convert random age float to nearest .5 age
                guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
                
        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)

    print(train_df.head())

    #create age bands
    train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
    print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)) #check correlation

    #replace age with age band value (why not use the age bands values???????)
    for dataset in combine:    
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']
    
    train_df = train_df.drop(['AgeBand'], axis=1) #drop the age band
    combine = [train_df, test_df]
    print(train_df.head())

    #create "family size" and "is alone" feature
    for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)) #check correlation
    
    for dataset in combine:
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

    print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()) #check correlation

        #drop parents and sibliings (replaced with family size)
    train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
    combine = [train_df, test_df]

    #create "artificial" feature combining class and age
    for dataset in combine:
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
    
    #completing categorical feature - embarked port
        #complete with most common occurance (mode)
    most_common_port = train_df.Embarked.dropna().mode()[0]
    for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].fillna(most_common_port)
        #convert to numeric
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        #complete Fare withe median
    test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

        #create Fare Bands
    train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
    print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)) #check correlation
        #replace with bands value
    for dataset in combine:
        dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

    train_df = train_df.drop(['FareBand'], axis=1)
    combine = [train_df, test_df]

    #divide X and Y from data set
    X_train = train_df.drop("Survived", axis=1)
    Y_train = train_df["Survived"]
    X_test  = test_df.drop("PassengerId", axis=1).copy()
    print(X_train.shape, Y_train.shape, X_test.shape)

    #MODEL - coursera 2 course
        #DeepNN
    layers = [{"n":18,      "activation":"relu",        "dropout_keep_prob":1.0},
              {"n":10,      "activation":"relu",        "dropout_keep_prob":1.0},    
              {"n":1,      "activation":"sigmoid",     "dropout_keep_prob":1.0}]

    deep_nn_adam = coursera_dp_master.DeepNN(layers = layers)
    #train
    train_X, train_Y, test_X = X_train.values.T, Y_train.values.reshape(1, Y_train.shape[0]), X_test.values.T

    costs = deep_nn_adam.fit(train_X, train_Y, num_epocs=2000, mini_batch_size=64, learning_rate=0.001, adam_hyperparams=coursera_dp_master.AdamHyperParams())
    plt.clf()
    plt.plot(costs)
    #plt.show()

    #predict
    train_Y_pred = deep_nn_adam.predict(train_X)
    test_Y = deep_nn_adam.predict(test_X)

     # Print train Errors= 
    train_accuracy = coursera_dp_master.calc_accuracy(train_Y_pred, train_Y)
    print("train accuracy: {} %".format(train_accuracy))

    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": test_Y.flatten()
    })
    #submission.to_csv('output/submission.csv', index=False)
    """


    
