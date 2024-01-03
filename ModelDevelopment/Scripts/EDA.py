import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats


# numerical univariate fuction
def UVA_numeric(data, var_group):
  '''
  Univariate_Analysis_numeric
  takes a group of variables (INTEGER and FLOAT) and plot/print all the descriptives and properties along with KDE.
  '''
  size = len(var_group)
  plt.figure(figsize = (7*size,3), dpi = 100)

  #looping for each variable
  for j,i in enumerate(var_group):
    # calculating descriptives of variable
    mini = data[i].min()
    maxi = data[i].max()
    ran = data[i].max()-data[i].min()
    mean = data[i].mean()
    median = data[i].median()
    st_dev = data[i].std()
    skew = data[i].skew()
    kurt = data[i].kurtosis()
    # calculating points of standard deviation
    points = mean-st_dev, mean+st_dev
    #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.kdeplot(data[i], fill=True)
    sns.lineplot(x=points, y=[0,0], color = 'black', label = "std_dev")
    sns.scatterplot(x=[mini,maxi], y=[0,0], color = 'orange', label = "min/max")
    sns.scatterplot(x=[mean], y=[0], color = 'red', label = "mean")
    sns.scatterplot(x=[median], y=[0], color = 'blue', label = "median")
    plt.xlabel('{}'.format(i), fontsize = 20)
    plt.ylabel('density')
    plt.title('std_dev = {}; kurtosis = {};\nskew = {}; range = {}\nmean = {}; median = {}'.format((round(points[0],2),round(points[1],2)),
                                                                                                   round(kurt,2),
                                                                                                   round(skew,2),
                                                                                                   (round(mini,2),round(maxi,2),round(ran,2)),
                                                                                                   round(mean,2),
                                                                                                   round(median,2)))
def UVA_outlier(data, var_group):
  '''
  Univariate_Analysis_outlier:
  takes a group of variables (INTEGER and FLOAT) and plot/print boplot and descriptives\n
  '''

  size = len(var_group)
  plt.figure(figsize = (7*size,3), dpi = 100)

  #looping for each variable
  for j,i in enumerate(var_group):

    # calculating descriptives of variable
    quant25 = data[i].quantile(0.25)
    quant75 = data[i].quantile(0.75)
    IQR = quant75 - quant25
    med = data[i].median()
    whis_low = quant25-(1.5*IQR)
    whis_high = quant75+(1.5*IQR)

    # Calculating Number of Outliers
    outlier_high = len(data[i][data[i]>whis_high])
    outlier_low = len(data[i][data[i]<whis_low])

    #Plotting the variable with every information
    plt.subplot(1,size,j+1)
    sns.boxplot(data=data[i], orient="v")
    plt.ylabel('{}'.format(i))
    plt.title('With Outliers\nIQR = {}; Median = {} \n 1st,3rd  quartile = {};\n Outlier (low/high) = {} \n'.format(
                                                                                                   round(IQR,2),
                                                                                                   round(med,2),
                                                                                                   (round(quant25,2),round(quant75,2)),
                                                                                                   (outlier_low,outlier_high)
                                                                                                   ))

def BVA_categorical_plot(data, tar, cat):
    '''
    performing a chi-squared test to determine if there is a significant association between two categorical variables.
    take data and two categorical variables,
    calculates the chi2 significance between the two variables
    and prints the result with countplot & CrossTab
    '''
    #isolating the variables
    data = data[[cat,tar]][:]

    #forming a crosstab
    table = pd.crosstab(data[tar],data[cat])

    #performing chi2 test
    from scipy.stats import chi2_contingency
    chi, p, dof, expected = chi2_contingency(table)

    #checking whether results are significant
    if p<0.05:
        sig = True
    else:
        sig = False

    #setting up the matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    desired_order = sorted(data[tar].unique(), reverse=False)

    #plotting grouped plot
    sns.countplot(x=cat, hue=tar, data=data, ax=axes[0], hue_order=desired_order)
    axes[0].set_title(f"p-value = {round(p, 8)}\n difference significant? = {sig}\n")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)


    #plotting percent stacked bar plot
    ax1 = pd.crosstab(data[cat], data[tar], normalize='index')[desired_order]
    ax1.plot(kind='bar', stacked=True, ax=axes[1], title=str(ax1))
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=90)


    plt.tight_layout()
    plt.show()
    
def remove_outliers(data, column_name):
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    data_no_outlier = data[(data[column_name] >= Q1-1.5*IQR) & (data[column_name] <= Q3+1.5*IQR)]
    return data_no_outlier

def one_way_ANOVA(data, cont, cat, remove_outliers_flag=False):
    '''
    find differences between  groups of an independent variable on a continuous
    '''
    # Remove outliers from 'cont' column if flag is True
    if remove_outliers_flag:
        data = remove_outliers(data, cont)

    # Creating n samples based on categories
    samples = []
    for cat_value in data[cat].unique():
        sample = data[cont][data[cat] == cat_value].values
        samples.append(sample)

    # Performing Kruskal-Wallis H-test
    h_stat, p_val = stats.kruskal(*samples)

    # Table
    table = pd.pivot_table(data=data, values=cont, columns=cat, aggfunc=np.mean).applymap("{:.2f}".format)

    # Plotting
    plt.figure(figsize = (10, 2), dpi=140)

    # Barplot
    plt.subplot(1, 2, 1)
    sns.barplot(x=cat, y=cont, data=data)
    plt.ylabel(f'Mean {cont}')
    plt.xlabel(cat)
    plt.title(f'Kruskal-Wallis H-test p-value = {p_val:.2e} \n{table}')

    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=cat, y=cont, data=data)
    plt.title('Categorical boxplot')                                                                                                  
