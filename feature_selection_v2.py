import pandas as pd
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt


# Create your df here:
df = pd.read_csv("profiles.csv")
pd.set_option('display.max_columns', None)


# Mapping values to certain features
drinks_mapping = {'not at all': 0, 'rarely': 1, 'socially': 2, 'often': 3, 'very often': 4, 'desperately': 5}
smokes_mapping = {'no': 0, 'trying to quit': 1, 'sometimes': 2, 'when drinking': 3, 'yes': 4}
drugs_mapping = {'never': 0, 'sometimes': 1, 'often': 2}
sex_mapping = {'m': 0, 'f': 1}

# dropping unnecessary rows
df['education'] = df[~df.education.str.contains('space camp', na=False)]['education']  # Drops rows containing "space camp"
print(df.education.head())

# An attempt to rank education both on level of schooling and whether the individual finished or not
work = 'working on '
drop = 'dropped out of '
grad = 'graduated from '

education_mapping = {
    '{}high school'.format(drop): 0,
    '{}high school'.format(work): 1,
    'high school': 2,
    '{}high school'.format(grad): 3,
    '{}two-year college'.format(drop): 4,
    '{}two-year college'.format(work): 5,
    'two-year college': 6,
    '{}two-year college'.format(grad): 7,
    '{}college/university'.format(drop): 8,
    '{}college/university'.format(work): 9,
    'college/university': 10,
    '{}college/university'.format(grad): 11,
    '{}masters program'.format(drop): 12,
    '{}masters program'.format(work): 13,
    'masters program': 14,
    '{}masters program'.format(grad): 15,
    '{}med school'.format(drop): 12,
    '{}med school'.format(work): 13,
    'med school': 14,
    '{}med school'.format(grad): 15,
    '{}law school'.format(drop): 12,
    '{}law school'.format(work): 13,
    'law school': 14,
    '{}law school'.format(grad): 15,

}

#Attempting to rank diet based on how strict/loose someone is with their diet, so diets with more restrictions are rated higher
#and people who said "strictly" were ranked highest within each category of diet
diet_mapping = {
    'anything': 0,
    'mostly anything': 1,
    'strictly anything': 2,
    'mostly vegetarian': 3,
    'vegetarian': 4,
    'strictly vegetarian': 5,
    'mostly vegan': 6,
    'vegan': 7,
    'strictly vegan': 8,
    'mostly other': 2,
    'other': 3,
    'strictly other': 4,
    'mostly halal': 1,
    'halal': 2,
    'strictly halal': 3,
    'mostly kosher': 1,
    'kosher': 2,
    'strictly kosher': 3
}

#Editing the rows in df['sign'] so they all just say the sign and can be used in further machine learning algorithms
signs = ['aquarius', 'aries', 'taurus', 'gemini', 'cancer', 'leo',
         'virgo', 'libra', 'scorpio', 'sagittarius', 'capricorn', 'pisces'
         ]

df.dropna(subset=['sign'], inplace=True)
df['sign_refined'] = np.where(df['sign'].str.contains(signs[0]), signs[0],
                        np.where(df['sign'].str.contains(signs[1]), signs[1],
                        np.where(df['sign'].str.contains(signs[2]), signs[2],
                        np.where(df['sign'].str.contains(signs[3]), signs[3],
                        np.where(df['sign'].str.contains(signs[4]), signs[4],
                        np.where(df['sign'].str.contains(signs[5]), signs[5],
                        np.where(df['sign'].str.contains(signs[6]), signs[6],
                        np.where(df['sign'].str.contains(signs[7]), signs[7],
                        np.where(df['sign'].str.contains(signs[8]), signs[8],
                        np.where(df['sign'].str.contains(signs[9]), signs[9],
                        np.where(df['sign'].str.contains(signs[10]), signs[10],
                        np.where(df['sign'].str.contains(signs[11]), signs[11],
                        'No'))))))))))))


#Adding the maps to the dataframe
df['drinks_code'] = df.drinks.map(drinks_mapping)
df['drugs_code'] = df.drugs.map(drugs_mapping)
df['smokes_code'] = df.smokes.map(smokes_mapping)
df['education_code'] = df.education.map(education_mapping)
df['diet_code'] = df.diet.map(diet_mapping)
df['sex_code'] = df.sex.map(sex_mapping)
df['income_reported'] = df.income.drop(df[df.income == -1].index) #There was some weird data in income that I had to remove


essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
drop_words = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]

#Combining the essays
#df['all_essays'] = all_essays.apply(lambda x: str(x).lower().split(), axis=1).reset_index(drop=True)
all_essays = df[essay_cols].apply(lambda x: str(x).lower(), axis=1).reset_index(drop=True)
df['all_essays'] = df[essay_cols].apply(lambda x: str(x).lower(), axis=1).reset_index(drop=True)
df['essay_length'] = df['all_essays'].apply(lambda x: str(x).split()).apply(lambda x: len(x))

# Removing the NaNs
df.dropna(subset=essay_cols, inplace=True)
df.to_pickle("clean_df_v2")
