['angry', 'calm' ,'disgust', 'fear', 'happy', 'neutral', 'sad','suprise']
import pandas as pd

df = pd.read_csv('./iemocap_features_2.csv')

# def filter_rows_by_values(df, col, values):
#     return df[~df[col].isin(values)]

print(df.labels.unique())
# # ['neu' 'xxx' 'fru' 'ang' 'sad' 'hap' 'exc' 'sur' 'oth' 'fea' 'dis']
# ['xxx', 'oth' 'exc']
# df = filter_rows_by_values(df, 'labels', ['xxx', 'oth' , 'exc' ,'fru'])
# df.replace(['neu', 'ang', 'sad', 'hap', 'sur', 'fea', 'dis'], ['neutral','angry','sad','happy', 'surprise', 'fear', 'disgust' ], inplace=True)

# df.to_csv('./iemocap_features_2.csv', index=False)