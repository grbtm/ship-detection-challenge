import os
import pandas as pd

if __name__ == '__main__':
    folder = '../../data/searchwing/reflect_val'
    write_to_csv = '../../data/searchwing/reflect_val.csv'
    df = pd.DataFrame()
    filenames = []
    for file in os.listdir(folder):
        if file.endswith('.jpg'):
            filenames.append(file)

    df['ImageId'] = filenames
    classes = []
    for _ in range(len(filenames)):
        classes.append('nature')
    df['class'] = classes

    df.to_csv(write_to_csv)
