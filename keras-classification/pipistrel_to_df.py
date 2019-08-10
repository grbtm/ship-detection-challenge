import os, argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipistrel_folder', type=str, required=True)
    parser.add_argument('--csv_target', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pipistrel_folder = args.pipistrel_folder
    csv_target = args.csv_target

    classes = ['nature', 'boat']
    dic = {}
    dic['ImageId'] = []
    dic['class'] = []
    for i, cls in enumerate(classes):
        for file in os.listdir(os.path.join(pipistrel_folder, cls)):
            dic['ImageId'].append(file)
            dic['class'].append(cls)

    df = pd.DataFrame(data=dic)

    df.to_csv(csv_target)
