import os
import pandas as pd

if __name__ == '__main__':
    pipistrel_folder = '../../data/searchwing/Hackathon/SingleFrame_ObjectProposalClassification/test'
    csv_target = '../../workspace/data/searchwing/Hackathon/SingleFrame_ObjectProposalClassification/test/pipistrel_image_cls.csv'
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
