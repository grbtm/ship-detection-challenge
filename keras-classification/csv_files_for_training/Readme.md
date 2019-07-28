# CSV Files to the different datasets

All files use the same convention for the column names.

The filename is in the "ImageId" col and the class is a string of "nature" or "boat" in a class column.

Files and corresponding image folders:

- Bodensee Proposal test: in the Hackathon/SingleFrame_ObjectProposalClassification/test.zip
- Bodensee test: Hackathon/RAW_DATA
- Bodensee train: Hackathon/RAW_DATA (didn't use that)
- Mediterranean refelction test: Images Friedrich posted during the deep berlin Hackathon
- All Airbus csv files use the train_v2 image folders

We transformed the masks from the csv file in the Airbus data to a classes.  
After that we created different training sets:
- images with no or one ship
- images with no, one or two ships
- images with no, now, two or three ships
- images with no, one, two, three or four ships
- a dataset with a balanced number if images with no or one ship on them

All airbus sets have a train / val split of 0.9 / 0.1?!
