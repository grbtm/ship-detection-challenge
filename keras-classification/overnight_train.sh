# airbus 0-1
python train.py --train_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/airbus_ship_data/train_zero_to_one_ships.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_0-1 --lr 0.0003 --mini_batch_size 32 --epochs 2
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_0-1 --log_file ../logs/airbus_0-1/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_zero_to_one_ships.csv --log_dir ../logs/airbus_0-1 --log_file ../logs/airbus_0-1/test_result_airbus.txt

# bodensee
python train.py --train_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTrain.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/bodensee --lr 0.0003 --mini_batch_size 32 --epochs 2
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/bodensee --log_file ../logs/bodensee/test_result_bodensee.txt

# airbus balanced
python train.py --train_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/airbus_ship_data/train_balanced_zero_to_one_ships.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_balanced --lr 0.0003 --mini_batch_size 32 --epochs 2
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_balanced --log_file ../logs/airbus_balanced/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_balanced_zero_to_one_ships.csv --log_dir ../logs/airbus_balanced --log_file ../logs/airbus_balanced/test_result_airbus.txt

# airbus 0-3
python train.py --train_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/airbus_ship_data/train_zero_to_three_ships.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_0-3 --lr 0.0003 --mini_batch_size 32 --epochs 2
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_0-3 --log_file ../logs/airbus_0-3/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_zero_to_three_ships.csv --log_dir ../logs/airbus_0-3 --log_file ../logs/airbus_0-3/test_result_airbus.txt

# airbus all
python train.py --train_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/airbus_ship_data/train_zero_to_four_ships.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_0-n --lr 0.0003 --mini_batch_size 32 --epochs 2
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/airbus_0-n --log_file ../logs/airbus_0-n/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_zero_to_four_ships.csv --log_dir ../logs/airbus_0-n --log_file ../logs/airbus_0-n/test_result_airbus.txt

# augmentation
#?

# densenet
# airbus 0-1
python train.py --train_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/airbus_ship_data/train_zero_to_one_ships.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/dense_airbus_0-1 --lr 0.0003 --mini_batch_size 32 --epochs 2 --densenet
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/dense_airbus_0-1 --log_file ../logs/dense_airbus_0-1/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_zero_to_one_ships.csv --log_dir ../logs/dense_airbus_0-1 --log_file ../logs/dense_airbus_0-1/test_result_airbus.txt


# bodensee
python train.py --train_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTrain.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/dense_bodensee --lr 0.0003 --mini_batch_size 32 --epochs 2 --densenet
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/dense_bodensee --log_file ../logs/dense_bodensee/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_zero_to_one_ships.csv --log_dir ../logs/dense_airbus_balanced --log_file ../logs/dense_airbus_balanced/test_result_airbus.txt

# airbus all
python train.py --train_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_img_dir  ../../data/searchwing/Hackathon/RAW_DATA --train_csv ../../data/searchwing/airbus_ship_data/train_zero_to_four_ships.csv --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/dense_airbus_0-n --lr 0.0003 --mini_batch_size 32 --epochs 2 --densenet
python test.py --test_img_dir ../../data/searchwing/Hackathon/RAW_DATA --test_csv ../../data/searchwing/Hackathon/SingleFrame_ImageClassification/labelsTest.csv --log_dir ../logs/dense_airbus_0-n --log_file ../logs/dense_airbus_0-n/test_result_bodensee.txt
python test.py --test_img_dir ../../data/searchwing/airbus_ship_data/train_v2 --test_csv ../../data/searchwing/airbus_ship_data/valid_zero_to_four_ships.csv --log_dir ../logs/dense_airbus_0-n --log_file ../logs/dense_airbus_0-n/test_result_airbus.txt --densenet
