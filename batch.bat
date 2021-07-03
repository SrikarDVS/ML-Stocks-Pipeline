@echo off

python "get_data.py"
python "get_value.py"
python "train_model.py"
python "test_model.py"
python "predict.py"
pause