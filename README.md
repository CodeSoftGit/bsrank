# bsrank
*Documentation for the script is not yet completed.*

## Dependencies
Dependencies can be installed with:
```
pip install joblib pandas scikit-learn ttkthemes
```

## Dataset structure
```
dataset\
  13star\
    beatmap-128bpm.dat
  20star\
    Beat Map-203bpm.dat
```
As you can see, each folder has it's own star rating. Each file must end with the map's starting BPM!
#### *The UI has no capability of training yet. Dig around in train.py to train a model of your own!* *hint hint its in the `if __name__ == "__main__":` statement*