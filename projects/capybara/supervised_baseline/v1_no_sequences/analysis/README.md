# Baseline results & analysis
    
## Walking VS Standing

### Results
* Accuracy = 98%. 
* Conclusion: the model does well but it's a trivial task.

### Settings
* skip stable encoding = False
* union moving average window = 255
* vote window = 125
* num epochs = 10
* `MAX_LABEL_REPS = {0:5000, 4:5000}`

## All activities

### Results
* Accuracy = 33%
* Conclusion: the model gets `WALKING_UPSTAIRS` and `LAYING` right but it mixes 
all the "walking" categories (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`), 
and all the "still" categories (`LAYING`, `STANDING`, `SITTING`). Also, it seems 
that it gets `WALKING_UPSTAIRS` and `LAYING` right because they are the last ones
of their category ("walking" or "still") to be seen.

### Settings
* skip stable encoding = False
* union moving average window = 255
* vote window = 125
* num epochs = 10
* `MAX_LABEL_REPS = {i:5000 for i in range(len(LABELS))}`

## Walking VS Walking upstairs

### Results

* Accuracy = 50%.
* Conclusion: it does no better than random and thinks everything is 
`WALKING_UPSTAIRS`.

### Settings
* skip stable encoding = False
* union moving average window = 255
* vote window = 125
* num epochs = 10
* `MAX_LABEL_REPS = {0:5000, 1:5000}`