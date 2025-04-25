 
from model.autoregressive import ( 
    autoregressive 
)
from model.recurrentgat import RecurrentGAT

MODEL_LIST = [ 
            autoregressive, RecurrentGAT
            ]


MODEL_DICT = {x.__name__.lower(): x for x in MODEL_LIST}


def str2model(model_name):

    try:
        model = MODEL_DICT[model_name]

    except:
        raise NotImplementedError(f"{model_name} not implemented. Check model registry")

    return model
