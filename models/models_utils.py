import json
from os import path
from scikit_wrappers import CausalCNNEncoderClassifier
from config import PATH_TO_MODELS, PATH_TO_MODELS_UCR


def load_pretrained(model_name=None, model_type='encoder', model_origin='ucr'):
    # Set path
    if model_origin=='ucr':
        path_to_model = path.join(PATH_TO_MODELS, PATH_TO_MODELS_UCR)
    elif model_origin == 'default':
        path_to_model = '.'
        model_name = 'default'
        model_type = None
    else:
        raise(NotImplemented, "Only models pretrained on UCR are currently available")

    # Load hyperparams
    model = CausalCNNEncoderClassifier()
    hf = open( path.join(path_to_model,
                         model_name + '_hyperparameters.json'),
               'r' )
    hp_dict = json.load(hf)
    hf.close()
    hp_dict['cuda'] = False
    hp_dict['gpu'] = False
    # Load model
    model.set_params(**hp_dict)

    if model_type == 'encoder':
        model.load_encoder(path.join(path_to_model,
                                     model_name)
                           )
    elif model_type == 'classifier':
        model.load( path.join(path_to_model,
                              model_name)
                    )
    return model


"""
if __name__ == '__main__':
    model_name = 'PigCVP'
    clf = load_pretrained(model_name)
"""
