import torch
def getBestModel(model):
    # load the best model
    modelSavePath='best_models/best_model_'+model.__class__.__name__+'.pth'
    model.load_state_dict(torch.load(modelSavePath))
    return model