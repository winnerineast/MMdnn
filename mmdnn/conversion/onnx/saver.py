import onnx


def save_model(MainModel, network_filepath, weight_filepath, dump_filepath):
    model = MainModel.KitModel(weight_filepath)
    onnx.save(model, dump_filepath)
    print('ONNX model file is saved as [{}], generated by [{}.py] and [{}].'.format(
        dump_filepath, network_filepath, weight_filepath))
