#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

import argparse
import os
from six import text_type as _text_type
from mmdnn.conversion.examples.imagenet_test import TestKit
import torch
import torchvision.models as models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def _main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--network',
                        type=_text_type, help='Model Type', required=True,
                        choices=model_names)

    parser.add_argument('-i', '--image', type=_text_type, help='Test Image Path')

    args = parser.parse_args()

    file_name = "imagenet_{}.pth".format(args.network)
    if not os.path.exists(file_name):
        model = models.__dict__[args.network](pretrained=True)
        torch.save(model, file_name)
        print("PyTorch pretrained model is saved as [{}].".format(file_name))
    else:
        print("File [{}] existed!".format(file_name))
        model = None

    if args.image:
        if model == None:
            model = torch.load(file_name)
        import numpy as np
        func = TestKit.preprocess_func['pytorch'][args.network]
        img = func(args.image)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).copy()
        data = torch.from_numpy(img)
        data = torch.autograd.Variable(data, requires_grad=False)

        model.eval()
        predict = model(data).data.numpy()
        predict = np.squeeze(predict)
        top_indices = predict.argsort()[-5:][::-1]
        result = [(i, predict[i]) for i in top_indices]
        print(result)

        # layer_name = 'block2_pool'
        # intermediate_layer_model = keras.Model(inputs=model.input,
        #                                  outputs=model.get_layer(layer_name).output)
        # intermediate_output = intermediate_layer_model.predict(img)
        # print (intermediate_output)
        # print (intermediate_output.shape)
        # print ("%.30f" % np.sum(intermediate_output))


if __name__ == '__main__':
    _main()
