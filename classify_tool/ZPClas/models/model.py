import torch.nn as nn
from torchvision import datasets, models



def set_parameter_requires_grad(model, feature_learning):
    '''if or not update params of model
    '''
    if feature_learning:
        for param in model.parameters():
            param.requires_grad = True

def initialize_model(model_name, num_classes, feature_learn=True, use_pretrained=True):
    '''
    num_classes: how many categores
    use_preatrained: if or not loading pretrained_model
    '''
    model_ft = None
    input_size = 0

    # assert model_name in []
    if model_name == "convnext":
        '''need torchvison>=1.2
        2022.1-beyond swin-transformer
        convnext_tiny = models.convnext_tiny(pretrained=True)
        convnext_small = models.convnext_small(pretrained=True)
        convnext_base = models.convnext_base(pretrained=True)
        convnext_large = models.convnext_large(pretrained=True)
        '''
        model_ft = models.convnext_tiny(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnext":
        '''32_4d: 32组数，4各支路初始通道数;
        '''
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)

        # model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet50":
        """ Resne50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        #jsut change the last fc_layer
        num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet101":
        """ Resne101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet101":
        """ Resne101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    

    #轻量化网络
    elif model_name == "shufflenet_v2_x0_5":
        """ shufflenet_v2_x0_5
        [24, 48, 96, 192, 1024]
        """
        model_ft = models.shufflenet_v2_x0_5(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "shufflenet_v2_x1_0":
        """ shufflenet_v2_x1_0
        [24, 116, 232, 464, 1024]
        """
        model_ft = models.shufflenet_v2_x1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "shufflenet_v2_x1_5":
        """ shufflenet_v2_x1_5
        [24, 176, 352, 704, 1024]
        """
        model_ft = models.shufflenet_v2_x1_5(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "shufflenet_v2_x2_0":
        """ shufflenet_v2_x2_0
        [24, 244, 488, 976, 2048]
        """
        model_ft = models.shufflenet_v2_x2_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes) 
        input_size = 224

    elif model_name == "inception":
        """ Inception v3 
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_learn)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()
    
    return model_ft, input_size
