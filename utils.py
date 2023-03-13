

def get_model(args, model_type):
    """return autodecoder or denoiser"""
    if model_type == 'AutoEncoder':
        feats = args['model']['num_features']
        chs = args['model']['num_channels']
        lays = args['model']['num_layers']
        reg = args['model']['regularized']
        reg_chs = args['denoiser']['num_channels']
        model = ShapeAutoEncoder(
            features=feats, channels=chs, layers=lays, reg=reg, reg_channels=reg_chs).cuda()
        return model
    elif model_type == 'Denoiser':
        chs = args['denoiser']['num_channels']
        lays = args['denoiser']['num_layers']
        model = ShapeDenoiser(num_channels=chs, num_layers=lays).cuda()
        return model
    else:
        raise Exception(f'Undefined Model Type {model_type}')
