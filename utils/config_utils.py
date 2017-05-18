class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def init_configs(flags):
    config = Config()

    # Check if path to VGG-16 is defined
    if flags.vgg is None:
        raise ValueError("Path to VGG-16 model must be defined")
    config.vgg = flags.vgg

    # Check if model ID is specified
    if flags.model_id is None:
        raise ValueError("Model ID must be defined")
    config.id = flags.model_id

    if flags.type is None:
        raise ValueError("Run type must be specified")
    config.type = flags.type
    config.is_train = flags.type == 'train'
    config.continue_train = bool(getattr(flags, 'continue'))

    if flags.model_log_dir is None:
        raise ValueError("Path to log directory must be specified")
    config.log_dir = flags.model_log_dir + '/' + config.id

    if flags.images_input is None:
        raise ValueError("Path to images must be specified")
    config.inputs = flags.images_input
    config.output_format = flags.images_input_format

    config.save_images = flags.images_output_enable is not None
    config.outputs = flags.images_output
    config.output_format = flags.images_output_format
    config.output_save_step = flags.images_output_step
    if config.save_images is not None:
        if config.outputs is not None:
            config.output_path = config.outputs + '/' + config.id
        else:
            raise ValueError("When enabled image saving path must be specified")

    config.batch_size = 1
    if flags.images_batch_size is not None:
        config.batch_size = flags.images_batch_size

    config.epochs = flags.model_epochs
    config.learning_rate = flags.model_learning_rate
    config.learning_decay = flags.model_learning_rate_decay
    config.learning_decay_step = flags.model_learning_rate_step

    config.save_model = bool(flags.model_save)
    if config.save_model is not None and flags.model_save_path is None:
        raise ValueError("When model is saved, path must be specified")
    config.save_path = flags.model_save_path
    config.save_model_step = flags.model_save_pass

    if not config.continue_train:
        config.model_dir = config.save_path
    if config.continue_train:
        if flags.model_dir is not None:
            config.model_dir = flags.model_dir
        else:
            raise ValueError("When training continue, path to model must be specified")

    config.model_fullpath = config.model_dir + '/' + config.id
    config.save_path = '/'
    if config.save_model:
        config.save_path = config.save_path + '/' + config.id

    return config
