import tensorflow as tf

def get_preprocess(model_name):
        if model_name == 'vgg16':
            return tf.keras.applications.vgg16.preprocess_input
        
        if model_name == 'inception':
            return tf.keras.applications.inception_v3.preprocess_input
        
        if model_name == 'mobilenet':
            return tf.keras.applications.mobilenet.preprocess_input
        
        if model_name == 'resnet':
            return tf.keras.applications.resnet.preprocess_input

        raise ValueError('Unrecognized network name!')

class keras_models:

    def __init__(self, num_classes, network, drop_value=0.75, freeze=False, act='linear', kernel='he_uniform'):
        
        self.network = network
        self.prep_func = get_preprocess(network)
        
        base_model = self.get_base_model()
        
        if freeze:
            for layer in base_model.layers:
                layer.trainable = False
        
        x = tf.keras.layers.Dropout(drop_value)(base_model.output)
        x = tf.keras.layers.Dense(num_classes, activation=act, kernel_initializer = kernel, name='fine_out')(x)
        self.model = tf.keras.models.Model(inputs = base_model.input, outputs = x)
        
    def get_base_model(self):
        if self.network == 'vgg16':
            return tf.keras.applications.vgg16.VGG16(input_shape=(224,224,3), include_top=False,
                                                      pooling='avg', weights='imagenet')

        
        if self.network == 'inception':
            return tf.keras.applications.inception_v3.InceptionV3(input_shape=(299,299,3), include_top=False,
                                                               pooling ="avg", weights='imagenet')
        
        if self.network == 'mobilenet':
            return tf.keras.applications.mobilenet.MobileNet(input_shape=(224,224,3), alpha=1, include_top=False, 
                                                          pooling='avg', weights='imagenet')
        
        if self.network == 'resnet':
            return tf.keras.applications.resnet.ResNet50(input_shape=(224,224,3), include_top=False,
                                                      pooling='avg', weights='imagenet')
        