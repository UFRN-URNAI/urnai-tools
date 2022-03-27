class ModelBuilder:
    LAYER_INPUT = 'input'
    LAYER_OUTPUT = 'output'
    LAYER_FULLY_CONNECTED = 'fullyconn'
    LAYER_CONVOLUTIONAL = 'conv'
    LAYER_MAXPOOLING = 'maxpooling'
    LAYER_FLATTEN = 'flatten'

    DEFAULT_BUILD_MODEL = [
        {
            'type': LAYER_INPUT,
            'nodes': 50,
            'shape': [None, 10],
        },
        {
            'type': LAYER_FULLY_CONNECTED,
            'nodes': 50,
            'name': 'fc1',
        },
        {
            'type': LAYER_FULLY_CONNECTED,
            'nodes': 50,
            'name': 'fc2',
        },
        {
            'type': LAYER_OUTPUT,
            'length': 50,
        },
    ]

    DEFAULT_BUILD_CONV_MODEL = [
        {
            'type': LAYER_INPUT,
            'nodes': 50,
            'shape': [None, 10],
        },
        {
            'type': LAYER_CONVOLUTIONAL,
            'filters': 1,
            'filter_shape': (3, 3),
            'padding': 'same',
            'name': 'conv1',
            'max_pooling_pool_size_shape': (2, 2),
        },
        {
            'type': LAYER_FULLY_CONNECTED,
            'nodes': 50,
            'name': 'fc1',
        },
        {
            'type': LAYER_FULLY_CONNECTED,
            'nodes': 50,
            'name': 'fc2',
        },
        {
            'type': LAYER_OUTPUT,
            'length': 50,
        },
    ]

    def __init__(self):
        self.layers = []

    def add_input_layer(self, nodes=50):
        # shape = None
        # if custom_shape == None:
        #     shape = [None, size]
        # else:
        #     shape = custom_shape

        # if type(shape) == list:
        self.layers.append({
            'type': ModelBuilder.LAYER_INPUT,
            'nodes': nodes,
            'shape': [],
        })
        # else:
        #     raise TypeError('Input layer shape should be a list with its dimensions in it.')

    def add_convolutional_layer(self, filters=2, kernel_size=(3, 3), strides=(1, 1),
                                padding='valid',
                                data_format=None, dilation_rate=(1, 1), groups=1, activation='relu',
                                name='default', input_shape=None):
        if name == 'default':
            cont = 0
            for layer in self.layers:
                if 'name' in layer:
                    if 'default' in layer['name']:
                        cont += 1

            name = ModelBuilder.LAYER_CONVOLUTIONAL + str(cont)

        if type(filters) == int:
            self.layers.append(
                {
                    'type': ModelBuilder.LAYER_CONVOLUTIONAL,
                    'filters': filters,
                    'filter_shape': kernel_size,
                    'strides': strides,
                    'padding': padding,
                    'data_format': data_format,
                    'dilation_rate': dilation_rate,
                    'groups': groups,
                    'activation': activation,
                    'name': name,
                    'input_shape': input_shape,
                },
            )
        else:
            raise TypeError("Convolutional layer's filters should be an integer.")

    def add_maxpooling_layer(self, pool_size=(2, 2), strides=None, padding='valid',
                             data_format=None, name='default'):
        if name == 'default':
            cont = 0
            for layer in self.layers:
                if 'name' in layer:
                    if 'default' in layer['name']:
                        cont += 1

            name = ModelBuilder.LAYER_MAXPOOLING + str(cont)

        if type(pool_size) == tuple:
            self.layers.append({
                'type': ModelBuilder.LAYER_MAXPOOLING,
                'pool_size': pool_size,
                'strides': strides,
                'padding': padding,
                'data_format': data_format,
            })
        else:
            raise TypeError("Maxpooling layer's filter_shape should be a tuple.")

    def add_flatten_layer(self, data_format=None, name='default'):
        if name == 'default':
            cont = 0
            for layer in self.layers:
                if 'name' in layer:
                    if 'default' in layer['name']:
                        cont += 1

            name = ModelBuilder.LAYER_FLATTEN + str(cont)

        self.layers.append({
            'type': ModelBuilder.LAYER_FLATTEN,
            'data_format': data_format,
        })

    def add_fullyconn_layer(self, nodes, name='default'):
        if name == 'default':
            cont = 0
            for layer in self.layers:
                if 'name' in layer:
                    if 'default' in layer['name']:
                        cont += 1

            name = ModelBuilder.LAYER_FULLY_CONNECTED + str(cont)

        if type(nodes) == int:
            self.layers.append({
                'type': ModelBuilder.LAYER_FULLY_CONNECTED,
                'nodes': nodes,
                'name': name,
            })
        else:
            raise TypeError("Fully connected layer's number of nodes should be an integer.")

    def add_output_layer(self):
        # if type(length) == int:
        self.layers.append({
            'type': ModelBuilder.LAYER_OUTPUT,
            'length': 0,
        })
        # else:
        #     raise TypeError('Output layer's length should be an integer.')

    def get_model_layout(self):
        return self.layers

    @staticmethod
    def has_convolutional_layers(layers):
        for layer in layers:
            if layer['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                return True
        return False

    @staticmethod
    def get_last_convolutional_layer_index(layers):
        cont = -1
        for layer in layers:
            if layer['type'] == ModelBuilder.LAYER_CONVOLUTIONAL:
                if cont == -1:
                    cont = 0
                else:
                    cont += 1
        return cont

    @staticmethod
    def is_last_conv_layer(layer, layers):
        return ModelBuilder.get_last_convolutional_layer_index(layers) == layers.index(layer)
