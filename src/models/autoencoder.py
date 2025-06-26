from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam

class AutoencoderFeatureExtractor:
    def __init__(self, input_dim, latent_dim=16):  # Change to 16
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = None
        self.autoencoder = self.build_model()


    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))

        # Encoder
        x = Dense(64, activation='relu')(input_layer)
        x = Dense(32, activation='relu')(x)
        bottleneck = Dense(
            self.latent_dim,
            activation='relu',
            activity_regularizer=l1(1e-6)  # Lower sparsity penalty
        )(x)

        # Decoder
        x = Dense(32, activation='relu')(bottleneck)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(self.input_dim, activation='linear')(x)

        # Build models
        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        self.encoder = Model(inputs=input_layer, outputs=bottleneck)

        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder


    def train(self, X, epochs=50, batch_size=32):
        self.autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    def extract_features(self, X):
        return self.encoder.predict(X)
