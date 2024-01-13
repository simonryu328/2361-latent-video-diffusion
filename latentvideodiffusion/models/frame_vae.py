import equinox as eqx
import jax
import jax.numpy as jnp
import pickle

class ConvResBlock(eqx.Module):
    inp_layer: eqx.nn.Conv
    outp_layer: eqx.nn.Conv
    layer_norm: eqx.nn.LayerNorm
    def __init__(self, n_latent, h, w, key):
        a,b = jax.random.split(key, 2) 
        padding = [(1,1),(1,1)]
        shape = (n_latent, h, w)
        self.inp_layer = eqx.nn.Conv(num_spatial_dims=2, in_channels=n_latent, out_channels=n_latent*2, kernel_size=(3,3), stride=1,padding=padding,key=a)
        self.outp_layer = eqx.nn.Conv(num_spatial_dims=2, in_channels=n_latent*2, out_channels=n_latent, kernel_size=(3,3), stride=1,padding=padding,key=b)
        self.layer_norm = eqx.nn.LayerNorm(shape=shape, use_bias=False, use_weight=False) #how can the shape be calculated? #this can't be pickled...why?
        
        
    def __call__(self, x):
        a = self.inp_layer(x)
        b = jax.nn.leaky_relu(a)
        c = self.outp_layer(b)
        d = c + x
        y = self.layer_norm(d) #ValueError: `LayerNorm(shape)(x)` must satisfy the invariant `shape == x.shape`Received `shape=(8,) and `x.shape=(8, 256, 150)`
                               #You might need to replace `layer_norm(x)` with `jax.vmap(layer_norm)(x)`.
        return y

    # def __getstate__(self):
    #     # Include the layer_norm shape in the state
    #     state = self.__dict__.copy()
    #     state['layer_norm'] = self.layer_norm.shape
    #     return state

    # def __setstate__(self, state):
    #     # Restore the layer_norm attribute during unpickling
    #     layer_norm_shape = state.pop('layer_norm', None)
    #     self.layer_norm = eqx.nn.LayerNorm(shape=layer_norm_shape, use_bias=False, use_weight=False)
    #     self.__dict__.update(state)

class VAEEncoder(eqx.Module): 
    #Maps from image to latents
    conv_layers: list
    mean_output: eqx.nn.Linear 
    log_var_output: eqx.nn.Linear
    def __init__(self, n_latent, input_size, k, key):
        keys = jax.random.split(key, 13) 
        self.conv_layers = [
            eqx.nn.Conv(num_spatial_dims=2, in_channels=3, out_channels=8*k, kernel_size=(2,2), stride=2,key=keys[0]),
            ConvResBlock(8*k,256,150, key=keys[1]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=8*k, out_channels=16*k, kernel_size=(2,2), stride=2,key=keys[2], padding=[(0,0),(1,1)]),
            ConvResBlock(16*k,128,76,key=keys[3]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=16*k, out_channels=32*k, kernel_size=(2,2), stride=2,key=keys[4]),
            ConvResBlock(32*k,64,38,key=keys[5]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=32*k, out_channels=64*k, kernel_size=(2,2), stride=2,key=keys[6], padding=[(0,0),(1,1)]),
            ConvResBlock(64*k,32,20,key=keys[7]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=64*k, out_channels=128*k, kernel_size=(2,2), stride=2,key=keys[8]),
            ConvResBlock(128*k,16,10,key=keys[8]),
            eqx.nn.Conv(num_spatial_dims=2, in_channels=128*k, out_channels=256*k, kernel_size=(2,2), stride=2,key=keys[9]),
            ConvResBlock(256*k,8,5, key=keys[10])
        ]
        self.mean_output = eqx.nn.Linear(10240*k, n_latent, key=keys[11]) #10240 (original value)
        self.log_var_output = eqx.nn.Linear(10240*k, n_latent, key=keys[12]) #10240 (original value)


    def __call__(self,x):
        h = (x/256)-0.5 #normalize pixel values (bs, 3, w, h)
        for layer in self.conv_layers:
            #print(str(h.shape) + " VAE Encoder Layer Shape")
            h = layer(h) #( bs, 256,8,5)
        mean = self.mean_output(h.reshape(-1))
        log_var = -(jnp.abs(self.log_var_output(h.reshape(-1)))+2)
        return mean, log_var

class VAEDecoder(eqx.Module): 
    #Maps from latents to images
    input_layer: eqx.nn.Linear
    conv_layers: list
    mean_output: eqx.nn.Conv
    log_var_output: eqx.nn.Conv
    def __init__(self, n_latent, input_size, k, key):
        keys = jax.random.split(key, 14) 
        self.input_layer = eqx.nn.Linear(n_latent, 10240*k, key=keys[11]) #10240
        self.conv_layers = [
            ConvResBlock(256*k,8,5,key=keys[10]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=256*k, out_channels=128*k, kernel_size=(2,2), stride=2,key=keys[9]),
            ConvResBlock(128*k,16,10,key=keys[8]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=128*k, out_channels=64*k, kernel_size=(2,2), stride=2,key=keys[8]),
            ConvResBlock(64*k,32,20,key=keys[7]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=64*k, out_channels=32*k, kernel_size=(2,2), stride=2,key=keys[6], padding=[(0,0),(1,1)]),
            ConvResBlock(32*k,64,38, key=keys[5]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=32*k, out_channels=16*k, kernel_size=(2,2), stride=2,key=keys[4]),
            ConvResBlock(16*k,128,76, key=keys[3]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=16*k, out_channels=8*k, kernel_size=(2,2), stride=2,key=keys[2], padding=[(0,0),(1,1)]),
            ConvResBlock(8*k,256,150,key=keys[1]),
            eqx.nn.ConvTranspose(num_spatial_dims=2, in_channels=8*k, out_channels=8*k, kernel_size=(2,2), stride=2,key=keys[0]),
        ]
        self.mean_output = eqx.nn.Conv(num_spatial_dims=2, in_channels=8*k, out_channels=3, kernel_size=(1,1), key=keys[12])
        self.log_var_output = eqx.nn.Conv(num_spatial_dims=2, in_channels=8*k, out_channels=3, kernel_size=(1,1), key=keys[13])

    def __call__(self,x):
        h = self.input_layer(x).reshape(-1,8,5) # reshapes data for decoder in_channels(256) 
        for layer in self.conv_layers:
            h = layer(h)
        mean = (self.mean_output(h)+0.5)*128 #map normalized values to pixel values
        log_var = self.log_var_output(h)+3
        return mean, log_var

