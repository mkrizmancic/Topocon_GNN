# Changelog
## 2024-06-04
1. **Dataset features:**
    - The dictionary with functions used to calculate the dataset features is now a class (not instance) variable. This makes is it possible to get the list of used features with `dataset.features`.
1. **Config for parameters:**
    - All the (hyper)parameters are stored in a config dictionary. This makes it easy to save them in Weights & Biases and adjust them for an automated sweep.
    - Information about the dataset is stored as a nested dictionary.
    - TODO: Find a way to store dataset information in a sweep.
2. **Dataset splits:**
    - The dataset is split into training and testing automatically using the `dataset_config["split"]` parameter (percentage for train, 0-1).
3. **DataLoader vs. Data object:**
    - The `DataLoader` class is can be used to automatically batch and iterate over the dataset.
    - If the whole dataset is small enough to fit into memory, the `Data` object can be used to load the whole dataset at once.
    - `Data` can be extracted from the `DataLoader` by taking the `next()` of the `DataLoader` iterator.
    - This is more efficient than loading the data from the `DataLoader` every epoch.
4. **Train and test functions:**
    - To work with the `DataLoader` and `Data` objects, the `train` and `test` functions have been updated.
    - `training_pass` / `testing_pass` performs a single pass over the training / testing batch (which can be one of the multiple batches or the whole dataset).
    - `do_train` / `do_test` performs the training / testing loop over all the batches.
5. **GNN wrapper:**
    - PyG comes with loads of pre-made GNNs. These models can be used for node-level tasks. For graph-level tasks, additional pooling layer is needed. The `GNNWrapper` class wraps the GNN model with a pooling layer.
    - `MyGCN` is a customizible implementation of the GCN model. Each hidden layer can have a different size. Specify the sizes as a list, e.g. `[10, 10, 5]`.
6. **New high-level functions:**
    - `main` loads the config, generates the model, optimzer, and loss function, starts the W&B run, and calls the training and evaluation procedures.
    - `train` prepares logging, starts timers, and runs the training loop.
    - `evaulate` calculates detailed metrics on the test set.
    - Using this functions helps limit the number of global variables.
7. **Two run types:**
    - When running the notebook or the script, `main` is given a hardcoded `global_config` dictionary with parameters and plots all the metrics.
    - When running a sweep, the `global_config` is automatically replaced by the sweep configuration given by the sweep agent and the console output is limited.
8. **Global variables:**
    - Watch out. There are still some global variables in the code. They should be documented where they are used. They will be removed in the future.


# Code snippets
### Initializing and printing weights and biases
There is no direct easy way to initialize and print GNNs weights and biases. It has to be done by accessing low-level elements of models. Below is an example for a simple GCN model.

```python
# Initialization
model = GNNWrapper("GCN", ...)
torch.nn.init.xavier_uniform_(model.gnn.convs[0].lin.weight)
torch.nn.init.zeros_(model.gnn.convs[0].bias)
torch.nn.init.ones_(model.gnn.lin.weight)

# Printing
def print_weights(model):
    def print_lin_layer(lin):
        weights = "\n\t".join(str(lin.weight).split("\n"))
        print(f"Weights:\n\t{weights}")
        if hasattr(lin, "bias") and lin.bias is not None:
            bias = "\n\t".join(str(lin.bias).split("\n"))
            print(f"Bias:\n\t{bias}")

    def print_gnn_layer(layer):
        print_lin_layer(layer.lin)
        bias = "\n\t".join(str(layer.bias).split("\n"))
        print(f"Bias:\n\t{bias}")
        print()

    for layer in model.mp_layers._modules.values():
        if isinstance(layer, GCNConv):
            print(layer)
            print_gnn_layer(layer)
    print(model.lin)
    print_lin_layer(model.lin)
```