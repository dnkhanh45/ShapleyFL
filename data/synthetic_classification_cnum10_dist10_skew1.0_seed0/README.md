# Data description for classification task:
## Number of clients: `10`
## Input size: `60`
## Number of class: `10`
# Model description:
## Linear layer:
### Weight: `shape = 60 x 10`
### Bias: `shape = 10`
# Datastore structure (file data.json): (key: value)
## `store`: `XY` (type of data storage)
## `client_names`: `['Client00', 'Client01', 'Client02', 'Client03', 'Client04', 'Client05', 'Client06', 'Client07', 'Client08', 'Client09']` (names of clients)
## `dtest`: server test dataset
### `x`: 2d matrix (`shape = test_size x 60`)
### `y`: 1d matrix (`shape = test_size`)
## `Client{id}`: For each client in `client_names`
### `dtrain`: client's train dataset, same structure with server test dataset
### `dvalid`: client's validate dataset, same structure with server test dataset
### `optimal`: 2d matrix (`shape = 61 x 10`) - client's optimal solution includes:
#### Linear layer's weight: first `60` vector with `shape = 10`
#### Linear layer's bias: last vector with `shape = 10`