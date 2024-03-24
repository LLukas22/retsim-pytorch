
�Eroot"_tf_keras_network*�E{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "embedding_model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "retvec>ScaledSinusoidalPositionalEmbedding", "config": {"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "dtype": "float32", "hidden_size": 24, "min_timescale": 1.0, "max_timescale": 10000.0}, "name": "scaled_sinusoidal_positional_embedding", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "encoder_start", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder_start", "inbound_nodes": [[["scaled_sinusoidal_positional_embedding", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "swish"}, "name": "activation", "inbound_nodes": [[["encoder_start", 0, 0, {}]]]}, {"class_name": "retvec>GAU", "config": {"name": "gau", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "retvec>GAU", "config": {"name": "gau_1", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau_1", "inbound_nodes": [[["gau", 0, 0, {}]]]}, {"class_name": "Similarity>GeneralizedMeanPooling1D", "config": {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "p": 3.0, "data_format": "channels_last", "keepdims": false}, "name": "generalized_mean_pooling1d", "inbound_nodes": [[["gau_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["generalized_mean_pooling1d", 0, 0, {}]]]}, {"class_name": "Similarity>MetricEmbedding", "config": {"name": "similarity", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "similarity", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["similarity", 0, 0]]}, "name": "embedding_model", "inbound_nodes": [[["input_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["embedding_model", 1, 0]]}, "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 24]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512, 24]}, "float32", "input_1"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512, 24]}, "float32", "input_1"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Functional", "config": {"name": "embedding_model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "retvec>ScaledSinusoidalPositionalEmbedding", "config": {"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "dtype": "float32", "hidden_size": 24, "min_timescale": 1.0, "max_timescale": 10000.0}, "name": "scaled_sinusoidal_positional_embedding", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "encoder_start", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder_start", "inbound_nodes": [[["scaled_sinusoidal_positional_embedding", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "swish"}, "name": "activation", "inbound_nodes": [[["encoder_start", 0, 0, {}]]]}, {"class_name": "retvec>GAU", "config": {"name": "gau", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "retvec>GAU", "config": {"name": "gau_1", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau_1", "inbound_nodes": [[["gau", 0, 0, {}]]]}, {"class_name": "Similarity>GeneralizedMeanPooling1D", "config": {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "p": 3.0, "data_format": "channels_last", "keepdims": false}, "name": "generalized_mean_pooling1d", "inbound_nodes": [[["gau_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["generalized_mean_pooling1d", 0, 0, {}]]]}, {"class_name": "Similarity>MetricEmbedding", "config": {"name": "similarity", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "similarity", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["similarity", 0, 0]]}, "name": "embedding_model", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": [["input_1", 0, 0]], "output_layers": [["embedding_model", 1, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}2
�@root.layer_with_weights-0"_tf_keras_network*�?{"name": "embedding_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": false, "class_name": "Functional", "config": {"name": "embedding_model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "retvec>ScaledSinusoidalPositionalEmbedding", "config": {"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "dtype": "float32", "hidden_size": 24, "min_timescale": 1.0, "max_timescale": 10000.0}, "name": "scaled_sinusoidal_positional_embedding", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "encoder_start", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder_start", "inbound_nodes": [[["scaled_sinusoidal_positional_embedding", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "swish"}, "name": "activation", "inbound_nodes": [[["encoder_start", 0, 0, {}]]]}, {"class_name": "retvec>GAU", "config": {"name": "gau", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "retvec>GAU", "config": {"name": "gau_1", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau_1", "inbound_nodes": [[["gau", 0, 0, {}]]]}, {"class_name": "Similarity>GeneralizedMeanPooling1D", "config": {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "p": 3.0, "data_format": "channels_last", "keepdims": false}, "name": "generalized_mean_pooling1d", "inbound_nodes": [[["gau_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["generalized_mean_pooling1d", 0, 0, {}]]]}, {"class_name": "Similarity>MetricEmbedding", "config": {"name": "similarity", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "similarity", "inbound_nodes": [[["dropout_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["similarity", 0, 0]]}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 12, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 24]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512, 24]}, "float32", "input_2"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512, 24]}, "float32", "input_2"]}, "keras_version": "2.13.1", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "embedding_model", "trainable": true, "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 24]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "retvec>ScaledSinusoidalPositionalEmbedding", "config": {"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "dtype": "float32", "hidden_size": 24, "min_timescale": 1.0, "max_timescale": 10000.0}, "name": "scaled_sinusoidal_positional_embedding", "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 2}, {"class_name": "Dense", "config": {"name": "encoder_start", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "encoder_start", "inbound_nodes": [[["scaled_sinusoidal_positional_embedding", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "swish"}, "name": "activation", "inbound_nodes": [[["encoder_start", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "retvec>GAU", "config": {"name": "gau", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau", "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "retvec>GAU", "config": {"name": "gau_1", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "name": "gau_1", "inbound_nodes": [[["gau", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Similarity>GeneralizedMeanPooling1D", "config": {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "p": 3.0, "data_format": "channels_last", "keepdims": false}, "name": "generalized_mean_pooling1d", "inbound_nodes": [[["gau_1", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "name": "dropout_4", "inbound_nodes": [[["generalized_mean_pooling1d", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Similarity>MetricEmbedding", "config": {"name": "similarity", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "similarity", "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 11}], "input_layers": [["input_2", 0, 0]], "output_layers": [["similarity", 0, 0]]}}}2
�
�.root.layer_with_weights-0.layer_with_weights-0"_tf_keras_layer*�{"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>ScaledSinusoidalPositionalEmbedding", "config": {"name": "scaled_sinusoidal_positional_embedding", "trainable": true, "dtype": "float32", "hidden_size": 24, "min_timescale": 1.0, "max_timescale": 10000.0}, "inbound_nodes": [[["input_2", 0, 0, {}]]], "shared_object_id": 2}2
�.root.layer_with_weights-0.layer_with_weights-1"_tf_keras_layer*�{"name": "encoder_start", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "encoder_start", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 3}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 4}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["scaled_sinusoidal_positional_embedding", 0, 0, {}]]], "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 24}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 24]}}2
�!root.layer_with_weights-0.layer-3"_tf_keras_layer*�{"name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "swish"}, "inbound_nodes": [[["encoder_start", 0, 0, {}]]], "shared_object_id": 6}2
�.root.layer_with_weights-0.layer_with_weights-2"_tf_keras_layer*�{"name": "gau", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>GAU", "config": {"name": "gau", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "inbound_nodes": [[["activation", 0, 0, {}]]], "shared_object_id": 7}2
�.root.layer_with_weights-0.layer_with_weights-3"_tf_keras_layer*�{"name": "gau_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>GAU", "config": {"name": "gau_1", "trainable": true, "dtype": "float32", "dim": 256, "max_len": 512, "shared_dim": 128, "expansion_factor": 1, "activation": "swish", "attention_activation": "sqrrelu", "norm_type": "scaled", "position_encoding_type": "rope", "dropout_rate": 0.0, "attention_dropout_rate": 0.0, "spatial_dropout_rate": 0.0, "epsilon": 1e-06}, "inbound_nodes": [[["gau", 0, 0, {}]]], "shared_object_id": 8}2
�!root.layer_with_weights-0.layer-6"_tf_keras_layer*�{"name": "generalized_mean_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Similarity>GeneralizedMeanPooling1D", "config": {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "p": 3.0, "data_format": "channels_last", "keepdims": false}, "inbound_nodes": [[["gau_1", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 17}}2
�!root.layer_with_weights-0.layer-7"_tf_keras_layer*�{"name": "dropout_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_4", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "inbound_nodes": [[["generalized_mean_pooling1d", 0, 0, {}]]], "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}2
�.root.layer_with_weights-0.layer_with_weights-4"_tf_keras_layer*�{"name": "similarity", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Similarity>MetricEmbedding", "config": {"name": "similarity", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dropout_4", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 18}}2
�Z3root.layer_with_weights-0.layer_with_weights-2.norm"_tf_keras_layer*�{"name": "scaled_norm", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>ScaledNorm", "config": {"name": "scaled_norm", "trainable": true, "dtype": "float32", "begin_axis": -1, "epsilon": 1e-06}, "shared_object_id": 19}2
�[4root.layer_with_weights-0.layer_with_weights-2.proj1"_tf_keras_layer*�{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 640, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 22, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 23}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2
�\4root.layer_with_weights-0.layer_with_weights-2.proj2"_tf_keras_layer*�{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2
�]7root.layer_with_weights-0.layer_with_weights-2.dropout1"_tf_keras_layer*�{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 27, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2
�^7root.layer_with_weights-0.layer_with_weights-2.dropout2"_tf_keras_layer*�{"name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 28, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 640]}}2
�_Iroot.layer_with_weights-0.layer_with_weights-2.attention_activation_layer"_tf_keras_layer*�{"name": "sqr_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>SqrReLU", "config": {"name": "sqr_re_lu", "trainable": true, "dtype": "float32"}, "shared_object_id": 29}2
�g3root.layer_with_weights-0.layer_with_weights-3.norm"_tf_keras_layer*�{"name": "scaled_norm_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>ScaledNorm", "config": {"name": "scaled_norm_1", "trainable": true, "dtype": "float32", "begin_axis": -1, "epsilon": 1e-06}, "shared_object_id": 30}2
�h4root.layer_with_weights-0.layer_with_weights-3.proj1"_tf_keras_layer*�{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 640, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 32}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 34}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2
�i4root.layer_with_weights-0.layer_with_weights-3.proj2"_tf_keras_layer*�{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 256, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.02, "seed": null}, "shared_object_id": 31}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2
�j7root.layer_with_weights-0.layer_with_weights-3.dropout1"_tf_keras_layer*�{"name": "dropout_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_2", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 38, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2
�k7root.layer_with_weights-0.layer_with_weights-3.dropout2"_tf_keras_layer*�{"name": "dropout_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "Dropout", "config": {"name": "dropout_3", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}, "shared_object_id": 39, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 640]}}2
�lIroot.layer_with_weights-0.layer_with_weights-3.attention_activation_layer"_tf_keras_layer*�{"name": "sqr_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "retvec>SqrReLU", "config": {"name": "sqr_re_lu_1", "trainable": true, "dtype": "float32"}, "shared_object_id": 40}2
�t%root.layer_with_weights-0.layer-6.gap"_tf_keras_layer*�{"name": "generalized_mean_pooling1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "preserve_input_structure_in_config": false, "autocast": true, "class_name": "GlobalAveragePooling1D", "config": {"name": "generalized_mean_pooling1d", "trainable": true, "dtype": "float32", "data_format": "channels_last", "keepdims": false}, "shared_object_id": 41, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 256]}}2