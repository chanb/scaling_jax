import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, parentdir)

from flax import nnx

import jax.numpy as jnp

from src.constants import CONST_SAME_PADDING


def identity(x):
    return x


class MLP(nnx.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_layers,
        activation,
        *,
        rngs,
        use_bias = True,
        use_batch_norm = False,
        use_layer_norm = False,
        dropout_p = 0.0,
    ):
        all_layers = [in_dim] + hidden_layers

        layers = []
        for layer_i, (hidden_in, hidden_out) in enumerate(zip(
            all_layers[:-1],
            all_layers[1:],
        )):
            layers.append(
                nnx.Linear(hidden_in, hidden_out, use_bias=use_bias, rngs=rngs)
            )

            if use_layer_norm:
                layers.append(nnx.LayerNorm(hidden_out, rngs=rngs))
            if use_batch_norm:
                layers.append(nnx.BatchNorm(hidden_out, rngs=rngs))
            layers.append(nnx.Dropout(dropout_p, deterministic=dropout_p == 0.0, rngs=rngs))
            layers.append(activation)

        layers.append(
            nnx.Linear(all_layers[-1], out_dim, use_bias=use_bias, rngs=rngs)
        )

        self.mlp = layers

    def __call__(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class CNN(nnx.Module):
    def __init__(
        self,
        in_features,
        hidden_features,
        kernel_sizes,
        paddings,
        activation,
        *,
        rngs,
        use_bias = True,
        use_batch_norm = False,
        dropout_p = 0.0,
    ):
        all_layers = [in_features] + hidden_features[:-1]

        layers = []
        for layer_i, (hidden_in, hidden_out, kernel_size, padding) in enumerate(zip(
            all_layers[:-1],
            all_layers[1:],
            kernel_sizes,
            paddings,
        )):
            layers.append(
                nnx.Conv(
                    hidden_in,
                    hidden_out,
                    kernel_size,
                    padding=padding,
                    use_bias=use_bias,
                    rngs=rngs,
                )
            )

            if use_batch_norm:
                layers.append(nnx.BatchNorm(hidden_out, rngs=rngs))
            layers.append(
                nnx.Dropout(
                    dropout_p,
                    deterministic=dropout_p == 0.0,
                    rngs=rngs,
                )
            )
            layers.append(activation)

        layers.append(
            nnx.Linear(
                all_layers[-1],
                hidden_features[-1],
                use_bias=use_bias,
                rngs=rngs,
            )
        )

        self.conv = layers

    def __call__(self, x):
        for layer in self.conv:
            x = layer(x)
        return x


class ResNetV1Block(nnx.Module):
    def __init__(
        self,
        in_features,
        out_features,
        stride,
        use_projection,
        use_bottleneck,
        use_batch_norm,
        *,
        rngs,
    ):
        self.out_features = out_features
        self.stride = stride
        self.use_projection = use_projection
        self.use_batch_norm = use_batch_norm
        self.use_bottleneck = use_bottleneck
        if self.use_projection:
            self.projection = nnx.Conv(
                in_features,
                self.out_features,
                kernel_size=(1, 1),
                strides=self.stride,
                use_bias=False,
                padding=CONST_SAME_PADDING,
                rngs=rngs,
            )
            if self.use_batch_norm:
                self.projection_batchnorm = nnx.BatchNorm(
                    self.out_features,
                    momentum=0.9,
                    rngs=rngs,
                )

        
        conv_features = self.out_features
        conv_0_kernel = (3, 3)
        conv_0_stride = self.stride
        conv_1_stride = 1
        if self.use_bottleneck:
            conv_features = self.out_features // 4
            conv_0_kernel = (1, 1)
            conv_0_stride = 1
            conv_1_stride = self.stride

        conv_0 = nnx.Conv(
            self.out_features,
            conv_features,
            kernel_size=conv_0_kernel,
            strides=conv_0_stride,
            use_bias=False,
            padding=CONST_SAME_PADDING,
            rngs=rngs,
        )

        if self.use_batch_norm:
            batch_norm_0 = nnx.BatchNorm(
                conv_features,
                momentum=0.9,
                rngs=rngs,
            )

        conv_1 = nnx.Conv(
            conv_features,
            conv_features,
            kernel_size=(3, 3),
            strides=conv_1_stride,
            use_bias=False,
            padding=CONST_SAME_PADDING,
            rngs=rngs,
        )

        if self.use_batch_norm:
            batch_norm_1 = nnx.BatchNorm(
                conv_features,
                momentum=0.9,
                rngs=rngs,
            )

        if self.use_batch_norm:
            layers = [
                (conv_0, batch_norm_0),
                (conv_1, batch_norm_1),
            ]
        else:
            layers = [conv_0, conv_1]

        if self.use_bottleneck:
            conv_2 = nnx.Conv(
                conv_features,
                self.out_features,
                kernel_size=(1, 1),
                strides=1,
                use_bias=False,
                padding=CONST_SAME_PADDING,
                rngs=rngs,
            )
            if self.use_batch_norm:
                batch_norm_2 = nnx.BatchNorm(
                    self.out_features,
                    momentum=0.9,
                    scale_init=nnx.initializers.zeros_init(),
                    rngs=rngs,
                )
                layers.append((conv_2, batch_norm_2))
            else:
                layers.append(conv_2)
        self.layers = layers

    def __call__(self, x):
        out = shortcut = x

        if self.use_projection:
            shortcut = self.projection(shortcut)
            if self.use_batch_norm:
                shortcut = self.projection_batchnorm(shortcut)

        idx = -1

        if self.use_batch_norm:
            for idx, (conv_i, batch_norm_i) in enumerate(self.layers[:-1]):
                out = conv_i(out)
                out = batch_norm_i(out, eval)
                out = nnx.relu(out)
            out = self.layers[-1][0](out)
            out = self.layers[-1][1](out, eval)
        else:
            for idx, conv_i in enumerate(self.layers[:-1]):
                out = conv_i(out)
                out = nnx.relu(out)
            out = self.layers[-1](out)

        out = nnx.relu(out + shortcut)
        return out


class ResNetV1BlockGroup(nnx.Module):
    def __init__(
        self,
        num_blocks,
        in_features,
        out_features,
        stride,
        use_projection,
        use_bottleneck,
        use_batch_norm,
        *,
        rngs,
    ):
        blocks = []
        for block_i in range(num_blocks):
            blocks.append(
                ResNetV1Block(
                    in_features if block_i == 0 else out_features,
                    out_features,
                    1 if block_i else stride,
                    block_i == 0 and use_projection,
                    use_bottleneck,
                    use_batch_norm,
                    rngs=rngs,
                )
            )
        self.resnet_blocks = blocks

    def __call__(self, x):
        for block in self.resnet_blocks:
            x = block(x)
        return x


class ResNetV1(nnx.Module):
    def __init__(
        self,
        num_blocks_per_group,
        in_features,
        out_features_per_group,
        stride_per_group,
        use_projection_per_group,
        use_bottleneck,
        use_batch_norm,
        *,
        rngs,
    ):
        self.use_batch_norm = use_batch_norm
        self.init_conv = nnx.Conv(
            in_features,
            64,
            kernel_size=(7, 7),
            strides=2,
            use_bias=False,
            padding=CONST_SAME_PADDING,
            rngs=rngs,
        )

        if self.use_batch_norm:
            self.init_bn = nnx.BatchNorm(
                64,
                momentum=0.9,
                rngs=rngs,
            )

        groups = []
        for group_i, (
            curr_num_blocks,
            curr_out_features,
            curr_stride,
            curr_projection,
        ) in enumerate(zip(
            num_blocks_per_group,
            out_features_per_group,
            stride_per_group,
            use_projection_per_group,
        )):
            group_in_feature = (
                in_features
                if group_i == 0 else
                out_features_per_group[group_i - 1]
            )
            groups.append(
                ResNetV1BlockGroup(
                    curr_num_blocks,
                    group_in_feature,
                    curr_out_features,
                    curr_stride,
                    curr_projection,
                    use_bottleneck,
                    use_batch_norm,
                    rngs=rngs,
                )
            )

        self.resnet_groups = groups

    def __call__(self, x):
        x = self.init_conv(x)

        if self.use_batch_norm:
            x = self.init_bn(x)
        x = nnx.relu(x)
        x = nnx.max_pool(
            x,
            window_shape=(3, 3),
            strides=(2, 2),
            padding=CONST_SAME_PADDING,
        )

        for group in self.resnet_groups:
            x = group(x)

        return jnp.mean(x, axis=(-3, -2))
