"""
================================================================================
Attention Res-UNet with Atrous Spatial Pyramid Pooling (ASPP) & Advanced Losses
Designed for State-of-the-Art Satellite Forest Segmentation & Change Detection.
================================================================================
"""

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, backend as K
    HAS_TF = True
except ImportError:
    HAS_TF = False


if HAS_TF:
    # -------------------------------------------------------------------------
    # RESIDUAL CONVOLUTION BLOCK
    # -------------------------------------------------------------------------
    def residual_conv_block(inputs, filters, dropout_rate=0.1):
        """
        Two 3x3 Conv layers with BatchNormalization, ReLU, and Residual Shortcut.
        Preserves fine canopy boundaries, micro-clearings, and avoids gradient vanishing.
        """
        shortcut = layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(inputs)
        shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        if dropout_rate > 0:
            x = layers.SpatialDropout2D(dropout_rate)(x)

        x = layers.Conv2D(filters, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Add()([shortcut, x])
        x = layers.Activation("relu")(x)
        return x

    # -------------------------------------------------------------------------
    # ATROUS SPATIAL PYRAMID POOLING (ASPP)
    # -------------------------------------------------------------------------
    def aspp_block(inputs, filters):
        """
        Multi-scale dilated convolutions capture forest features at multiple contextual scales:
        individual tree canopies, narrow logging roads, and continuous deep forest reserves.
        """
        shape = inputs.shape

        # 1. 1x1 conv
        b0 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(inputs)
        b0 = layers.BatchNormalization()(b0)
        b0 = layers.Activation("relu")(b0)

        # 2. 3x3 atrous conv, rate=6
        b1 = layers.Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(inputs)
        b1 = layers.BatchNormalization()(b1)
        b1 = layers.Activation("relu")(b1)

        # 3. 3x3 atrous conv, rate=12
        b2 = layers.Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(inputs)
        b2 = layers.BatchNormalization()(b2)
        b2 = layers.Activation("relu")(b2)

        # 4. 3x3 atrous conv, rate=18
        b3 = layers.Conv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False)(inputs)
        b3 = layers.BatchNormalization()(b3)
        b3 = layers.Activation("relu")(b3)

        # 5. Global Image Pooling branch
        b4 = layers.GlobalAveragePooling2D(keepdims=True)(inputs)
        b4 = layers.Conv2D(filters, 1, padding="same", use_bias=False)(b4)
        b4 = layers.BatchNormalization()(b4)
        b4 = layers.Activation("relu")(b4)
        b4 = layers.Resizing(shape[1], shape[2])(b4)

        # Concatenate & Project
        out = layers.Concatenate()([b0, b1, b2, b3, b4])
        out = layers.Conv2D(filters, 1, padding="same", use_bias=False)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Activation("relu")(out)
        out = layers.Dropout(0.3)(out)
        return out

    # -------------------------------------------------------------------------
    # ATTENTION GATE (AG)
    # -------------------------------------------------------------------------
    def attention_gate(skip_features, gating_signal, inter_filters):
        """
        Additive Attention Gate:
        Selectively emphasizes canopy features from the skip connection
        while suppressing background soil, rock reflections, and cloud shadows.
        """
        theta_x = layers.Conv2D(inter_filters, kernel_size=1, strides=1, padding="same", use_bias=False)(skip_features)
        phi_g = layers.Conv2D(inter_filters, kernel_size=1, strides=1, padding="same", use_bias=False)(gating_signal)

        concat = layers.Add()([theta_x, phi_g])
        act = layers.Activation("relu")(concat)
        psi = layers.Conv2D(1, kernel_size=1, strides=1, padding="same", activation="sigmoid")(act)
        return layers.Multiply()([skip_features, psi])

    # -------------------------------------------------------------------------
    # ATTENTION RES-UNET WITH ASPP
    # -------------------------------------------------------------------------
    def build_attention_res_unet(input_shape=(128, 128, 4), base_filters=32, use_aspp=True):
        """
        Builds the state-of-the-art Attention Residual U-Net with optional ASPP bottleneck.
        """
        inputs = layers.Input(shape=input_shape, name="satellite_input")

        # Encoder Path
        c1 = residual_conv_block(inputs, base_filters)
        p1 = layers.MaxPooling2D((2, 2))(c1)

        c2 = residual_conv_block(p1, base_filters * 2)
        p2 = layers.MaxPooling2D((2, 2))(c2)

        c3 = residual_conv_block(p2, base_filters * 4)
        p3 = layers.MaxPooling2D((2, 2))(c3)

        c4 = residual_conv_block(p3, base_filters * 8)
        p4 = layers.MaxPooling2D((2, 2))(c4)

        # Bottleneck (with Multi-Scale ASPP)
        if use_aspp:
            bottleneck = aspp_block(p4, base_filters * 16)
        else:
            bottleneck = residual_conv_block(p4, base_filters * 16, dropout_rate=0.3)

        # Decoder Path with Attention Gates
        u4 = layers.Conv2DTranspose(base_filters * 8, kernel_size=2, strides=2, padding="same")(bottleneck)
        ag4 = attention_gate(skip_features=c4, gating_signal=u4, inter_filters=base_filters * 4)
        m4 = layers.Concatenate()([u4, ag4])
        d4 = residual_conv_block(m4, base_filters * 8)

        u3 = layers.Conv2DTranspose(base_filters * 4, kernel_size=2, strides=2, padding="same")(d4)
        ag3 = attention_gate(skip_features=c3, gating_signal=u3, inter_filters=base_filters * 2)
        m3 = layers.Concatenate()([u3, ag3])
        d3 = residual_conv_block(m3, base_filters * 4)

        u2 = layers.Conv2DTranspose(base_filters * 2, kernel_size=2, strides=2, padding="same")(d3)
        ag2 = attention_gate(skip_features=c2, gating_signal=u2, inter_filters=base_filters)
        m2 = layers.Concatenate()([u2, ag2])
        d2 = residual_conv_block(m2, base_filters * 2)

        u1 = layers.Conv2DTranspose(base_filters, kernel_size=2, strides=2, padding="same")(d2)
        ag1 = attention_gate(skip_features=c1, gating_signal=u1, inter_filters=base_filters // 2)
        m1 = layers.Concatenate()([u1, ag1])
        d1 = residual_conv_block(m1, base_filters)

        # Logits output
        logits = layers.Conv2D(1, kernel_size=1, activation=None, name="tree_cover_logits")(d1)

        model = models.Model(inputs=inputs, outputs=logits, name="Forest_Attention_Res_UNet_ASPP")
        return model

    # -------------------------------------------------------------------------
    # ADVANCED SEGMENTATION LOSSES (Dice + Focal Tversky)
    # -------------------------------------------------------------------------
    def dice_loss(y_true, logits, smooth=1e-6):
        y_true = tf.cast(y_true, tf.float32)
        probs = tf.sigmoid(logits)
        y_true_f = tf.reshape(y_true, [-1])
        probs_f = tf.reshape(probs, [-1])
        intersection = tf.reduce_sum(y_true_f * probs_f)
        total = tf.reduce_sum(y_true_f) + tf.reduce_sum(probs_f)
        dice = (2.0 * intersection + smooth) / (total + smooth)
        return 1.0 - dice

    def focal_tversky_loss(alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        """
        Focal Tversky Loss:
        Heavily penalizes False Negatives (beta=0.7) to prevent missing subtle deforestation tracks,
        while gamma modulates focus onto hard edge pixels.
        """
        def loss_fn(y_true, logits):
            y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            probs = tf.sigmoid(logits)
            probs_f = tf.reshape(probs, [-1])

            tp = tf.reduce_sum(y_true_f * probs_f)
            fp = tf.reduce_sum((1.0 - y_true_f) * probs_f)
            fn = tf.reduce_sum(y_true_f * (1.0 - probs_f))

            tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
            return tf.pow(1.0 - tversky_index, gamma)

        return loss_fn

    def dice_bce_combo_loss(positive_weight=1.0, negative_weight=1.0, dice_ratio=0.5):
        """Compound loss combining weighted BCE with Soft Dice Loss."""
        def loss_fn(y_true, logits):
            y_true_f = tf.cast(y_true, tf.float32)
            bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true_f, logits=logits)
            weights = y_true_f * positive_weight + (1.0 - y_true_f) * negative_weight
            weighted_bce = tf.reduce_mean(bce * weights)
            d_loss = dice_loss(y_true_f, logits)
            return (1.0 - dice_ratio) * weighted_bce + dice_ratio * d_loss

        return loss_fn
