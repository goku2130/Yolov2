import glob
import re
import os
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Concatenate, concatenate, Dropout, LeakyReLU, Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda

print('Tensorflow version : {}'.format(tf.__version__))

# Model parameters

LABELS           = ('sugarbeet', 'weed')
IMAGE_H, IMAGE_W = 512, 512
GRID_H,  GRID_W  = 16, 16 # GRID size = IMAGE size / 32
BOX              = 5
CLASS            = len(LABELS)
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

TRAIN_BATCH_SIZE = 10
VAL_BATCH_SIZE   = 10
EPOCHS           = 10

LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1

MAX_ANNOT        = 0

class DarkNet19(keras.Model):
    """
    Implementation of the DarkNet19 architecture in Tensorflow 2.0
    """

    def __init__(self,
                 labels,
                 image_h, image_w,
                 grid_h, grid_w,
                 box,
                 confidence_score_threshold, iou_threshold,
                 anchors,
                 training_batch, validation_batch,
                 epochs,
                 lambda_noobject, lambda_object,
                 lambda_class, lambda_coord,
                 max_annotations,
                 name,
                 **kwargs
                 ):

        super(DarkNet19, self).__init__(name = name, **kwargs)

        # loss parameters
        self.lambda_coord = lambda_coord
        self.lambda_class = lambda_class
        self.lambda_object = lambda_object
        self.lambda_noobject = lambda_noobject

        #training parameters
        self.epochs = epochs
        self.validation_batch = validation_batch
        self.training_batch = training_batch

        # boudning box parameters
        self.anchors = anchors
        self.iou_threshold = iou_threshold
        self.confidence_score_threshold = confidence_score_threshold
        self.max_annotations = max_annotations
        self.box = box
        self.grid_w = grid_w
        self.grid_h = grid_h

        # image parameters
        self.image_w = image_w
        self.image_h = image_h
        self.labels = labels

        #self.model = None
        # Model specifications for DarkNet-19

        # input layer
        input_image = tf.keras.layers.Input((self.image_h, self.image_w, 3), dtype='float32')

        # Layer 1
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
        x = BatchNormalization(name='norm_2')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 3
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
        x = BatchNormalization(name='norm_3')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 4
        x = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
        x = BatchNormalization(name='norm_4')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 5
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
        x = BatchNormalization(name='norm_5')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 7
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
        x = BatchNormalization(name='norm_7')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 8
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
        x = BatchNormalization(name='norm_8')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 9
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
        x = BatchNormalization(name='norm_9')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 10
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
        x = BatchNormalization(name='norm_10')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 11
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
        x = BatchNormalization(name='norm_11')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 12
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
        x = BatchNormalization(name='norm_12')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 13
        x = Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
        x = BatchNormalization(name='norm_13')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # skip connection to the 21st conv layer
        skip_connection = x
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 14
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
        x = BatchNormalization(name='norm_14')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 15
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
        x = BatchNormalization(name='norm_15')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 16
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
        x = BatchNormalization(name='norm_16')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 17
        x = Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
        x = BatchNormalization(name='norm_17')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 18
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
        x = BatchNormalization(name='norm_18')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 19
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
        x = BatchNormalization(name='norm_19')(x)
        x = LeakyReLU(alpha=0.1)(x)

        # Layer 20
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
        x = BatchNormalization(name='norm_20')(x)
        x = LeakyReLU(alpha=0.1)(x)

        class SpaceToDepth(keras.layers.Layer):

            def __init__(self, block_size, **kwargs):
                self.block_size = block_size
                super(SpaceToDepth, self).__init__(**kwargs)

            def call(self, inputs):
                x = inputs
                batch, height, width, depth = K.int_shape(x)
                batch = -1
                reduced_height = height // self.block_size
                reduced_width = width // self.block_size
                y = K.reshape(x, (batch, reduced_height, self.block_size,
                                  reduced_width, self.block_size, depth))
                z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
                t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size ** 2))
                return t

            def compute_output_shape(self, input_shape):
                shape = (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                         input_shape[3] * self.block_size ** 2)
                return tf.TensorShape(shape)

        # Layer 21 (Skip connection)
        skip_connection = Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
            skip_connection)
        skip_connection = BatchNormalization(name='norm_21')(skip_connection)
        skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
        skip_connection = SpaceToDepth(block_size=2)(skip_connection)
        x = concatenate([skip_connection, x])

        # Layer 22
        x = Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
        x = BatchNormalization(name='norm_22')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(0.3)(x)  # add dropout

        # Layer 23
        x = Conv2D(self.box * (4 + 1 + len(self.labels)), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
        output = Reshape((self.grid_w, self.grid_h, self.box, 4 + 1 + len(self.labels)))(x)

        self.model = keras.models.Model(input_image, output)

        #print(self.model.summary())

    def iou(self, x1, y1, w1, h1, x2, y2, w2, h2):
        '''
        Calculate IOU between box1 and box2

        Parameters
        ----------
        - x, y : box center coords
        - w : box width
        - h : box height

        Returns
        -------
        - IOU
        '''
        xmin1 = x1 - 0.5 * w1
        xmax1 = x1 + 0.5 * w1
        ymin1 = y1 - 0.5 * h1
        ymax1 = y1 + 0.5 * h1
        xmin2 = x2 - 0.5 * w2
        xmax2 = x2 + 0.5 * w2
        ymin2 = y2 - 0.5 * h2
        ymax2 = y2 + 0.5 * h2

        interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
        intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)

        # intersection area
        inter = interx * intery

        # union area
        union = w1 * h1 + w2 * h2 - inter

        iou = inter / (union + 1e-6)
        return iou

    def yolov2_loss(self, detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid, y_pred, info = False):
        '''
        Calculate YOLO V2 loss from prediction (y_pred) and ground truth tensors (detector_mask,
        matching_true_boxes, class_one_hot, true_boxes_grid,)

        Parameters
        ----------
        - detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
            1 if bounding box detected by grid cell, else 0
        - matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
            Contains adjusted coords of bounding box in YOLO format
        - class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
            One hot representation of bounding box label
        - true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
            true_boxes_grid format : x, y, w, h, c (coords unit : grid cell)
        - y_pred : prediction from model. tensor (shape : batch_size, GRID_W, GRID_H, anchors count, (5 + labels count)
        - info : boolean. True to get some infox about loss value

        Returns
        -------
        - loss : scalar
        - sub_loss : sub loss list : coords loss, class loss and conf loss : scalar
        '''

        # anchors tensor
        anchors = np.array(ANCHORS)
        anchors = anchors.reshape(len(anchors) // 2, 2)

        # grid coords tensor
        coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
        coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
        coords = tf.tile(tf.concat([coord_x, coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])

        # coordinate loss
        pred_xy = K.sigmoid(y_pred[:, :, :, :, 0:2])  # adjust coords between 0 and 1
        pred_xy = (pred_xy + coords)  # add cell coord for comparaison with ground truth. New coords in grid cell unit
        pred_wh = K.exp(y_pred[:, :, :, :,
                        2:4]) * anchors  # adjust width and height for comparaison with ground truth. New coords in grid cell unit
        # pred_wh = (pred_wh * anchors) # unit : grid cell
        nb_detector_mask = K.sum(tf.cast(detector_mask > 0.0, tf.float32))
        xy_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(matching_true_boxes[..., :2] - pred_xy)) / (
                nb_detector_mask + 1e-6)  # Non /2
        wh_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(K.sqrt(matching_true_boxes[..., 2:4]) -
                                                                K.sqrt(pred_wh))) / (nb_detector_mask + 1e-6)
        coord_loss = xy_loss + wh_loss

        # class loss
        pred_box_class = y_pred[..., 5:]
        true_box_class = tf.argmax(class_one_hot, -1)
        # class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        class_loss = K.sparse_categorical_crossentropy(target=true_box_class, output=pred_box_class, from_logits=True)
        class_loss = K.expand_dims(class_loss, -1) * detector_mask
        class_loss = LAMBDA_CLASS * K.sum(class_loss) / (nb_detector_mask + 1e-6)

        # confidence loss
        pred_conf = K.sigmoid(y_pred[..., 4:5])
        # for each detector : iou between prediction and ground truth
        x1 = matching_true_boxes[..., 0]
        y1 = matching_true_boxes[..., 1]
        w1 = matching_true_boxes[..., 2]
        h1 = matching_true_boxes[..., 3]
        x2 = pred_xy[..., 0]
        y2 = pred_xy[..., 1]
        w2 = pred_wh[..., 0]
        h2 = pred_wh[..., 1]
        ious = self.iou(x1, y1, w1, h1, x2, y2, w2, h2)
        ious = K.expand_dims(ious, -1)

        # for each detector : best ious between prediction and true_boxes (every bounding box of image)
        pred_xy = K.expand_dims(pred_xy, 4)  # shape : m, GRID_W, GRID_H, BOX, 1, 2
        pred_wh = K.expand_dims(pred_wh, 4)
        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half
        true_boxe_shape = K.int_shape(true_boxes_grid)
        true_boxes_grid = K.reshape(true_boxes_grid,
                                    [true_boxe_shape[0], 1, 1, 1, true_boxe_shape[1], true_boxe_shape[2]])
        true_xy = true_boxes_grid[..., 0:2]
        true_wh = true_boxes_grid[..., 2:4]
        true_wh_half = true_wh * 0.5
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half
        intersect_mins = K.maximum(pred_mins, true_mins)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
        intersect_maxes = K.minimum(pred_maxes, true_maxes)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, 1, 1
        true_areas = true_wh[..., 0] * true_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
        best_ious = K.expand_dims(best_ious)  # shape : m, GRID_W, GRID_H, BOX, 1

        # no object confidence loss
        no_object_detection = K.cast(best_ious < 0.6, K.dtype(best_ious))
        noobj_mask = no_object_detection * (1 - detector_mask)
        nb_noobj_mask = K.sum(tf.cast(noobj_mask > 0.0, tf.float32))

        noobject_loss = LAMBDA_NOOBJECT * K.sum(noobj_mask * K.square(-pred_conf)) / (nb_noobj_mask + 1e-6)
        # object confidence loss
        object_loss = LAMBDA_OBJECT * K.sum(detector_mask * K.square(ious - pred_conf)) / (nb_detector_mask + 1e-6)
        # total confidence loss
        conf_loss = noobject_loss + object_loss

        # total loss
        loss = conf_loss + class_loss + coord_loss
        sub_loss = [conf_loss, class_loss, coord_loss]

        #     # 'triple' mask
        #     true_box_conf_IOU = ious * detector_mask
        #     conf_mask = noobj_mask * LAMBDA_NOOBJECT
        #     conf_mask = conf_mask + detector_mask * LAMBDA_OBJECT
        #     nb_conf_box  = K.sum(tf.to_float(conf_mask  > 0.0))
        #     conf_loss = K.sum(K.square(true_box_conf_IOU - pred_conf) * conf_mask)  / (nb_conf_box  + 1e-6)

        #     # total loss
        #     loss = conf_loss /2. + class_loss + coord_loss /2.
        #     sub_loss = [conf_loss /2., class_loss, coord_loss /2.]

        if info:
            print('conf_loss   : {:.4f}'.format(conf_loss))
            print('class_loss  : {:.4f}'.format(class_loss))
            print('coord_loss  : {:.4f}'.format(coord_loss))
            print('    xy_loss : {:.4f}'.format(xy_loss))
            print('    wh_loss : {:.4f}'.format(wh_loss))
            print('--------------------')
            print('total loss  : {:.4f}'.format(loss))

            # display masks for each anchors
            for i in range(len(anchors)):
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
                f.tight_layout()
                f.suptitle('MASKS FOR ANCHOR {} :'.format(anchors[i, ...]))

                ax1.matshow((K.sum(detector_mask[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
                ax1.set_title(
                    'detector_mask, count : {}'.format(K.sum(tf.cast(detector_mask[0, :, :, i] > 0., tf.int32))))
                ax1.xaxis.set_ticks_position('bottom')

                ax2.matshow((K.sum(no_object_detection[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
                ax2.set_title('no_object_detection mask')
                ax2.xaxis.set_ticks_position('bottom')

                ax3.matshow((K.sum(noobj_mask[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
                ax3.set_title('noobj_mask')
                ax3.xaxis.set_ticks_position('bottom')
                plt.show()
        return loss, sub_loss

    def grad(self, model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes, training=True):
        with tf.GradientTape() as tape:
            y_pred = model(img, training)
            loss, sub_loss = self.yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes, y_pred)
        return loss, sub_loss, tape.gradient(loss, model.trainable_variables)

    # save weights
    def save_best_weights(self, model, name, val_loss_avg):
        # delete existing weights file
        files = glob.glob(os.path.join('weights/', name + '*'))
        for file in files:
            os.remove(file)
        # create new weights file
        name = name + '_' + str(val_loss_avg) + '.h5'
        path_name = os.path.join('weights/', name)
        model.save_weights(path_name)

    # log (tensorboard)
    def log_loss(self, loss, val_loss, step):
        tf.summary.scalar('loss', loss, step)
        tf.summary.scalar('val_loss', val_loss, step)
# model = DarkNet19(labels= LABELS,
#                 image_h = IMAGE_H,
#                 image_w = IMAGE_W,
#                 grid_h = GRID_H,
#                 grid_w = GRID_W,
#                 box = BOX,
#
#                 confidence_score_threshold = SCORE_THRESHOLD,
#                 iou_threshold = IOU_THRESHOLD,
#                 anchors = ANCHORS,
#
#                 training_batch = TRAIN_BATCH_SIZE,
#                 validation_batch = VAL_BATCH_SIZE,
#                 epochs = EPOCHS,
#
#                 lambda_noobject = LAMBDA_NOOBJECT,
#                 lambda_object = LAMBDA_OBJECT,
#                 lambda_class = LAMBDA_CLASS,
#                 lambda_coord = LAMBDA_COORD,
#
#                 max_annotations = MAX_ANNOT,
#                 name = "Yolov2")



