
from utils import *
import tensorflow.keras.backend as K
from model import DarkNet19

LABELS           = ('sugarbeet', 'weed')
IMAGE_H, IMAGE_W = 512, 512
GRID_H,  GRID_W  = 16, 16 # GRID size = IMAGE size / 32
BOX              = 5 # anchor box
CLASS            = len(LABELS)
SCORE_THRESHOLD  = 0.5
IOU_THRESHOLD    = 0.45
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

TRAIN_BATCH_SIZE = 3
VAL_BATCH_SIZE   = 3
EPOCHS           = 100

LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1

MAX_ANNOT        = 0


# training
def train(epochs, mymodel, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val, train_name='train'):
    '''
    Train YOLO model for n epochs.
    Eval loss on training and validation dataset.
    Log training loss and validation loss for tensorboard.
    Save best weights during training (according to validation loss).

    Parameters
    ----------
    - epochs : integer, number of epochs to train the model.
    - model : YOLO model.
    - train_dataset : YOLO ground truth and image generator from training dataset.
    - val_dataset : YOLO ground truth and image generator from validation dataset.
    - steps_per_epoch_train : integer, number of batch to complete one epoch for train_dataset.
    - steps_per_epoch_val : integer, number of batch to complete one epoch for val_dataset.
    - train_name : string, training name used to log loss and save weights.

    Notes :
    - train_dataset and val_dataset generate YOLO ground truth tensors : detector_mask,
      matching_true_boxes, class_one_hot, true_boxes_grid. Shape of these tensors (batch size, tensor shape).
    - steps per epoch = number of images in dataset // batch size of dataset

    Returns
    -------
    - loss history : [train_loss_history, val_loss_history] : list of average loss for each epoch.
    '''
    num_epochs = epochs
    steps_per_epoch_train = steps_per_epoch_train
    steps_per_epoch_val = steps_per_epoch_val
    train_loss_history = []
    val_loss_history = []
    best_val_loss = 1e6

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # log (tensorboard)
    summary_writer = tf.summary.create_file_writer(os.path.join('logs/', train_name), flush_millis=20000)
    summary_writer.set_as_default()

    # training
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_val_loss = []
        epoch_val_sub_loss = []
        print('Epoch {} :'.format(epoch))
        # train
        for batch_idx in range(steps_per_epoch_train):
            img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(train_dataset)
            loss, _, grads = mymodel.grad(mymodel.model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes)
            optimizer.apply_gradients(zip(grads, mymodel.model.trainable_variables))
            epoch_loss.append(loss)
            print('-', end='')
        print(' | ', end='')
        # val
        for batch_idx in range(steps_per_epoch_val):
            img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(val_dataset)
            loss, sub_loss, grads = mymodel.grad(mymodel.model, img, detector_mask, matching_true_boxes, class_one_hot, true_boxes,
                                         training=False)
            epoch_val_loss.append(loss)
            epoch_val_sub_loss.append(sub_loss)
            print('-', end='')

        loss_avg = np.mean(np.array(epoch_loss))
        val_loss_avg = np.mean(np.array(epoch_val_loss))
        sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
        train_loss_history.append(loss_avg)
        val_loss_history.append(val_loss_avg)

        # log
        mymodel.log_loss(loss_avg, val_loss_avg, epoch)

        # save
        if val_loss_avg < best_val_loss:
            mymodel.save_best_weights(mymodel.model, train_name, val_loss_avg)
            best_val_loss = val_loss_avg

        print(' loss = {:.4f}, val_loss = {:.4f} (conf={:.4f}, class={:.4f}, coords={:.4f})'.format(
            loss_avg, val_loss_avg, sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2]))

    return [train_loss_history, val_loss_history]


if __name__ == '__main__':

    train_image_folder = 'data/train/image/'
    train_annot_folder = 'data/train/annotation/'
    val_image_folder = 'data/val/image/'
    val_annot_folder = 'data/val/annotation/'
    pretrained_weights_folder = 'pretrained_weights/'

    weight_reader = WeightReader(pretrained_weights_folder + 'yolov2.weights')
    weight_reader.reset()
    nb_conv = 23

    mymodel = DarkNet19(labels=LABELS,
                      image_h=IMAGE_H,
                      image_w=IMAGE_W,
                      grid_h=GRID_H,
                      grid_w=GRID_W,
                      box=BOX,

                      confidence_score_threshold=SCORE_THRESHOLD,
                      iou_threshold=IOU_THRESHOLD,
                      anchors=ANCHORS,

                      training_batch=TRAIN_BATCH_SIZE,
                      validation_batch=VAL_BATCH_SIZE,
                      epochs=EPOCHS,

                      lambda_noobject=LAMBDA_NOOBJECT,
                      lambda_object=LAMBDA_OBJECT,
                      lambda_class=LAMBDA_CLASS,
                      lambda_coord=LAMBDA_COORD,

                      max_annotations=MAX_ANNOT,
                      name="Yolov2")

    # load pretrained weights from file
    for i in range(1, nb_conv + 1):
        conv_layer = mymodel.model.get_layer('conv_' + str(i))
        conv_layer.trainable = True

        if i < nb_conv:
            norm_layer = mymodel.model.get_layer('norm_' + str(i))
            norm_layer.trainable = True

            size = np.prod(norm_layer.get_weights()[0].shape)
            # set weights to batch norm layers using gamma beta mean and variance
            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    # random initialize last convolutional layer
    layer = mymodel.model.layers[-2]
    layer.trainable = True

    weights = layer.get_weights()

    new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
    new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)

    layer.set_weights([new_kernel, new_bias])

    train_dataset = None
    train_dataset = get_dataset('train' , train_image_folder, train_annot_folder, LABELS, IMAGE_H, IMAGE_W, TRAIN_BATCH_SIZE)

    val_dataset = None
    val_dataset = get_dataset('validation' , val_image_folder, val_annot_folder, LABELS, IMAGE_H, IMAGE_W, VAL_BATCH_SIZE)

    #test_sample(train_dataset)

    aug_train_dataset = augmentation_generator(train_dataset, image_w = IMAGE_W, image_h = IMAGE_H)
    #test_sample(aug_train_dataset)

    train_gen = ground_truth_generator(aug_train_dataset, ANCHORS, IMAGE_W, IMAGE_H, GRID_W, GRID_H, CLASS)
    val_gen = ground_truth_generator(val_dataset, ANCHORS, IMAGE_W, IMAGE_H, GRID_W, GRID_H, CLASS)

    img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(train_gen)

    # y
    matching_true_boxes = matching_true_boxes[0, ...]
    detector_mask = detector_mask[0, ...]
    class_one_hot = class_one_hot[0, ...]
    y = K.concatenate((matching_true_boxes[..., 0:4], detector_mask, class_one_hot), axis=-1)

    # y_hat
    y_hat = mymodel.model.predict_on_batch(img)[0, ...]

    # img
    img = img[0, ...]

    # display prediction (Yolo Confidence value)
    #plt.figure(figsize=(2, 2))
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    ax1.imshow(img)
    ax1.set_title('Image')

    ax2.matshow((K.sum(y[:, :, :, 4], axis=2)))  # YOLO Confidence value
    ax2.set_title('Ground truth')
    ax2.xaxis.set_ticks_position('bottom')

    ax3.matshow(K.sum(y_hat[:, :, :, 4], axis=2))  # YOLO Confidence value
    ax3.set_title('Prediction')
    ax3.xaxis.set_ticks_position('bottom')

    f.tight_layout()
    plt.show()

    # get batch
    img, detector_mask, matching_true_boxes, class_one_hot, true_boxe_grid = next(train_gen)

    # first image in batch
    img = img[0:1]
    detector_mask = detector_mask[0:1]
    matching_true_boxes = matching_true_boxes[0:1]
    class_one_hot = class_one_hot[0:1]
    true_boxe_grid = true_boxe_grid[0:1]

    # predict
    y_pred = mymodel.model.predict_on_batch(img)

    # plot img, ground truth and prediction
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.imshow(img[0, ...])
    ax1.set_title('Image')
    ax2.matshow(K.sum(detector_mask[0, :, :, :, 0], axis=2))  # YOLO Confidence value
    ax2.set_title('Ground truth, count : {}'.format(K.sum(tf.cast(detector_mask > 0., tf.int32))))
    ax2.xaxis.set_ticks_position('bottom')
    ax3.matshow(K.sum(y_pred[0, :, :, :, 4], axis=2))  # YOLO Confidence value
    ax3.set_title('Prediction')
    ax3.xaxis.set_ticks_position('bottom')
    f.tight_layout()

    # loss info
    loss, sub_loss = mymodel.yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxe_grid, y_pred, info=True)

    results = train(EPOCHS, mymodel, train_gen, val_gen, 10, 2, 'training_1')

    plt.plot(results[0])
    plt.plot(results[1])
    plt.show()