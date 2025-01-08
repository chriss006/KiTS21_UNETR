import tensorflow as tf
from tensorflow.keras import losses
import tensorflow.keras.backend as K


# focal_loss
def focal_loss(alpha=0.25, gamma=2.0):

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())  # log(0)을 방지
        cross_entropy = -y_true * K.log(y_pred)  # Cross-Entropy 손실 계산
        weight = alpha * K.pow(1 - y_pred, gamma)  # Focal weight 계산
        return K.mean(weight * cross_entropy, axis=-1)  # 최종 손실 반환
    return loss

# dice_coefficient for focal_dice loss
def dice_coefficient(y_true, y_pred, smooth=1e-6):

    y_true_f = K.flatten(y_true)  # Flattening
    y_pred_f = K.flatten(y_pred)  # Flattening
    intersection = K.sum(y_true_f * y_pred_f)  # 교집합
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)  # Dice Coefficient 계산

# focal_dice loss
def focal_dice_loss(alpha=0.9, gamma=2.0):

    def loss(y_true, y_pred):
        focal = focal_loss(alpha, gamma)(y_true, y_pred)  # Focal Loss 계산
        dice = 1 - dice_coefficient(y_true, y_pred)  # Dice Loss 계산
        return focal + dice  # Focal Loss와 Dice Loss의 조합
    return loss


# To better train our segmentation networks, we can define new loss functions based on the [Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) which is a proper segmentation metric:
def dice_coeff(y_true, y_pred):
    """Define Dice coefficient.
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
       Return:
            score (tensor): Dice coefficient value
    """
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

# Dice coefficient loss (1 - Dice coefficient)
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# Loss function combining binary cross entropy and Dice loss
def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

# Weighted BCE+Dice
# Inspired by https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0
def weighted_bce_dice_loss(w_dice=0.5, w_bce=0.5):
    def loss(y_true, y_pred):
        return losses.binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice
    return loss

# Method to use PSNR as metric while training
def psnr( y_true, y_pred ):
  return tf.image.psnr( y_true, y_pred, max_val=1.0)

# Method to use SSIM as metric while training
def ssim ( y_true, y_pred ):
 return tf.image.ssim_multiscale( y_true, y_pred, max_val=1.0 )

# Evaluation metrics
def jaccard_index(y_true, y_pred, t=0.5):
    """Define Jaccard index.
       Args:
            y_true (tensor): ground truth masks.
            y_pred (tensor): predicted masks.
            t (float, optional): threshold to be applied.
       Return:
            jac (tensor): Jaccard index value
    """

    y_pred_ = tf.cast(y_pred > t, dtype=tf.int32)
    y_true = tf.cast(y_true, dtype=tf.int32)

    TP = tf.math.count_nonzero(y_pred_ * y_true)
    FP = tf.math.count_nonzero(y_pred_ * (y_true - 1))
    FN = tf.math.count_nonzero((y_pred_ - 1) * y_true)

    jac = tf.cond(tf.greater((TP + FP + FN), 0), lambda: TP / (TP + FP + FN),
                  lambda: tf.cast(0.000, dtype='float64'))

    return jac
