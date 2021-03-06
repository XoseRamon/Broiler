"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
import re
import warnings
import sys
import conv_net_classes as conv_classes
import timeit
from pprint import pprint
from pandas import *
import json
import cooking_preprocess_clean

warnings.filterwarnings("ignore")


# different non-linearities
def ReLU(x):
    y = T.maximum(0.0, x)
    return (y)


def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return (y)


def Tanh(x):
    y = T.tanh(x)
    return (y)


def Iden(x):
    y = x
    return (y)


def filter_set():
    with open('./test.json') as test_file:
        test_set = json.load(test_file)

    with open('./train.json') as train_file:
        train_set = json.load(train_file)

    ingredients_train = []
    for item in train_set:
        for ingredient in item["ingredients"]:
            ingredients_train.append(ingredient.replace(" ", "_"))
    filter = []
    for item in test_set:
        for ingredient in item["ingredients"]:
            if ingredient.ingredient.replace(" ", "_") not in ingredients_train:
                filter.append(ingredient.replace(" ", "_"))
    return filter


def train_conv_net(datasets,
                   cv,
                   submision,
                   U,
                   submision_complete,
                   img_w=300,
                   filter_hs=[3, 4, 5],
                   hidden_units=[100, 20],
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=10,
                   batch_size=50,
                   lr_decay=0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True):
    """
    Train a simple conv net
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper
    lr_decay = adadelta decay parameter
    """
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0]) - 1
    filter_w = img_w
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h - filter_h + 1, img_w - filter_w + 1))
    parameters = [("image shape", img_h, img_w), ("filter shape", filter_shapes), ("hidden_units", hidden_units),
                  ("dropout", dropout_rate), ("batch_size", batch_size), ("non_static", non_static),
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static)
        , ("sqr_norm_lim", sqr_norm_lim), ("shuffle_batch", shuffle_batch)]
    print parameters

    # define model architecture
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    Words = theano.shared(value=U.astype('float32'), name="Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w, dtype="float32")
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0, :], zero_vec_tensor))])
    layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((x.shape[0], 1, x.shape[1], Words.shape[1]))
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = conv_classes.LeNetConvPoolLayer(rng, input=layer0_input, image_shape=(batch_size, 1, img_h, img_w),
                                                     filter_shape=filter_shape, poolsize=pool_size,
                                                     non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs, 1)
    hidden_units[0] = feature_maps * len(filter_hs)
    classifier = conv_classes.MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations,
                                         dropout_rates=dropout_rate)

    # define parameters of the model and update functions using adadelta
    params = classifier.params
    for conv_layer in conv_layers:
        params += conv_layer.params
    if non_static:
        # if word vectors are allowed to change, add them as model parameters
        params += [Words]
    cost = classifier.negative_log_likelihood(y)
    dropout_cost = classifier.dropout_negative_log_likelihood(y)
    grad_updates = sgd_updates_adadelta(params, dropout_cost, lr_decay, 1e-6, sqr_norm_lim)

    # shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate
    # extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        train_set = np.asarray(np.random.permutation(datasets[0]), dtype="float32")
        extra_data = train_set[:extra_data_num]
        new_data = np.asarray(np.append(datasets[0], extra_data, axis=0), dtype="float32")
    else:
        new_data = np.asarray(datasets[0], dtype="float32")
    new_data = np.asarray(np.random.permutation(new_data), dtype="float32")
    n_batches = new_data.shape[0] / batch_size
    n_train_batches = int(np.round(n_batches * 0.9))
    # divide train set into train/val sets
    test_set_x = np.asarray(datasets[1][:, :img_h], "float32")
    test_set_y = np.asarray(datasets[1][:, -1], "int32")
    train_set = new_data[:n_train_batches * batch_size, :]
    val_set = new_data[n_train_batches * batch_size:, :]
    train_set_x, train_set_y = shared_dataset((train_set[:, :img_h], train_set[:, -1]))
    val_set_x, val_set_y = shared_dataset((val_set[:, :img_h], val_set[:, -1]))
    n_val_batches = n_batches - n_train_batches
    val_model = theano.function([index], classifier.errors(y),
                                givens={
                                    x: val_set_x[index * batch_size: (index + 1) * batch_size],
                                    y: val_set_y[index * batch_size: (index + 1) * batch_size]})

    # compile theano functions to get train/val/test errors
    test_model = theano.function([index], classifier.errors(y),
                                 givens={
                                     x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: train_set_y[index * batch_size: (index + 1) * batch_size]})
    train_model = theano.function([index], cost, updates=grad_updates,
                                  givens={
                                      x: train_set_x[index * batch_size:(index + 1) * batch_size],
                                      y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    test_pred_layers = []
    test_size = test_set_x.shape[0]
    test_layer0_input = Words[T.cast(x.flatten(), dtype="int32")].reshape((test_size, 1, img_h, Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    test_y_pred = classifier.predict(test_layer1_input)
    # hello_world_op = theano.printing.Print('hello world')
    # printed_test_y_pred = hello_world_op(test_y_pred)
    test_error = T.mean(T.neq(test_y_pred, y))
    get_predition = theano.function([x], test_y_pred, allow_input_downcast=True)
    test_model_all = theano.function([x, y], test_error)
    ###confusion matrix
    x2 = T.vector('x')
    classes = T.scalar('n_classes')
    onehot = T.eq(x2.dimshuffle(0, 'x'), T.arange(classes).dimshuffle('x', 0))
    oneHot = theano.function([x2, classes], onehot, allow_input_downcast=True)
    y2 = T.matrix('y')
    y_pred = T.matrix('y_pred')
    confMat = T.dot(y2.T, y_pred)
    confusionMatrix = theano.function(inputs=[y2, y_pred], outputs=confMat, allow_input_downcast=True)

    def confusion_matrix(x, y, n_class):
        return confusionMatrix(oneHot(x, n_class), oneHot(y, n_class))

    # start training over mini-batches
    print '... training'
    epoch = 0
    best_val_perf = 0
    val_perf = 0
    test_perf = 0
    cost_epoch = 0
    while epoch < n_epochs:
        start = timeit.default_timer()
        epoch += 1
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                cost_epoch = train_model(minibatch_index)
                # print "Loss value %f" % cost_epoch
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                cost_epoch = train_model(minibatch_index)
                # print "Loss value %f" % cost_epoch
                set_zero(zero_vec)
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_perf = 1 - np.mean(train_losses)
        val_losses = [val_model(i) for i in xrange(n_val_batches)]
        val_perf = 1 - np.mean(val_losses)
        stop = timeit.default_timer()
        print('epoch %i, train perf %f %%, val perf %f, time consumed %d' % (
            epoch, train_perf * 100., val_perf * 100., (stop - start)))
        if val_perf >= best_val_perf:
            best_val_perf = val_perf
            test_loss = test_model_all(test_set_x, test_set_y)
            predictions = get_predition(test_set_x)
            confusion_Matrix = DataFrame(confusion_matrix(predictions, test_set_y, 20))
            confusion_Matrix.to_csv(r'confusion_matrix' + str(epoch) + '.csv',
                                    index=True, sep=' ',
                                    mode='a')
            test_perf = 1 - test_loss

    remain = submision[1].shape[0]
    max_prediction_size = test_set_x.shape[0]
    counter = 0
    submision_prediction = []
    if (remain> max_prediction_size):
        while remain > 0:
            if (remain>=max_prediction_size):
                submision_prediction1 = get_predition(submision[0][counter:counter+max_prediction_size])
                submision_prediction = np.concatenate((submision_prediction, submision_prediction1), axis=0)
            else:
                fill = np.zeros(((max_prediction_size-remain), test_set_x.shape[1]))
                complete_vector = np.concatenate((submision[0][counter:counter+remain], fill), axis=0)
                submision_prediction1 = get_predition(complete_vector)
                submision_prediction = np.concatenate((submision_prediction, submision_prediction1[:remain]), axis=0)
            counter+=max_prediction_size
            remain-=max_prediction_size
    else:
        fill = np.zeros(((max_prediction_size-remain), test_set_x.shape[1]))
        complete_vector = np.concatenate((submision[0], fill), axis=0)
        submision_prediction = get_predition(complete_vector)
        submision_prediction = submision_prediction[:submision[0].shape[0]]
    submision_complete = np.vstack([submision_complete, submision_prediction])
    submision_prediction_df = DataFrame(np.transpose(submision_complete))
    submision_prediction_df.to_csv(r'submision_cv' + str(cv) + '.csv', header=False,
                                   index=False, sep=' ',
                                   mode='a')
    return test_perf, submision_complete


def shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32')


def sgd_updates_adadelta(params, cost, rho=0.95, epsilon=1e-6, norm_lim=9, word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name != 'Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def get_test_set(word_idx_map, max_l=51, filter_h=5):
    with open('../test.json') as test_file:
        test_set = json.load(test_file)
    test_set_sub = []
    test_set_sub_id = []
    for item in test_set:
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)
        for ingredient in item["ingredients"]:
            if ingredient in word_idx_map:
                x.append(word_idx_map[ingredient])
        while len(x) < max_l + 2 * pad:
            x.append(0)
        test_set_sub_id.append(item["id"])
        test_set_sub.append(x)
    return [np.array(test_set_sub, dtype="int"), np.array(test_set_sub_id, dtype="int")]


def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["recipe"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"] == cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]


if __name__ == "__main__":
    mode = sys.argv[1]
    word_vectors = sys.argv[2]
    embed_dim = sys.argv[3]
    minfreq = sys.argv[4]
    epochs = sys.argv[5]
    print('-' * 40)
    print "loading data..."
    data, W, W2, word_idx_map, vocab, cuisines = cooking_preprocess_clean.get_all_data(minfreq=int(minfreq),window=70, shuffle=1,embed_dim=int(embed_dim))
    print "data loaded!"
    if mode == "-nonstatic":
        print "model architecture: CNN-non-static"
        non_static = True
    elif mode == "-static":
        print "model architecture: CNN-static"
        non_static = False
    execfile("conv_net_classes.py")
    if word_vectors == "-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors == "-word2vec":
        print "using: word2vec vectors"
        U = W

    test_set_subm = get_test_set(word_idx_map, max_l=65, filter_h=5)
    submision_complete = test_set_subm[1]
    results = []
    r = range(0, 10)
    for i in r:
        datasets = make_idx_data_cv(data, word_idx_map, i, max_l=65, k=int(embed_dim), filter_h=5)
        perf = train_conv_net(datasets,
                              i,
                              test_set_subm,
                              U,
                              submision_complete,
                              lr_decay=0.95,
                              img_w=int(embed_dim),
                              filter_hs=[3, 4, 5],
                              conv_non_linear="relu",
                              hidden_units=[100, 20],
                              shuffle_batch=True,
                              n_epochs=int(epochs),
                              sqr_norm_lim=9,
                              non_static=non_static,
                              batch_size=50,
                              dropout_rate=[0.5])
        print "cv: " + str(i) + ", perf: " + str(perf[0])
        results.append(perf[0])
        submision_complete = perf[1]
    trans = np.transpose(submision_complete[1:, :])
    final_ensemble = []
    for i in range(trans.shape[0]):
        count = np.bincount(trans[i], minlength=20)
        final_ensemble.append(np.argmax(count))
    submision_complete = np.vstack([test_set_subm[1], np.array(cuisines)[final_ensemble]])
    submision_prediction_df = DataFrame(np.transpose(submision_complete))
    submision_prediction_df.to_csv(r'submision_final' + '.csv', header=False,
                                   index=False, sep=' ',
                                   mode='a')
    print str(np.mean(results))
    print('-' * 40)
