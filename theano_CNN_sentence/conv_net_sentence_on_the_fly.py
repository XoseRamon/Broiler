"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""
import json
import timeit
import warnings
from collections import OrderedDict
from pandas import *
import os
import sys
import theano
import theano.tensor as T

import conv_net_classes as conv_classes
import cooking
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
    with open('../test.json') as test_file:
        test_set = json.load(test_file)

    with open('../train.json') as train_file:
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
                  ("learn_decay", lr_decay), ("conv_non_linear", conv_non_linear),
                  ("sqr_norm_lim", sqr_norm_lim), ("shuffle_batch", shuffle_batch)]
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
    test_y_pred_prob = classifier.predict_p(test_layer1_input)
    # hello_world_op = theano.printing.Print('hello world')
    # printed_test_y_pred = hello_world_op(test_y_pred)
    test_error = T.mean(T.neq(test_y_pred, y))
    get_predition = theano.function([x], test_y_pred, allow_input_downcast=True)
    get_predition_prob = theano.function([x], test_y_pred_prob, allow_input_downcast=True)
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
            test_perf = 1 - test_loss

    remain = submision[1].shape[0]
    max_prediction_size = test_set_x.shape[0]
    counter = 0
    submision_prediction = np.empty(0, dtype=int)
    if (remain> max_prediction_size):
        while remain > 0:
            if (remain>=max_prediction_size):
                submision_prediction1 = get_predition(submision[0][counter:counter+max_prediction_size])
                submision_prediction = np.concatenate((submision_prediction, submision_prediction1), axis=0)
                submision_prediction1_prob = get_predition_prob(submision[0][counter:counter+max_prediction_size])
                if (counter==0):
                    submision_prediction_prob = submision_prediction1_prob
                else:
                    submision_prediction_prob = np.concatenate((submision_prediction_prob, submision_prediction1_prob), axis=0)
            else:
                fill = np.zeros(((max_prediction_size-remain), test_set_x.shape[1]))
                complete_vector = np.concatenate((submision[0][counter:counter+remain], fill), axis=0)
                submision_prediction1 = get_predition(complete_vector)
                submision_prediction = np.concatenate((submision_prediction, submision_prediction1[:remain]), axis=0)
                #getting probs
                submision_prediction_prob1 = get_predition_prob(complete_vector)
                submision_prediction_prob = np.concatenate((submision_prediction_prob, submision_prediction_prob1[:remain][:]),axis=0)
            counter+=max_prediction_size
            remain-=max_prediction_size
    else:
        fill = np.zeros(((max_prediction_size-remain), test_set_x.shape[1]))
        complete_vector = np.concatenate((submision[0], fill), axis=0)
        submision_prediction = get_predition(complete_vector)
        submision_prediction = submision_prediction[:submision[0].shape[0]]
        #getting probs
        submision_prediction_prob = get_predition_prob(complete_vector)
        submision_prediction_prob = submision_prediction_prob[:submision[0].shape[0]]


    submision_complete = np.vstack([submision_complete, submision_prediction])
    submision_prediction_df = DataFrame(np.transpose(submision_complete))
    submision_prediction_df.to_csv(os.path.join('./','submision_cv' + str(1) + '.csv'), header=False,
                                   index=False, sep=' ',
                                   mode='w')
    return test_perf, submision_complete, submision_prediction, submision_prediction_prob


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


def pad_with_zeroes(sent, max_l=51, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    for word in sent:
        x.append(word)
    while len(x) < max_l + 2 * pad:
        x.append(0)
    return x


def get_test_set(X_predict, X_id, max_l=51, filter_h=5):
    test_set_sub = []
    for i in range(len(X_predict)):
        sent = pad_with_zeroes(X_predict[i], max_l, filter_h)
        test_set_sub.append(sent)

    return [np.array(test_set_sub, dtype="int"), np.array(X_id, dtype="int")]

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        i += 1
    return W


def make_idx_data_cv(X_train, y_train, X_test, y_test, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for i in range(len(y_train)):
        sent = pad_with_zeroes(X_train[i], max_l, filter_h)
        sent.append(y_train[i])
        train.append(sent)

    for i in range(len(y_test)):
        sent = pad_with_zeroes(X_test[i], max_l, filter_h)
        sent.append(y_test[i])
        test.append(sent)

    train = np.array(train, dtype="int")
    test = np.array(test, dtype="int")
    return [train, test]

#Note: this method ensures each part of the dataset is trained and in the dataset
#It depends on the order of the dataset, so it is sub-optimal
def make_cv_set(X, labels, cur_cv, cv, max_l=51, k=300, filter_h=5):

    train_sent, test_sent=[], []

    for i in range(len(X)):
        sent = pad_with_zeroes(X[i], max_l, filter_h)
        sent.append(labels[i])
        if (i%cv==cur_cv):
            test_sent.append(sent)
        else:
            train_sent.append(sent)

    train_sent = np.asarray(train_sent, dtype=int)
    test_sent = np.asarray(test_sent, dtype=int)

    print len(train_sent)+" training instances"
    print len(test_sent)+" test instances"

    return [train_sent, test_sent]



def random_we(vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    word_vecs = {}
    for word in vocab:
        if word not in word_vecs:  # and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
    return word_vecs


if __name__ == "__main__":
    print "Reading parameters"
    embed_dim = np.cast[int](sys.argv[1])
    epochs = sys.argv[2]
    # 0 word2vec 1 random
    randVecs = np.cast[int](sys.argv[3])

    X, labels = [],[]
    X_train, y_train = [],[]
    X_test, y_test = [],[]
    X_prediction, X_id = [],[]
    vocab, cuisines = [],[]

    number_cv = 10

    print("Loading data...")
    if (number_cv>0):
        (X, labels), (X_prediction, X_id), vocab, cuisines = cooking.load_full_data(shuffle=2)
        print(len(X), 'dataset sequences')
    else:
        (X_train, y_train), (X_test, y_test), (X_prediction, X_id), vocab, cuisines = cooking.load_data(test_split=0.1, shuffle=2)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        nb_classes = np.max(y_train) + 1

    max_features = len(vocab)
    print("Pad sequences (samples x time)")


    print "data loaded!"
    print('-' * 40)
    rand_vecs =[]
    if (randVecs):
        print "Using random vectors"
        rand_vecs = random_we(vocab, k=int(embed_dim))
        U = get_W(rand_vecs, embed_dim)
    else :
        # Already with randoms for OOV
        print "Training word2vec"
        word2vec = cooking.generateWord2Vec(vocab, size=embed_dim, window=70, use_classes=True)
        U = get_W(word2vec, embed_dim)
        print "trained"

    execfile("conv_net_classes.py")

    # filter = filter_set()
    test_set_subm = get_test_set(X_prediction, X_id, max_l=65, filter_h=5)
    results = []
    submision_complete = np.array(X_id)

    for cur_cv in range(0, number_cv):
        datasets = make_cv_set(X=X, labels=labels, cur_cv=cur_cv, cv=number_cv, k=int(embed_dim), filter_h=5)
        #make_idx_data_cv(X_train, y_train, X_test, y_test, max_l=65, k=int(embed_dim), filter_h=5)

        perf = train_conv_net(datasets,
                          test_set_subm,
                          U,
                          submision_complete,
                          lr_decay=0.95,
                          img_w=int(embed_dim),
                          filter_hs=[3, 7, 11, 15],
                          conv_non_linear="relu",
                          hidden_units=[100, 20],
                          shuffle_batch=True,
                          n_epochs=int(epochs),
                          sqr_norm_lim=9,
                          non_static=True,
                          batch_size=50,
                          dropout_rate=[0.5])
        print "cv: "+ cur_cv + ", perf: " + str(perf[0])
        print "probs shape"
        print perf[3].shape
        results.append(perf[0])

    trans = np.transpose(submision_complete)[:, 1:]
    final_ensemble = []
    for i in range(submision_complete.shape[0]):
         print trans[i]
         count = np.bincount(trans[i], minlength=20)
         final_ensemble.append(np.argmax(count))
    count = np.apply_along_axis(np.bincount, axis=1, arr=np.transpose(submision_complete)[:, 1:], minlength=20)
    final_ensemble = np.apply_along_axis(np.argmax, 1, count)


    submision_complete = np.vstack([np.array(X_id), np.array(cuisines)[final_ensemble]])
    submision_prediction_df = DataFrame(np.transpose(submision_complete))
    submision_prediction_df.to_csv(os.path.join('./','submision_final' + '.csv'), header=False,
                                   index=False, sep=' ',
                                   mode='w')
    print str(np.mean(results))
    print('-' * 40)
