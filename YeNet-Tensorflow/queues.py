import numpy as np
import tensorflow as tf
import threading
import h5py
import functools

def hdf5baseGen(filepath, thread_idx, n_threads):
    with h5py.File(filepath, 'r') as f:
        keys = f.keys()
        nb_data = f[keys[0]].shape[0]
        idx = thread_idx
        while True:
            yield [np.expand_dims(f[key][idx], 0) for key in keys]
            idx = (idx + n_threads) % nb_data

class GeneratorRunner():
    """
    This class manage a multithreaded queue filled with a generator
    """
    def __init__(self, generator, capacity):
        """
        inputs: generator feeding the data, must have thread_idx 
            as parameter (but the parameter may be not used)
        """
        self.generator = generator
        _input = generator(0,1).next()
        if type(_input) is not list:
            raise ValueError("generator doesn't return" \
                             "a list: %r" %  type(_input))
        input_batch_size = _input[0].shape[0]
        if not all(_input[i].shape[0] == input_batch_size for i in range(len(_input))):
            raise ValueError("all the inputs doesn't have " + \
            				 "the same batch size," \
                             "the batch sizes are: %s" % [_input[i].shape[0] \
                             for i in range(len(_input))])
        self.data = []
        self.dtypes = []
        self.shapes = []
        for i in range(len(_input)):
            self.shapes.append(_input[i].shape[1:])
            self.dtypes.append(_input[i].dtype)
            self.data.append(tf.placeholder(dtype=self.dtypes[i], \
                                shape=(input_batch_size,) + self.shapes[i]))
        self.queue = tf.FIFOQueue(capacity, shapes=self.shapes, \
                                dtypes=self.dtypes)
        self.enqueue_op = self.queue.enqueue_many(self.data)
        self.close_queue_op = self.queue.close(cancel_pending_enqueues=True)
        
    def get_batched_inputs(self, batch_size):
        """
        Return tensors containing a batch of generated data
        """
        batch = self.queue.dequeue_many(batch_size)
        return batch
    
    def thread_main(self, sess, thread_idx=0, n_threads=1):
        try:
            for data in self.generator(thread_idx, n_threads):
                sess.run(self.enqueue_op, feed_dict={i: d \
                            for i, d in zip(self.data, data)})
                if self.stop_threads:
                    return
        except RuntimeError:
            pass
        except tf.errors.CancelledError:
            pass
    
    def start_threads(self, sess, n_threads=1):
        self.stop_threads = False
        self.threads = []
        for n in range(n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, n, n_threads))
            t.daemon = True
            t.start()
            self.threads.append(t)
        return self.threads

    def stop_runner(self, sess):
        self.stop_threads = True
        # j = 0
        # while np.any([t.is_alive() for t in self.threads]):
        #     j += 1
        #     if j % 100 = 0:
        #         print [t.is_alive() for t in self.threads]
        sess.run(self.close_queue_op)

def queueSelection(runners, sel, batch_size):
    selection_queue = tf.FIFOQueue.from_list(sel, [r.queue for r in runners])
    return selection_queue.dequeue_many(batch_size)

def doubleQueue(runner1, runner2, is_runner1, batch_size1, batch_size2):
    return tf.cond(is_runner1, lambda: runner1.queue.dequeue_many(batch_size1), \
                lambda: runner2.queue.dequeue_many(batch_size2))

if __name__ == '__main__':
    def randomGen(img_size, enqueue_batch_size, thread_idx, n_threads):
        while True:
            batch_of_1_channel_imgs = np.random.rand(enqueue_batch_size, \
                                                     img_size, img_size, 1)
            batch_of_labels = np.random.randint(0,11,enqueue_batch_size)
            return [batch_of_1_channel_imgs, batch_of_labels]

    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 10
    train_runner = GeneratorRunner(functool.partial(randomGen, \
                                   (128, 10)), TRAIN_BATCH_SIZE * 10)
    valid_runner = GeneratorRunner(functool.partial(randomGen, \
                                   (128, 10)), VALID_BATCH_SIZE * 10)
    is_training = tf.Variable(True)
    batch_size = tf.Variable(TRAIN_BATCH_SIZE)
    enable_training_op = tf.group(tf.assign(is_training, True), \
                                  tf.assign(batch_size, TRAIN_BATCH_SIZE))
    disable_training_op = tf.group(tf.assign(is_training, False), \
                                  tf.assign(batch_size, VALID_BATCH_SIZE))
    img_batch, label_batch = queueSelection([valid_runner, train_runner], \
                                             tf.cast(is_training, tf.int32), \
                                             batch_size)
    # img_batch, label_batch = doubleQueue(train_runner, valid_runner, \
    #                                      is_training, TRAIN_BATCH_SIZE, \
    #                                      VALID_BATCH_SIZE)
