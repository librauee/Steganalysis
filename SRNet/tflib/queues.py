import tensorflow as tf
import threading

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
		_input = generator(0,1).__next__()
		if type(_input) is not list:
			raise ValueError("generator doesn't return" \
							 "a list: %r" %  type(_input))
		input_batch_size = _input[0].shape[0]
		if not all(_input[i].shape[0] == input_batch_size for i in range(len(_input))):
			raise ValueError("all the inputs doesn't have the same batch size,"\
							 "the batch sizes are: %s" % [_input[i].shape[0] for i in range(len(_input))])
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
		sess.run(self.close_queue_op)

def queueSelection(runners, sel, batch_size):
	selection_queue = tf.FIFOQueue.from_list(sel, [r.queue for r in runners])
	return selection_queue.dequeue_many(batch_size)