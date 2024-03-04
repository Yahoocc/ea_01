import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from db import MysqlDatabases

from model.run_classifier import ChnSentiCorpProcessor, SimplifyWBProcessor, InputExample, convert_examples_to_features, input_fn_builder, model_fn_builder, file_based_convert_examples_to_features, file_based_input_fn_builder
from model import modeling
from model import optimization
from model import tokenization

import tensorflow.compat.v1 as tf
import numpy as np
import os

db = MysqlDatabases()

class SettingFrame(tk.Frame):
	def __init__(self, root):
		super().__init__(root)
		self.model = tk.IntVar()
		tk.Label(self, text='').grid(row=0, column=1)
		tk.Label(self, text='模型选择：').grid(row=1, column=1)
		tk.Radiobutton(self, text='模型1(2种情感)', value=1, variable=self.model).grid(row=1, column=2)
		tk.Radiobutton(self, text='模型2(4种情感)', value=2, variable=self.model).grid(row=1, column=3)
		self.model.set(1)

class AboutFrame(tk.Frame):
	def __init__(self, root):
		super().__init__(root)
		tk.Label(self, text='').pack()
		tk.Label(self, text='版权声明', font=('', 20, 'bold')).pack()
		tk.Label(self, text='关于作品：本作品由tkinter制作').pack()
		tk.Label(self, text='关于作者: Ycc').pack()
		tk.Label(self, text='版权所有：杨成昌').pack()

class RecognizeFrame(tk.Frame):
	def __init__(self, root, setting_frame):
		super().__init__(root)
		self.setting_frame = setting_frame
		tk.Label(self, text='输入文本：', font=('', 20, 'bold')).grid(row=0, column=1, pady=10)
		self.text = tk.Text(self, width='50', height='10')
		self.text.grid(row=1, column=1)
		tk.Button(self, text='开始识别', command=self.recognize).grid(row=2, column=1, pady=10)
		tk.Label(self, text='识别结果：', font=('', 20, 'bold')).grid(row=3, column=1, pady=10)
		self.emotion_result = tk.Label(self, text='')
		self.emotion_result.grid(row=4, column=1)
	
	def recognize(self):
		self.model_id = self.setting_frame.model.get()
		if self.model_id == 1:
			self.model_address = "./model/chnsenticorp_output/model.ckpt-7765"
			self.model_dir = "./model/chnsenticorp_output"
		elif self.model_id == 2:
			self.model_address = "./model/simplifywb_output/model.ckpt-60000"
			self.model_dir = "./model/simplifywb_output"

		self.sequence = self.text.get('0.0', 'end')
		tf.logging.set_verbosity(tf.logging.INFO)
		tokenization.validate_case_matches_checkpoint(True, self.model_address)
		bert_config = modeling.BertConfig.from_json_file("./model/bert_config.json")
		if 128 > bert_config.max_position_embeddings:
			raise ValueError(
	        	"Cannot use sequence length %d because the BERT model "
	        	"was only trained up to sequence length %d" %
	        	(128, bert_config.max_position_embeddings))
		tf.gfile.MakeDirs("./model/tmp/output")
		if self.model_id == 1:
			processor = ChnSentiCorpProcessor()
		elif self.model_id == 2:
			processor = SimplifyWBProcessor()
		else:
			processor = None
		label_list = processor.get_labels()
		tokenizer = tokenization.FullTokenizer(
      		vocab_file="./model/vocab.txt", do_lower_case=True)

		tpu_cluster_resolver = None
		is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
		run_config = tf.estimator.tpu.RunConfig(
			cluster=tpu_cluster_resolver,
			master=None,
			model_dir=self.model_dir,
			save_checkpoints_steps=None,
			tpu_config=tf.estimator.tpu.TPUConfig(
				iterations_per_loop=1000,
				num_shards=8,
				per_host_input_for_training=is_per_host))

		train_examples = None
		num_train_steps = None
		num_warmup_steps = None

		# predict_examples = processor.get_test_examples(FLAGS.data_dir)

		model_fn = model_fn_builder(
  			bert_config=bert_config,
  			num_labels=len(label_list),
  			init_checkpoint=self.model_address,
  			learning_rate=2e-5,
  			num_train_steps=num_train_steps,
  			num_warmup_steps=num_warmup_steps,
  			use_tpu=False,
  			use_one_hot_embeddings=False)

		estimator = tf.estimator.tpu.TPUEstimator(
  			use_tpu=False,
  			model_fn=model_fn,
  			config=run_config,
  			train_batch_size=32,
  			eval_batch_size=32,
  			predict_batch_size=32)

		predict_examples = []
		text_a = tokenization.convert_to_unicode(self.sequence)
		label = "0"
		predict_examples.append(InputExample(text_a=text_a, text_b=None, label=label))

		num_actual_predict_examples = len(predict_examples)

		predict_features = convert_examples_to_features(predict_examples, label_list, 128, tokenizer)

  		# tf.logging.info("***** Running prediction*****")
  		# tf.logging.info("  Num examples = %d (%d actual, %d padding)",
        #           len(predict_examples), num_actual_predict_examples,
        #           len(predict_examples) - num_actual_predict_examples)
  		# tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

		predict_drop_remainder = False

		predict_input_fn = input_fn_builder(
  			features=predict_features,
  			seq_length=128,
  			is_training=False,
  			drop_remainder=predict_drop_remainder)

		results = estimator.predict(input_fn=predict_input_fn)
		for result in results:
			r = result["probabilities"]
			max_index = np.argmax(r)
			if self.model_id == 1:
				self.show_result_two(max_index)
			elif self.model_id == 2:
				self.show_result_four(max_index)

	def record_info(self, str):
		record = {"sequence": self.sequence, "emotion": str}
		db.insert(record)

	def show_result_two(self, max_index):
		if max_index == 1:
			self.emotion_result['text']="😊"
			self.record_info("😊")
		else:
			self.emotion_result['text']="😭"
			self.record_info("😭")

	def show_result_four(self, max_index):
		if max_index == 0:
			self.emotion_result['text']="喜悦"
			self.record_info("喜悦")
		elif max_index == 1:
			self.emotion_result['text']="愤怒"
			self.record_info("愤怒")
		elif max_index == 2:
			self.emotion_result['text']="厌恶"
			self.record_info("厌恶")
		elif max_index == 3:
			self.emotion_result['text']="低落"
			self.record_info("低落")

class RecordFrame(tk.Frame):
	def __init__(self, root):
		super().__init__(root)	
		tk.Label(self, text='识别记录', font=('', 20, 'bold')).pack()
		tk.Label(self, text='').pack()
		self.table_view = tk.Frame()
		self.table_view.pack()
		self.create_page()
		tk.Button(self, text='刷新记录', command=self.show_data_frame).pack()
		self.show_data_frame()

	def create_page(self):
		columns = ("sequences", "emotions")
		columns_values = ("文本序列", "情感")
		self.tree_view = ttk.Treeview(self, show='headings', columns=columns)
		self.tree_view.column('sequences', width=500, anchor='center')
		self.tree_view.column('emotions', width=80, anchor='center')
		self.tree_view.heading('sequences', text='文本序列')
		self.tree_view.heading('emotions', text='情感')
		self.tree_view.pack(fill=tk.BOTH, expand=True)

	def show_data_frame(self):
		for _ in map(self.tree_view.delete, self.tree_view.get_children('')):
			pass
		records = db.all()
		index = 0
		for record in records:
			self.tree_view.insert('', index + 1, values=(
				record['sequence'], record['emotion']
			))

class BatchFrame(tk.Frame):
	def __init__(self, root, setting_frame):
		super().__init__(root)
		self.setting_frame = setting_frame
		self.path = tk.StringVar()
		self.tip_text = tk.StringVar()
		tk.Label(self, text = "").grid(row = 0, column = 0, pady = 10)
		tk.Label(self, text = "目标路径:").grid(row = 1, column = 0)
		tk.Entry(self, textvariable = self.path, width = '30').grid(row = 1, column = 1)
		tk.Button(self, text = "路径选择", command = self.selectPath).grid(row = 1, column = 2)
		tk.Button(self, text="开始识别", command=self.recognize).grid(row=2, column=1, pady=10)
		tk.Label(self, textvariable = self.tip_text).grid(row = 3, column = 1)
		
	def selectPath(self):
		self.path_ = filedialog.askopenfilename()
		#通过replace函数替换绝对文件地址中的/来使文件可被程序读取 
    	#注意：\\转义后为\，所以\\\\转义后为\\
		# path_=path_.replace("/","\\\\")
		self.path.set(self.path_)

	def recognize(self):
		self.model_id = self.setting_frame.model.get()
		if self.model_id == 1:
			self.model_address = "./model/chnsenticorp_output/model.ckpt-7765"
			self.model_dir = "./model/chnsenticorp_output"
		elif self.model_id == 2:
			self.model_address = "./model/simplifywb_output/model.ckpt-60000"
			self.model_dir = "./model/simplifywb_output"

		tf.logging.set_verbosity(tf.logging.INFO)
		tokenization.validate_case_matches_checkpoint(True, self.model_address)
		bert_config = modeling.BertConfig.from_json_file("./model/bert_config.json")
		if 128 > bert_config.max_position_embeddings:
			raise ValueError(
				"Cannot use sequence length %d because the BERT model "
				"was only trained up to sequence length %d" %
				(128, bert_config.max_position_embeddings))
		tf.gfile.MakeDirs("./model/tmp/output")
		if self.model_id == 1:
			processor = ChnSentiCorpProcessor()
		elif self.model_id == 2:
			processor = SimplifyWBProcessor()
		else:
			processor = None
		label_list = processor.get_labels()
		tokenizer = tokenization.FullTokenizer(
			vocab_file="./model/vocab.txt", do_lower_case=True)
		tpu_cluster_resolver = None
		is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
		run_config = tf.estimator.tpu.RunConfig(
    		cluster=tpu_cluster_resolver,
    		master=None,
    		model_dir=self.model_dir,
    		save_checkpoints_steps=None,
    		tpu_config=tf.estimator.tpu.TPUConfig(
    			iterations_per_loop=1000,
    			num_shards=8,
    			per_host_input_for_training=is_per_host))

		train_examples = None
		num_train_steps = None
		num_warmup_steps = None

		model_fn = model_fn_builder(
    		bert_config=bert_config,
    		num_labels=len(label_list),
    		init_checkpoint=self.model_address,
    		learning_rate=2e-5,
    		num_train_steps=num_train_steps,
    		num_warmup_steps=num_warmup_steps,
    		use_tpu=False,
    		use_one_hot_embeddings=False)

		estimator = tf.estimator.tpu.TPUEstimator(
    		use_tpu=False,
    		model_fn=model_fn,
    		config=run_config,
    		train_batch_size=32,
    		eval_batch_size=32,
    		predict_batch_size=32)

		predict_examples = processor.get_test_examples(self.path_)
		num_actual_predict_examples = len(predict_examples)
		predict_file = os.path.join("./output", "predict.tf_record")
		file_based_convert_examples_to_features(predict_examples, label_list, 128, tokenizer, predict_file)

		predict_drop_remainder = False

		predict_input_fn = file_based_input_fn_builder(
    		input_file=predict_file,
    		seq_length=128,
    		is_training=False,
    		drop_remainder=predict_drop_remainder)

		result = estimator.predict(input_fn=predict_input_fn)

		output_predict_file = os.path.join("./output", "test_results.tsv")

		with tf.gfile.GFile(output_predict_file, "wb") as writer:
			num_written_lines = 0
			tf.logging.info("***** Predict results *****")
			for (i, prediction) in enumerate(result):
				probabilities = prediction["probabilities"]
				if i >= num_actual_predict_examples:
					break
				max_index = np.argmax(probabilities)
				if self.model_id == 1:
					output_line = self.result_line_two(max_index)
				elif self.model_id == 2:
					output_line = self.result_line_four(max_index)
				writer.write(output_line)
				num_written_lines += 1
		assert num_written_lines == num_actual_predict_examples
		self.tip_text.set("已成功导出")

	def result_line_two(self, max_index):
		if max_index == 1:
			return "正向" + "\n"
		else:
			return "负向" + "\n"
	def result_line_four(self, max_index):
		if max_index == 0:
			return "喜悦" + "\n"
		elif max_index == 1:
			return "愤怒" + "\n"
		elif max_index == 2:
			return "厌恶" + "\n"
		elif max_index == 3:
			return "低落" + "\n"




