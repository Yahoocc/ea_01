import tkinter as tk
from Views import AboutFrame, BatchFrame, RecognizeFrame, RecordFrame, SettingFrame

class MainPage:
	def __init__(self, master:tk.Tk):
		self.root = master
		self.root.title('中文情感识别系统')
		self.root.geometry('600x360')
		self.create_page()
		self.show_recognize()

	def create_page(self):
		menubar = tk.Menu(self.root)
		file_menu = tk.Menu(menubar, tearoff=False)
		file_menu.add_command(label='情感识别', command=self.show_recognize)
		file_menu.add_command(label='识别记录', command=self.show_record)
		file_menu.add_command(label='批量识别', command=self.show_batch)
		file_menu.add_command(label='设置', command=self.show_setting)
		file_menu.add_command(label='关于', command=self.show_about)
		menubar.add_cascade(label='更多', menu=file_menu)
		self.root['menu'] = menubar

		self.setting_frame = SettingFrame(self.root)
		self.recognize_frame = RecognizeFrame(self.root, self.setting_frame)
		self.batch_frame = BatchFrame(self.root, self.setting_frame)

		self.about_frame = AboutFrame(self.root)
		self.record_frame = RecordFrame(self.root)
		
	def show_about(self):
		self.about_frame.pack()
		self.recognize_frame.pack_forget()
		self.batch_frame.pack_forget()
		self.record_frame.pack_forget()
		self.setting_frame.pack_forget()

	def show_recognize(self):
		self.recognize_frame.pack()
		self.about_frame.pack_forget()
		self.batch_frame.pack_forget()
		self.record_frame.pack_forget()
		self.setting_frame.pack_forget()

	def show_batch(self):
		self.batch_frame.pack()
		self.about_frame.pack_forget()
		self.recognize_frame.pack_forget()
		self.record_frame.pack_forget()
		self.setting_frame.pack_forget()

	def show_record(self):
		self.record_frame.pack()
		self.about_frame.pack_forget()
		self.recognize_frame.pack_forget()
		self.batch_frame.pack_forget()
		self.setting_frame.pack_forget()

	def show_setting(self):
		self.setting_frame.pack()
		self.record_frame.pack_forget()
		self.about_frame.pack_forget()
		self.recognize_frame.pack_forget()
		self.batch_frame.pack_forget()


if __name__ == '__main__':
	root = tk.Tk()
	MainPage(root)
	root.mainloop()