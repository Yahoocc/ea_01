import json


class MysqlDatabases:
	def __init__(self):
		with open('sequences.json', mode='r', encoding='utf-8') as f:
			text = f.read()
		self.records = json.loads(text)

	def all(self):
		return self.records

	def insert(self, record):
		self.records.append(record)


if __name__ =='__main__':
	db = MysqlDatabases()