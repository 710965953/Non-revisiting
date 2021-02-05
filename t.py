class A():
	def __init__(self):
		pass
	def a(self):
		print(self.__class__.__name__)
	
class B(A):
	def __init__(self):
		pass

a = B()
a.a()
