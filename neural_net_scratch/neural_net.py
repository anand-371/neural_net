
import numpy as np


class neural_net():

	def __init__(self,x,y,nodes,number_hiddenlayers,number_classes,epochs,batch_size):
                
		xo = np.ones(shape = (x.shape[0],1))
		self.y=y
		self.epochs=epochs
		self.x = np.hstack((xo,x))
		self.m = x.shape[0]
		self.n = x.shape[1]
		self.number_nodes=[]
		self.nodes = nodes
		self.number_nodes.append(nodes)
		self.number_nodes.insert(0,self.n)
		self.number_nodes.insert(-1,number_classes)
		self.number_hiddenlayers = number_hiddenlayers
		self.number_classes = number_classes
		self.batch_size=batch_size


	def inialise_weights_manually(self,nnumber_params):
                #user can pre-load a given set of weights into the model
		self.Theta=nnumber_params

		
	def inialise_weights_automatically():
                #to randomly create a matrix of small valued weights
		self.Theta = []
		for i in range(len(self.n_nodes)-1):
			a = np.random.normal(size =(self.n_nodes[i+1],self.n_nodes[i]+1))
			self.Theta.append(a)


			
	def sigmoid(self,z):
		return 1/(1+np.exp(-z))
	
	
	
	def gradient_sigmoid(self,z):
		return (self.sigmoid(z)*(1-self.sigmoid(z)))

	

	def feedforward(self):
		zl = []
		a=self.bx
		
		for i in range(self.number_hiddenlayers):
                        
			z = np.dot(a,np.transpose(self.Theta[i]))
			zl.append(z)
			a = self.sigmoid(z)
			ao = np.ones(shape = (a.shape[0],1))
			a = np.hstack((ao,a))

		z = np.dot(a,np.transpose(self.Theta[-1]))
		return z,zl
	
	
	def sigmoid_cross_entropy(self):
		z,_ = self.feedforward()
		a = self.sigmoid(z)
		self.cost = (-1/self.batch_size)*np.sum(self.by*np.log(a) + (1-self.by)*np.log(1 - a))
		return(self.cost)


	def cost_gradient(self,z):
                
			a = self.sigmoid(z)
			return (a - self.by)
		
        def sigmoid_squared_error(self):
		z,zl = self.feedforward()
		a = self.sigmoid(z)
		self.cost = (1/(2*self.batch_size))*np.sum(self.by-a)**2
		return(self.cost)


	def backprop(self):
		d = []
		z,zl = self.feedforward()
		a = self.sigmoid(z)
		d.append(self.cost_gradient(z))
		self.Theta_gradient = [np.zeros(shape = t.shape) for t in self.Theta]
		
		for i in range(self.number_hiddenlayers):
                        
			a_grad = self.gradient_sigmoid(zl[::-1][i])
			d.append(np.dot(d[i],self.Theta[::-1][i])[:,1:]*a_grad)

		d=d[::-1]
		self.Theta_gradient[0] = self.Theta_gradient[0] + np.dot(np.transpose(d[0]),self.bx)
		
		for i in range(1,len(self.Theta_gradient)):
                        
			a = self.sigmoid(zl[i-1])
			ao = np.ones(shape = (a.shape[0],1))
			a = np.hstack((ao,a))
			self.Theta_gradient[i] = self.Theta_gradient[i] + np.dot(np.transpose(d[i]),a)

		self.Theta_gradient = [(1/self.batch_size)*k for k in self.Theta_gradient]



	def master(self):
		j=0
		z=0
		while z<=self.epochs:
                        
			self.bx = self.x[j:j+self.batch_size]
			self.by = self.y[j:j+self.batch_size]
			print("cost calculated is ",self.sigmoid_cross_entropy())
			self.backprop()

			for i in range(len(self.Theta)):
                                
				self.Theta[i] = self.Theta[i] - (alpha*self.Theta_gradient[i])
				
			i=i+self.batch_size
			
			if (i+self.batch_size>self.m):
                                
				break	
			z+=1
		print('!!!!!!!done!!!!!!')	


	def prediction(self,inputuser):
                
		z,zl = self.feedforward(inputuser)
		result = np.argmax(z)
		print(result)

	"""
order in which functions of class are to be called
--------------------------------------------------------------------------------
1.x,y- passing the data to the class along with
2.nodes-desired number of nodes
3.number of hidden layers-required number of hidden layers of the neural net
4.number of labels- total number of labels that the net is expected to output
5.epochs- number of times one feedforward and backpropagation are to be run
6.batch_size-in this neural net a mini batch gradient descent approach is applied the user can prove his
             batch_size.
---------------------------------------------------------------------------------
the user can choose in applying either:
1.sigmoid cross entropy
2.sigmoid square error
each of these can be used by creating an instance of class
and using the function defined
by sigmoid_cross_entropy and
sigmoid_squared_error ...
all the training of the net happens in a function called master.
to write other activation the following class can be inherited
and modified upon.
----------------------------------------------------------------------------------
once trained the net can be used by calling the prediction function
"""

