import paramiko

class Conrtol:

	def __init__(self, host, proxy_host, user, password):
		self.client = paramiko.SSHClient()
		self.client.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
		self.proxy_client = paramiko.SSHClient()
		self.proxy_client.set_missing_host_key_policy(paramiko.MissingHostKeyPolicy())
		self.host = host
		self.proxy_host = proxy_host
		self.user = user
		self.password = password

	def connect(self):
		try:
			self.proxy_client.connect(self.proxy_host, username=self.user, password=self.password)
		except:
			raise Exception("Problem in Connecting to Proxy %s", self.proxy_host)

		try:
			self.proxy_channel = self.proxy_client.get_transport().open_channel('direct-tcpip',(self.host, 22, ),('127.0.0.1',0))
		except:
			raise Exception("Error connecting to host channel '%s:%s'", self.host, '22')	

		try:
			self.client.connect(self.host, username=self.user, password=self.password, sock=self.proxy_channel)
		except:
			raise Exception("Error connecting to host '%s:%s'", self.host, '22')
	def run_command(self):
		stdin, stdout, stderr = self.client.exec_command('for i in {1..5}; do echo "$i";sleep 3; done')
		for line in stdout:
			output = line.strip().decode('utf8')
			print output

tmp1 = Conrtol('prakt01', 'ssh.lis.ei.tum.de','', '')
tmp1.connect()
tmp1.run_command()