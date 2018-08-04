from multiprocessing import Process, Pipe
import numpy as np


def writeToConnection(conn):
	conn.send(np.ones(3))
	conn.close()


if __name__ == '__main__':

	recv_conn, send_conn = Pipe(duplex=False)

	p = Process(target=writeToConnection, args=(send_conn,))
	p.start()
	print(recv_conn.recv())
	p.join()


	recv_conn, send_conn = Pipe(duplex=False)
	send_conn.send('hello')
	res = recv_conn.recv()
	print(res)
