import numpy as np
import struct


def write_small(file_name='_embedding'):
    emb = [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
    emb = np.array(emb, dtype=np.float32)
    buf = struct.pack('iIQ', 0, 4, 2 * 3)
    buf += emb.astype('float32').tostring()
    with open(file_name, 'wb') as f:
        f.write(buf)

def write_middle(file_name='_embedding_mid'):
    dim = 24
    seq_len = 10
    emb = [[i] * dim for i in xrange(seq_len)]
    emb = np.array(emb, dtype=np.float32)
    buf = struct.pack('iIQ', 0, 4, dim * seq_len)
    buf += emb.astype('float32').tostring()
    with open(file_name, 'wb') as f:
        f.write(buf)

if __name__ == '__main__':
   write_middle(file_name='_embedding_mid')
