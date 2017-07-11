import numpy as np
import struct

emb = [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
emb = np.array(emb, dtype=np.float32)
buf = struct.pack('iIQ', 0, 4, 2 * 3)
buf += emb.astype('float32').tostring()
with open('_embedding', 'wb') as f:
    f.write(buf)

