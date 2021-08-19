import tensorflow as tf
import numpy as np

# a=tf.nn.sigmoid(-1.0)
#
# with tf.Session() as sess:
#     print(sess.run(a))

def _sparse_tuple_from(src_seq_list):
    """
    convert a list to sparse tuple
    usage:
        src_seq_list = [[1,2],[1],[1,2,3,4,5],[2,5,2,6]]
        sparse_tensor = sparse_tuple_from(src_seq_list)
        then :
            sparse_tensor[0](indices) = [[0,0],[0,1],[1,0],[2,0],[2,1],[2,2],[2,3],[2,4],[3,0],[3,1],[3,2],[3,3]]
            sparse_tensor[1](values) =  [1,2,1,1,2,3,4,5,2,5,2,6], squeezed src_seq_list's values
            sparse_tensor[2](shape) =  [4,5] , 4: number of sequence; 5: max_length of seq_labels
    """
    indices = []
    values = []
    for n, seq in enumerate(src_seq_list):
        indices.extend(zip([n] * np.shape(seq)[0], seq))
        values.extend(np.ones(np.shape(seq)[0]))

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    shape = np.asarray([np.shape(src_seq_list)[0], np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    print("src_seq_list:",src_seq_list)
    print("indices:",indices)
    print("values:",values)
    print("shape:",shape)
    return indices, values, shape

# a=[[1,2],[3,4],[5]]
# print(_sparse_tuple_from(a))

test="convert"
print(test[0:1])