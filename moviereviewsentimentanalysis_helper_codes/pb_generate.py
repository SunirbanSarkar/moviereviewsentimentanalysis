# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:21:31 2020

@author: sunir
"""

import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.tools import optimize_for_inference_lib
import fire
from elapsedtimer import ElapsedTimer

def model_freeze(path,MODEL_NAME='model'):
    output_frozen_graph_name = path + 'frozen_'+MODEL_NAME+'.pb'
    output_optimized_graph_name = path + 'optimized_'+MODEL_NAME+'.pb'
    with open(path+'model.pbtxt') as f:
      txt = f.read()
    gdef = text_format.Parse(txt, tf.compat.v1.GraphDef())
    
    tf.io.write_graph(gdef, 'D:/DEVELOPEMENT/moviereviewsentimentanalysis_helper_codes/datset', 
                         'frozen_model.pb', as_text=False)

    with tf.io.gfile.GFile(output_frozen_graph_name, "rb") as f:
        data = f.read()
        tf.compat.v1.GraphDef().ParseFromString(data)

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            tf.compat.v1.GraphDef(),
            ["inputs/X" ],#an array of the input node(s)
            ["positive_sentiment_probability"],
            tf.int32.as_datatype_enum # an array of output nodes
            )

    # Save the optimized graph

    f = tf.compat.v1.gfile.FastGFile(output_optimized_graph_name, "w")
    f.write(output_graph_def.SerializeToString())

if __name__ == '__main__':
    with ElapsedTimer('Model Freeze'):
        fire.Fire(model_freeze)