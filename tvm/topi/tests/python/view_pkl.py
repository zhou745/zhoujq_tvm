import cPickle as pickle  
f=open('.pkl_memoize_py2/topi.tests.test_topi_conv2d_int8.verify_conv2d_nchw.get_ref_data.pkl')
info=pickle.load(f)
print(info)
