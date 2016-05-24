import h5py
import os

class File(h5py.File):
   def __init__(self,name,mode=None,driver=None,**kwds):
      """
      Simple extension to h5py.File

      Additional mode 'c' will 

      """
      if mode == 'c':
         if os.path.exists(name):
            os.remove(name)
            print 'removing %s ' % name
         mode = 'a'

      h5py.File.__init__(self,name,mode=mode,driver=driver,**kwds)
      
   def __setitem__(self,key,val):
      if self.keys().count(unicode(key)) is 1:
         print "removing %s " % key
         del self[key]      
      h5py.File.__setitem__(self,key,val)

   def create_group(self,name):
      try:
         group = h5py.File.create_group(self,name)
      except ValueError:
         del self[name]
         group = h5py.File.create_group(self,name)
      return group

   def group2dict(self,name):
      """
      Retrive scalar datasets in a group.
      """
      d = {}
      for n in self[name].keys():
         d[n] = self[name][n][()]
      return d

   def dict2group(self,name,d):
      """
      Take all the values in a dictionary and shove them into a group
      """
      for n in d.keys():
         self[name][n] = d[n]


def dict_to_attrs(h5,d):
   for k in d.keys():
      h5.attrs[k] = d[k]


def copy_attrs(h5path0,h5path):
   """
   Copy top-level attributes from h5path0 to h5path

   Parameters
   ----------
   h5path0 : original h5 file
   h5path : new h5file
   """
   with h5py.File(h5path0,'r') as h50:
      with h5py.File(h5path) as h5:
         dict_to_attrs(h5,dict(h50.attrs))
