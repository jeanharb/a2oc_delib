import os, datetime, pickle as pkl

def create_dir(p, num_suffix=False):
  i = 0
  while True:
    try:
      new_dir = p+(("_v"+str(i) if i > 0 else "") if num_suffix else "")
      os.makedirs(new_dir)
      break
    except OSError, e:
      if e.errno != 17:
        raise # This was not a "directory exist" error..
      else:
        i += 1
        if not num_suffix:
          break
  return new_dir

def get_folder_name(args):
  folder_name = args.sub_env+"_"+str(args.num_options)+"opts_"+str(args.delib_cost)+"delib_"+ \
            str(args.num_threads)+"_"+str(args.max_num_frames//80000000)+"day"
  return folder_name

def foldercreation(folder_name, args):
  tempdir = os.path.join(os.getcwd(), args.save_path)
  create_dir(tempdir)
  #folder_name = folder_name if folder_name is not None else datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
  mydir = os.path.join(tempdir, folder_name)
  return create_dir(mydir, num_suffix=True)

def str2bool(v):
  if v.lower() not in ("yes", "true", "t", "1", "no", "false", "f", "0"):
    print "Inserted unrecognized string for bool value. Must be one of the following:"
    print " ".join(["yes", "true", "t", "1", "no", "false", "f", "0"])
    print "Note: Capitalization doesn't matter."
    raise NotImplementedError
  return v.lower() in ("yes", "true", "t", "1")
