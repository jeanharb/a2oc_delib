from multiprocessing import Process, Value, Array, RawArray
from collections import OrderedDict
import numpy as np
from OC_theano import AOCAgent_THEANO
import cv2,copy,sys,pickle,os,time,argparse
from PIL import Image
from utils.helper import foldercreation, str2bool, get_folder_name

class Environment():
  def reset(self):
    raise NotImplementedError

  def render(self):
    raise NotImplementedError

  def act(self):
    raise NotImplementedError

  def get_frame_count(self):
    raise NotImplementedError

class ALE_env(Environment):
  def __init__(self, args, rng=None):
    import gym
    env = gym.make(args.sub_env+"NoFrameskip-v4")
    self.args = args
    self.rng = rng
    self.env = env
    self.action_space = self.env.action_space.n
    self.obs_space = self.env.observation_space.shape

    if self.args.testing:
      import matplotlib.pyplot as plt
      plt.ion()
      plt.show(block=False)

  def get_lives(self):
    return self.env.unwrapped.ale.lives()

  def noops(self):
    num_actions = self.rng.randint(1, self.args.max_start_nullops)
    for i in range(np.max([num_actions//self.args.frame_skip, self.args.concat_frames])):
      self.act(0)
    if self.env.unwrapped.get_action_meanings()[1] == 'FIRE':
      self.act(1)

  def reset(self):
    self.current_x = np.zeros((self.args.concat_frames*(1 if self.args.grayscale else 3), 84, 84), dtype="float32")
    self.new_obs = self.env.reset()
    self.lives = self.get_lives()
    self.noops()
    return self.current_x

  def render(self):
    a = 2
    if a == 1: #can see what the agent sees
      import matplotlib.pyplot as plt
      plt.clf()
      if self.args.grayscale:
        plt.imshow(self.xx, cmap="Greys_r")
      else:
        x = np.swapaxes(self.xx, 0,2)
        x_ = np.copy(x[0])
        x[0] = x[2]
        x[2] = x_
        plt.imshow(x)
      plt.draw()
      plt.pause(0.0001)
    else:
      self.env.render()

  def get_new_frame(self, new_frame):
    a = 1 if self.args.grayscale else 3
    self.current_x[:-a] = self.current_x[a:]
    self.current_x[-a:] = new_frame

  def act(self, action):
    raw_reward, dones, done = 0, 0, False
    for i in range(self.args.frame_skip):
      if done:
        break
      new_obs, rew, done, info = self.env.step(action)
      self.old_obs = np.copy(self.new_obs)
      self.new_obs = new_obs
      raw_reward += rew
      dones += done
    new_frame = self.preprocess(self.new_obs, self.old_obs)
    self.get_new_frame(new_frame)
    dones += (self.get_frame_count() > self.args.max_frames_ep)
    
    new_lives = self.get_lives()
    death = new_lives < self.lives
    self.lives = new_lives
    if death and not bool(int(dones)):
      self.noops()
    return self.current_x, raw_reward, bool(int(dones)), death

  def preprocess(self, im, last_im):
    if self.args.color_max:
      im = np.maximum(im, last_im)
    if self.args.grayscale:
      proportions = [0.299, 0.587, 0.114]
      im = np.sum(im * proportions, axis=2)
    #im = cv2.resize(im, (84, 110), interpolation=cv2.INTER_AREA)[18:102, :]
    im = Image.fromarray(im).resize((84, 84), resample=Image.BILINEAR)
    x = np.array(im).astype("int32")
    if not self.args.grayscale:
      x = np.swapaxes(x, 0, 2)
    self.xx = x
    return x

  def get_frame_count(self):
    return self.env.unwrapped.ale.getEpisodeFrameNumber()

class Training():
  def __init__(self, rng, id_num, arr, num_moves, args):
    self.args = args
    self.rng = rng
    self.num_moves = num_moves
    self.id_num = id_num

    self.env = ALE_env(args, rng=rng)
    self.agent = AOCAgent_THEANO(self.env.action_space, id_num, arr, num_moves, args)

    self.train()

  def train(self):
    total_reward = 0
    x = self.env.reset()
    self.agent.reset(x)
    timer = time.time()
    recent_fps = []
    frame_counter = 0
    total_games = 0
    done = False

    while self.num_moves.value < self.args.max_num_frames:
      if done:
        #ugly code, beautiful print
        total_games += 1
        secs = round(time.time()-timer, 1)
        frames = self.env.get_frame_count()
        fps = int(frames/secs)
        recent_fps = recent_fps[-9:]+[fps]
        eta = ((self.args.max_num_frames-self.num_moves.value)*self.args.frame_skip/(self.args.num_threads*np.mean(recent_fps)))
        print "id: %d\treward: %d\ttime: %.1f\tframes: %d\t %dfps  \tmoves: %d \t ETA: %dh %dm %ds  \t%.2f%%" % \
        (self.id_num, total_reward, secs, frames, fps, self.num_moves.value, int(eta/3600), int(eta/60)%60, int(eta%60), 
          float(self.num_moves.value)/self.args.max_num_frames*100)
        timer = time.time()
        frame_counter = 0

        if total_games % 1 == 0 and self.id_num == 1 and not self.args.testing:
          self.agent.save_values(folder_name)
          print "saved model"
        total_reward = 0
        x = self.env.reset()
        self.agent.reset(x)
        done = False

      action = self.agent.get_action(x)
      new_x, reward, done, death = self.env.act(action)
      self.agent.store(x, new_x, action, reward, done, death)
      if self.args.testing:
        self.env.render()
      total_reward += reward
      x = np.copy(new_x)

def parse_params():
  parser = argparse.ArgumentParser()
  parser.add_argument('--sub-env', type=str, default="Breakout")
  parser.add_argument('--testing', type=str2bool, default=False)
  parser.add_argument('--update-freq', type=int, default=5)
  parser.add_argument('--max-update-freq', type=int, default=30)
  parser.add_argument('--num-threads', type=int, default=16)
  parser.add_argument('--death-ends-episode', type=str2bool, default=True)
  parser.add_argument('--max-start-nullops', type=int, default=30)
  parser.add_argument('--frame-skip', type=int, default=4)
  parser.add_argument('--concat-frames', type=int, default=4)
  parser.add_argument('--entropy-reg', type=float, default=0.01)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--clip', type=float, default=40)
  parser.add_argument('--clip-type', type=str, default="global", choices=["norm", "global"])
  parser.add_argument('--color-averaging', type=str2bool, default=False)
  parser.add_argument('--color-max', type=str2bool, default=True)
  parser.add_argument('--grayscale', type=str2bool, default=True)
  parser.add_argument('--max-num-frames', type=int, default=80000000)
  parser.add_argument('--max-frames-ep', type=int, default=72000)
  parser.add_argument('--init-lr', type=float, default=0.0007)
  parser.add_argument('--rms-shared', type=str2bool, default=True)
  parser.add_argument('--critic-coef', type=float, default=1.)
  parser.add_argument('--num-options', type=int, default=8)
  parser.add_argument('--option-epsilon', type=float, default=0.1)
  parser.add_argument('--delib-cost', type=float, default=0.0)
  parser.add_argument('--margin-cost', type=float, default=0.0)
  parser.add_argument('--save-path', type=str, default="models") 
  parser.add_argument('--load-folder', type=str, default="") # if not empty, will load folder to resume training
  parser.add_argument('--folder-name', type=str, default="")
  parser.add_argument('--resume-if-exists', type=str2bool, default=False) # for server that kills and restarts processes
  return parser.parse_known_args()[0] #parser.parse_args()


if __name__ == '__main__':
  params = parse_params()

  folder_name = get_folder_name(params) if params.folder_name == "" else params.folder_name
  attempted_path = "./"+params.save_path+"/"+folder_name
  print "->", attempted_path, os.path.isdir(attempted_path)
  if params.resume_if_exists and os.path.isdir(attempted_path):
    params.load_folder = attempted_path
    print "RESUMING TRAINING AUTOMATICALLY"

  init_num_moves = 0
  if params.load_folder != "":
    folder_name = params.load_folder
    with open(folder_name+"/data.csv", "rb") as file:
      for last in file:
        if last.split(",")[0].isdigit():
          init_num_moves = int(last.split(",")[0])
    init_weights = pickle.load(open(folder_name+"/model.pkl", "rb"))
    is_testing = copy.deepcopy(params.testing)
    params = pickle.load(open(params.load_folder+"/params.pkl", "rb"))
    params.testing = is_testing
    if is_testing:
      params.num_threads = 1
  else:
    folder_name = foldercreation(folder_name, params)
    pickle.dump(params, open(folder_name+"/params.pkl", "wb"))

  setattr(params, "folder_name", folder_name)

  setattr(params, "init_num_moves", init_num_moves)
  print "init_num_moves:", init_num_moves

  f = lambda rng, i, shared_arr, num_moves, args: Training(rng, i, shared_arr, num_moves, args)

  env = ALE_env(params)
  if init_num_moves == 0:
    init_weights = (AOCAgent_THEANO(env.action_space, 0, args=params)).get_param_vals()
    
  num_moves = Value("i", init_num_moves, lock=False)
  arr = [Array('f', m.flatten(), lock=False) for m in init_weights]
  seed = np.random.randint(10000)
  for i in range(params.num_threads):
    Process(target=f, args=(np.random.RandomState(seed+i), i+1, arr, num_moves, params)).start()


