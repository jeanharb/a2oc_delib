from nnet import Model, MLP3D
import numpy as np
import sys,pickle,os,theano
import theano.tensor as T
from lasagne.updates import norm_constraint
from collections import OrderedDict

def clip_grads(grads, clip, clip_type):
  if clip > 0.1: 
    if clip_type == "norm":
      grads = [norm_constraint(p, clip) if p.ndim > 1 else T.clip(p, -clip, clip) for p in grads]
    elif clip_type == "global":
      norm = T.sqrt(T.sum([T.sum(T.sqr(g)) for g in grads])*2) + 1e-7
      scale = clip * T.min([1/norm,1./clip]).astype("float32")
      grads = [g*scale for g in grads]
  return grads

def rmsprop(params, grads, clip=0, rho=0.99, eps=0.1, clip_type="norm"):
  grads = clip_grads(grads, clip, clip_type)
  updates = OrderedDict()
  all_grads, rms_weights = [], []
  for param, grad in zip(params, grads):
    acc_rms = theano.shared(param.get_value() * 0)
    rms_weights.append(acc_rms)
    acc_rms_new = rho * acc_rms + (1 - rho) * grad ** 2
    updates[acc_rms] = acc_rms_new
    all_grads.append(grad / T.sqrt(acc_rms_new + eps))
  return updates, all_grads, rms_weights


class AOCAgent_THEANO():
  def __init__(self, num_actions, id_num, shared_arr=None, num_moves=None, args=None):
    print "USING OPTION CRITIC"
    self.args = args
    self.id_num = id_num
    self.num_actions = num_actions
    self.num_moves = num_moves
    self.reset_storing()
    self.rng = np.random.RandomState(100+id_num)
    model_network = [{"model_type": "conv", "filter_size": [8,8], "pool": [1,1], "stride": [4,4], "out_size": 16, "activation": "relu"},
                     {"model_type": "conv", "filter_size": [4,4], "pool": [1,1], "stride": [2,2], "out_size": 32, "activation": "relu"},
                     {"model_type": "mlp", "out_size": 256, "activation": "relu"}]
    out = [None,model_network[-1]["out_size"]]
    self.conv = Model(model_network, input_size=[None,args.concat_frames*(1 if args.grayscale else 3),84,84])
    self.termination_model = Model([{"model_type": "mlp", "out_size": args.num_options, "activation": "sigmoid", "W":0}], input_size=out)
    self.Q_val_model = Model([{"model_type": "mlp", "out_size": args.num_options, "activation": "linear", "W":0}], input_size=out)
    self.options_model = MLP3D(input_size=out[1], num_options=args.num_options, out_size=num_actions, activation="softmax")
    self.params = self.conv.params + self.Q_val_model.params + self.options_model.params + self.termination_model.params
    self.set_rms_shared_weights(shared_arr)

    x = T.ftensor4()
    y = T.fvector()
    a = T.ivector()
    o = T.ivector()
    delib = T.fscalar()

    s = self.conv.apply(x/np.float32(255))
    intra_option_policy = self.options_model.apply(s, o)

    q_vals = self.Q_val_model.apply(s)
    disc_q = theano.gradient.disconnected_grad(q_vals)
    current_option_q = q_vals[T.arange(o.shape[0]), o]
    disc_opt_q = disc_q[T.arange(o.shape[0]), o]
    terms = self.termination_model.apply(s)
    o_term = terms[T.arange(o.shape[0]), o]
    V = T.max(q_vals, axis=1)*(1-self.args.option_epsilon) + (self.args.option_epsilon*T.mean(q_vals, axis=1))
    disc_V = theano.gradient.disconnected_grad(V)

    aggr = T.mean #T.sum
    log_eps = 0.0001

    critic_cost = aggr(args.critic_coef*0.5*T.sqr(y-current_option_q))
    termination_grad = aggr(o_term*((disc_opt_q-disc_V)+delib))
    entropy = -aggr(T.sum(intra_option_policy*T.log(intra_option_policy+log_eps), axis=1))*args.entropy_reg
    pg = aggr((T.log(intra_option_policy[T.arange(a.shape[0]), a]+log_eps)) * (y-disc_opt_q))
    cost = pg + entropy - critic_cost - termination_grad
    
    grads = T.grad(cost*args.update_freq, self.params)
    #grads = T.grad(cost, self.params)
    updates, grad_rms, self.rms_weights = rmsprop(self.params, grads, clip=args.clip, clip_type=args.clip_type)
    self.share_rms(shared_arr)

    self.get_state = theano.function([x], s, on_unused_input='warn')
    self.get_policy = theano.function([s, o], intra_option_policy)
    self.get_termination = theano.function([x], terms)
    self.get_q = theano.function([x], q_vals)
    self.get_q_from_s = theano.function([s], q_vals)
    self.get_V = theano.function([x], V)

    self.rms_grads = theano.function([x,a,y,o, delib], grad_rms, updates=updates, on_unused_input='warn')
    print "ALL COMPILED"

    if not self.args.testing:
      self.init_tracker()
    self.initialized = False

  def update_weights(self, x, a, y, o, moves, delib):
    args = self.args
    self.num_moves.value += moves
    lr = np.max([args.init_lr * (args.max_num_frames-self.num_moves.value)/args.max_num_frames, 0]).astype("float32")

    cumul = self.rms_grads(x,a,y,o,delib)
    for i in range(len(cumul)):
      self.shared_arr[i] += lr*cumul[i]
      self.params[i].set_value(self.shared_arr[i])
    return

  def load_values(self, values):
    assert(len(self.params+self.rms_weights) == len(values))
    for p, v in zip(self.params+self.rms_weights, values): p.set_value(v)

  def save_values(self, folder_name):
    pickle.dump([p.get_value() for p in self.params+self.rms_weights], open(folder_name+"/tmp_model.pkl", "wb"))
    os.system("mv "+folder_name+"/tmp_model.pkl "+folder_name+"/model.pkl")
    #try: # server creates too many core files
    #  os.system("rm ./core*")
    #except:
    #  pass

  def get_param_vals(self):
    return [m.get_value() for m in self.params+self.rms_weights]

  def set_rms_shared_weights(self, shared_arr):
    if shared_arr is not None:
      self.shared_arr = [np.frombuffer(s, dtype="float32").reshape(p.get_value().shape) for s, p in zip(shared_arr, self.params)] 
      self.rms_shared_arr = shared_arr[len(self.params):]
      if self.args.init_num_moves > 0:
        for s, p in zip(shared_arr, self.params):
          p.set_value(np.frombuffer(s, dtype="float32").reshape(p.get_value().shape))
        print "LOADED VALUES"

  def share_rms(self, shared_arr):
    # Ties rms params between threads with borrow=True flag
    if self.args.rms_shared and shared_arr is not None:
      assert(len(self.rms_weights) == len(self.rms_shared_arr))
      for rms_w, s_rms_w in zip(self.rms_weights, self.rms_shared_arr): 
        rms_w.set_value(np.frombuffer(s_rms_w, dtype="float32").reshape(rms_w.get_value().shape), borrow=True)

  def get_action(self, x):
    p = self.get_policy([self.current_s], [self.current_o])
    return self.rng.choice(range(self.num_actions), p=p[-1])

  def get_policy_over_options(self, s):
    return self.get_q_from_s(s)[0].argmax() if self.rng.rand() > self.args.option_epsilon else self.rng.randint(self.args.num_options)

  def update_internal_state(self, x):
    self.current_s = self.get_state([x])[0]
    self.delib = self.args.delib_cost

    if self.terminated:
      self.current_o = self.get_policy_over_options([self.current_s])
      self.o_tracker_chosen[self.current_o] += 1

    self.o_tracker_steps[self.current_o] += 1

  def init_tracker(self):
    csv_things = ["moves", "reward", "term_prob"]
    csv_things += ["opt_chosen"+str(ccc) for ccc in range(self.args.num_options)]
    csv_things += ["opt_steps"+str(ccc) for ccc in range(self.args.num_options)]
    with open(self.args.folder_name+"/data.csv", "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things])+"\n")

  def tracker(self):
    term_prob = float(self.termination_counter)/self.frame_counter*100
    csv_things = [self.num_moves.value, self.total_reward, round(term_prob,1)]+list(self.o_tracker_chosen)+list(self.o_tracker_steps)
    with open(self.args.folder_name+"/data.csv", "a") as myfile:
      myfile.write(",".join([str(cc) for cc in csv_things])+"\n")

  def reset_tracker(self):
    self.termination_counter = 0
    self.frame_counter = 0
    self.o_tracker_chosen = np.zeros(self.args.num_options,)
    self.o_tracker_steps = np.zeros(self.args.num_options,)

  def reset(self, x):
    if not self.args.testing and self.initialized: self.tracker()
    self.total_reward = 0
    self.terminated = True
    self.reset_tracker()
    self.update_internal_state(x)
    self.initialized = True
    
  def reset_storing(self):
    self.a_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
    self.o_seq = np.zeros((self.args.max_update_freq,), dtype="int32")
    self.r_seq = np.zeros((self.args.max_update_freq,), dtype="float32")
    self.x_seq = np.zeros((self.args.max_update_freq, self.args.concat_frames*(1 if self.args.grayscale else 3),84,84),dtype="float32")
    self.t_counter = 0

  def store(self, x, new_x, action, raw_reward, done, death):
    end_ep = done or (death and self.args.death_ends_episode)
    self.frame_counter += 1
    
    self.total_reward += raw_reward
    reward = np.clip(raw_reward, -1, 1)

    self.terminated = self.get_termination([new_x])[0][self.current_o] > self.rng.rand()
    self.termination_counter += self.terminated

    self.x_seq[self.t_counter] = np.copy(x)
    self.o_seq[self.t_counter] = np.copy(self.current_o)
    self.a_seq[self.t_counter] = np.copy(action)
    self.r_seq[self.t_counter] = np.copy(float(reward)) - (float(self.terminated)*self.delib*(1-float(end_ep)))

    self.t_counter += 1

    # do n-step return to option termination. 
    # cut off at self.args.max_update_freq
    # min steps: self.args.update_freq (usually 5 like a3c)
    # this doesn't make option length a minimum of 5 (they can still terminate). only batch size
    option_term = (self.terminated and self.t_counter >= self.args.update_freq)
    if self.t_counter == self.args.max_update_freq or end_ep or option_term:
      if not self.args.testing:
        V = self.get_V([new_x])[0] if self.terminated else self.get_q([new_x])[0][self.current_o]
        R = 0 if end_ep else V
        V = []
        for j in range(self.t_counter-1,-1,-1):
          R = np.float32(self.r_seq[j] + self.args.gamma*R)
          V.append(R)
        self.update_weights(self.x_seq[:self.t_counter], self.a_seq[:self.t_counter], V[::-1], 
                            self.o_seq[:self.t_counter], self.t_counter, self.delib+self.args.margin_cost)
      self.reset_storing()
    if not end_ep:
      self.update_internal_state(new_x)
